#!/usr/bin/env python3
"""MHC Post benchmark 报告生成器（合并版）。

合并自 bench_report_us.py 与 gen_mhc_bench_tables.py，输出单个 markdown 报告，
包含：① 实测 latency 微秒换算表；② tileops vs baseline 对比信息；③ 资源与
性能分析表（block / shared memory / HBM 访问 / 计算次数 / 带宽利用率）。

与旧脚本的差异（可读性优化）：
- 全部数据来自 profile_run.log 解析，无硬编码 case（旧 gen 脚本手抄 15 行）；
- 同一份解析结果驱动三个章节，数值口径唯一；
- 表格渲染统一走 render_table()，指标计算集中 analyze_case()；
- 分析口径常量集中声明，并与 tileops/kernels/mhc/mhc_post.py 的实现同步注释。

硬件范围：本脚本专用于 MetaX C500 benchmark 报告。1843 GB/s HBM 峰值、
32 B 最小访存事务和 64 KB/block shared memory 上限等 C500 硬件常量均为
有意绑定，不作为跨硬件通用接口。

用法示例：
  # 在仓库内直接运行：先检查 MHC Post 正确性，再执行 benchmark 并生成报告
  python3 scripts/mhc_bench_report.py

  # 只处理已有报告（--csv 可选输出分析宽表）
  python3 scripts/mhc_bench_report.py \
      --input profile_run.log \
      --output mhc_post_bench_report.md --csv mhc_post_bench_report.csv

默认报告写入项目根目录的 mhc_post_bench_report.md。自动定位不满足时，
仍可用 --cwd、--run 和 --output 覆盖默认行为。

设计约束：父进程只依赖标准库且不 import tilelang，避免报告进程提前加载
TileLang/C500 runtime；测试与 benchmark 均由独立 pytest 子进程执行。
"""

import argparse
import ast
import csv
import re
import shlex
import subprocess
import sys
from pathlib import Path

# ─────────────────────────── 常量（实现口径）───────────────────────────
# 与 tileops/kernels/mhc/mhc_post.py 同步：
#   - _ALL_N4_MIN_BATCH = 16：n_expand==4 且 batch>=16 走 all-n4 2D kernel
#   - block_x_b 实际取 min(block_x_b, batch)
#   - shared buffer 全 float32（h_post / x_layer_out）
#   - x_res / x_out 直接读写 HBM（bf16），不经 shared
ALL_N4_MIN_BATCH = 16
HBM_PEAK_GBS = 1843.0            # C500 报告固定使用的理论 DRAM-L2 带宽
DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4, "float64": 8}
FP32_BYTES = 4
MHC_POST_OP_NAME = "MHCPostOp"
DEFAULT_CHECK = "tests/ops/test_mhc.py -k mhc_post -v"
DEFAULT_RUN = "benchmarks/ops/bench_mhc.py::test_mhc_post_bench"
DEFAULT_OUTPUT_NAME = "mhc_post_bench_report.md"

# BenchmarkReport.dump 的数值列识别（result keys 口径）
_NUMERIC_COL_RE = re.compile(r"latency|tflops|bandwidth|gbps|flops|bytes|speedup", re.I)
# C500 访存最小粒度（用于事务数估算）
HBM_MIN_TRANSACTION = 32

HEADER_NOTE = (
    "> 本报告由 MHC Post benchmark 报告生成器生成：正文 latency 已换算为微秒 (µs)；"
    "含 tileops vs baseline 对比与资源/性能分析。\n"
)


# ─────────────────────────── log 解析（markdown → 结构）───────────────────────────
def _cells(line: str) -> list[str]:
    """拆一行 markdown 表格为单元格（去掉首尾 | 与空白）。"""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_report(text: str) -> dict:
    """解析 markdown 报告为 {op: {tag: [row]}}。

    row = {"params": {参数名: 值}, "config": 配置串|None, <数值列名>: float|None}。
    数值列按列名正则识别（latency/tflops/...），其余普通列归入 params。
    """
    report: dict = {}
    op: str | None = None
    tag: str | None = None
    headers: list[str] = []
    numeric_cols: set[int] = set()

    for line in text.splitlines():
        s = line.strip()
        if s.startswith("### "):
            tag = s[4:].strip()
            headers, numeric_cols = [], set()
            if op is not None:
                report[op].setdefault(tag, [])
            continue
        if s.startswith("## "):
            op = s[3:].strip()
            tag = None
            headers, numeric_cols = [], set()
            report.setdefault(op, {})
            continue
        if op is None or tag is None or not s.startswith("|"):
            continue

        cells = _cells(s)
        if not headers:
            headers = cells
            numeric_cols = {i for i, c in enumerate(cells) if _NUMERIC_COL_RE.search(c)}
            continue
        if all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
            continue

        row: dict = {"params": {}, "config": None}
        for i, cell in enumerate(cells):
            if i >= len(headers):
                continue
            name = headers[i]
            if name == "config":
                row["config"] = cell
            elif i in numeric_cols:
                try:
                    row[name] = float(cell)
                except ValueError:
                    row[name] = None
            else:
                row["params"][name] = cell
        report[op][tag].append(row)
    return report


# ─────────────────────────── 正文：µs 换算 ───────────────────────────
def convert_report(text: str) -> str:
    """把表格的 latency_ms 列换算为微秒（表头改名 latency_us，数值 ×1000，2 位小数）。"""
    col_idx: int | None = None
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            col_idx = None
            out.append(line)
            continue

        cells = _cells(stripped)
        if "latency_us" in cells:  # 已是微秒换算后的报告，幂等跳过
            out.append(line)
            continue
        if "latency_ms" in cells:  # 任何含该表头的行都刷新列位置
            col_idx = cells.index("latency_ms")
        if col_idx is None or col_idx >= len(cells):
            out.append(line)
            continue

        if cells[col_idx] == "latency_ms":  # 表头改名
            cells[col_idx] = "latency_us"
            out.append("| " + " | ".join(cells) + " |")
        elif all(re.fullmatch(r":?-{3,}:?", c) for c in cells):  # 分隔行
            out.append(line)
        else:  # 数据行换算
            try:
                val = float(cells[col_idx])
                cells[col_idx] = f"{val * 1000:.2f}"
            except ValueError:
                pass  # N/A 或非数值，原样
            out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _normalize_latency(row: dict) -> None:
    """兼容两种输入：原始报告（latency_ms）或已换算报告（latency_us）。

    统一归一化为毫秒口径（latency_ms），供换算/对比/分析三章节共用。
    """
    if row.get("latency_ms") is None and row.get("latency_us") is not None:
        row["latency_ms"] = row["latency_us"] / 1000


# ─────────────────────────── 章节：对比信息 ───────────────────────────
def _fmt(val, ndigits: int = 2) -> str:
    """数值格式化；None 显示 N/A。"""
    return "N/A" if val is None else f"{val:.{ndigits}f}"


def build_comparison(report: dict) -> str:
    """每 op 生成对比表：params | tileops_us | <baseline>_us | <baseline>_speedup | tflops。"""
    sections = ["## 对比信息（latency 单位 µs；speedup = baseline / tileops，>1 表示 tileops 更快）", ""]
    for op, tags in report.items():
        tl_rows = tags.get("tileops")
        base_tags = [t for t in tags if t != "tileops"]
        if not tl_rows or not base_tags:
            continue

        def _key(r: dict) -> tuple:
            return tuple(sorted(r.get("params", {}).items()))

        by_key: dict = {}
        for t, rows in tags.items():
            for r in rows:
                by_key.setdefault(_key(r), {})[t] = r

        param_names = sorted({k for rows in tags.values() for r in rows for k in r.get("params", {})})
        headers = (param_names + ["tileops_us"] + [f"{t}_us" for t in base_tags]
                   + [f"{t}_speedup" for t in base_tags]
                   + ["tileops_tflops"] + [f"{t}_tflops" for t in base_tags])
        body = []
        for key, rows in by_key.items():
            tlr = rows.get("tileops")
            if tlr is None:
                continue
            tl_ms, tl_tf = tlr.get("latency_ms"), tlr.get("tflops")
            row = [str(tlr["params"].get(p, "")) for p in param_names]
            row.append(_fmt(tl_ms * 1000 if tl_ms is not None else None))
            for t in base_tags:
                br = rows.get(t)
                b_ms = br.get("latency_ms") if br else None
                row.append(_fmt(b_ms * 1000 if b_ms is not None else None))
                row.append(_fmt(b_ms / tl_ms if tl_ms and b_ms else None, 3) + ("x" if tl_ms and b_ms else ""))
            row.append(_fmt(tl_tf))
            for t in base_tags:
                br = rows.get(t)
                row.append(_fmt(br.get("tflops") if br else None))
            body.append(row)
        sections.append(f"### {op}")
        sections.append("")
        sections.append(render_table(headers, body))
        sections.append("")
    return "\n".join(sections)


# ─────────────────────────── 章节：资源与性能分析 ───────────────────────────
def _parse_config(cfg_str: str | None) -> dict:
    """解析 config 列（str(dict) 形式）为 dict；失败返回空。"""
    if not cfg_str:
        return {}
    try:
        cfg = ast.literal_eval(cfg_str)
        return cfg if isinstance(cfg, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def analyze_case(row: dict) -> dict | None:
    """按 mhc_post.py 实现口径计算单个 tileops case 的资源/性能指标。

    返回 dict（键与输出列一致）；无法解析（缺 config / 缺 n_expand 等）返回 None。
    """
    p = row.get("params", {})
    try:
        batch, n, c_x = int(p["batch"]), int(p["n_expand"]), int(p["c_x"])
    except (KeyError, TypeError, ValueError):
        return None
    cfg = _parse_config(row.get("config"))
    bxb, block_C, threads = (int(cfg.get(k, 1)) for k in ("block_x_b", "block_C", "threads"))
    latency_ms = row.get("latency_ms")
    if latency_ms is None:
        return None

    bxb = min(bxb, batch)                      # kernel 内裁剪
    all_n4 = (n == 4 and batch >= ALL_N4_MIN_BATCH)  # _select_mhc_post_kernel 规则
    grid_x = (batch + bxb - 1) // bxb
    grid_y = (c_x + block_C - 1) // block_C

    if all_n4:  # 2D grid：n 在 block 内循环，x_layer_out 读 1 倍
        grid, blocks = (grid_x, grid_y), grid_x * grid_y
        shared = bxb * (n * FP32_BYTES + block_C * FP32_BYTES)
        out_block = bxb * n * block_C
        xlo_reads = batch * c_x
    else:       # 3D grid：(batch, n, c_x) 三维切分，x_layer_out 读放大 n 倍
        grid, blocks = (grid_x, n, grid_y), grid_x * n * grid_y
        shared = bxb * (FP32_BYTES + block_C * FP32_BYTES)
        out_block = bxb * block_C
        xlo_reads = batch * c_x * n

    h_reads = batch * n * grid_y               # h_post 被 grid_y 个 block 重复读
    xres_reads = xout_writes = batch * n * c_x
    total = xlo_reads + h_reads + xres_reads + xout_writes
    hbm_bytes = xlo_reads * DTYPE_BYTES.get(p.get("dtype", "bfloat16"), 2) \
        + h_reads * FP32_BYTES \
        + (xres_reads + xout_writes) * DTYPE_BYTES.get(p.get("dtype", "bfloat16"), 2)
    fma = batch * n * c_x
    latency_s = latency_ms * 1e-3
    return {
        "case": f"{batch},{n},{c_x}",
        "_params_key": tuple(sorted(p.items())),  # 供 speedup 回填匹配
        "kernel": "all-n4 2D" if all_n4 else "generic 3D",
        "config": f"{bxb},{block_C},{threads}",
        "grid": grid, "blocks": blocks, "shared_bytes": shared,
        "out_per_block": out_block, "out_per_thread": out_block / threads,
        "xlo_reads": xlo_reads, "h_post_reads": h_reads,
        "x_res_reads": xres_reads, "x_out_writes": xout_writes,
        "total_hbm_accesses": total, "hbm_bytes": hbm_bytes,
        "fma": fma, "real_flops": 2 * fma,
        "latency_us": latency_ms * 1000,
        "tflops": row.get("tflops"), "speedup": None,
        "actual_bw_gbs": hbm_bytes / latency_s / 1e9,
        "bw_util_pct": hbm_bytes / latency_s / 1e9 / HBM_PEAK_GBS * 100,
        "_dtype_bytes": DTYPE_BYTES.get(p.get("dtype", "bfloat16"), 2),
    }


def _fill_speedup(report: dict, rows: list[dict]) -> None:
    """从 baseline 行补齐 speedup（baseline / tileops，按 params 匹配）。

    同时写入 report 原始行与 analyze_case 返回的分析行（两者是独立 dict）。
    """
    for tags in report.values():
        tl_rows = {tuple(sorted(r["params"].items())): r for r in tags.get("tileops", [])}
        for t, brs in tags.items():
            if t == "tileops":
                continue
            for br in brs:
                key = tuple(sorted(br["params"].items()))
                tlr = tl_rows.get(key)
                if tlr is None or not br.get("latency_ms") or not tlr.get("latency_ms"):
                    continue
                speedup = br["latency_ms"] / tlr["latency_ms"]
                tlr["speedup"] = speedup
                for r in rows:  # 回填独立分析行
                    if r.get("_params_key") == key:
                        r["speedup"] = speedup


def fmt_small(v: float) -> str:
    """数值自适应显示：小值（<0.01）保留 4 位小数，其余 2 位。"""
    return f"{v:.4f}" if abs(v) < 0.01 else f"{v:.2f}"


def fmt_int(v: int) -> str:
    if v >= 1e9:
        return f"{v / 1e9:.3f}G"
    if v >= 1e6:
        return f"{v / 1e6:.2f}M"
    if v >= 1e3:
        return f"{v / 1e3:.1f}K"
    return str(v)


def fmt_bytes(b: int) -> str:
    if b >= 1 << 20:
        return f"{b / 1048576:.2f} MB"
    if b >= 1 << 10:
        return f"{b / 1024:.1f} KB"
    return f"{b} B"


def build_analysis(report: dict, rows: list[dict]) -> str:
    """生成分析章节：表 1（block/shared）、表 2（HBM）、表 3（计算与性能）+ 结论。"""
    _fill_speedup(report, rows)
    out = ["## 资源与性能分析", "",
           "> 口径：bf16=2B、h_post fp32=4B；shared 全 float32；HBM 流量含读放大、未计 L2 命中；"
           "利用率 = 实际带宽 / 1843 GB/s。", ""]

    # 表 1
    out.append("### 表 1：block / shared memory / 线程负载")
    out.append("")
    out.append(render_table(
        ["case (batch,n,c_x)", "kernel", "config (bxb,C,thr)", "grid", "blocks",
         "shared/block", "输出/block", "输出/线程"],
        [[r["case"], r["kernel"], r["config"], "(" + ", ".join(map(str, r["grid"])) + ")",
          str(r["blocks"]), fmt_bytes(r["shared_bytes"]),
          str(r["out_per_block"]), f"{r['out_per_thread']:.1f}"] for r in rows]))
    out.append("")
    out.append("> 注：bxb 实际取 min(config, batch)；shared 远低于 C500 64 KB/block 上限；"
               "输出/线程 <1 表示存在线程空转。")
    out.append("")

    # 表 2
    out.append("### 表 2：HBM 访问（元素粒度次数 + 理论流量）")
    out.append("")
    out.append(render_table(
        ["case", "xlo 读", "h_post 读", "x_res 读", "x_out 写", "总访问次数", "理论 HBM 流量", "32B 事务数"],
        [[r["case"], fmt_int(r["xlo_reads"]), fmt_int(r["h_post_reads"]),
          fmt_int(r["x_res_reads"]), fmt_int(r["x_out_writes"]),
          fmt_int(r["total_hbm_accesses"]), fmt_bytes(r["hbm_bytes"]),
          fmt_int((r["hbm_bytes"] + HBM_MIN_TRANSACTION - 1) // HBM_MIN_TRANSACTION)]
         for r in rows]))
    out.append("")
    out.append("> 注：事务数按 C500 最小访存粒度 32 B 估算；h_post 重复读绝对量小，L2 大概率命中。")
    out.append("")

    # 表 3
    out.append("### 表 3：计算次数与实测性能")
    out.append("")
    out.append(render_table(
        ["case", "真实 FMA", "真实 FLOPs", "latency µs",
         "tflops¹", "speedup", "实际带宽² GB/s", "带宽利用率³"],
        [[r["case"], fmt_int(r["fma"]), fmt_int(r["real_flops"]),
          f"{r['latency_us']:.2f}",
          fmt_small(r["tflops"]), _fmt(r["speedup"], 3) + ("x" if r["speedup"] else ""),
          f"{r['actual_bw_gbs']:.1f}", f"{r['bw_util_pct']:.1f}%"] for r in rows]))
    out.append("")
    out.append("> ¹ tflops 直接取自 log（口径随 bench_mhc.py calculate_flops；MHC post 当前为真实口径 2×batch×n×c_x）。")
    out.append("> ² 实际带宽 = 表 2 理论流量 ÷ latency（未计 L2 命中）。")
    out.append("> ³ 利用率 = 实际带宽 / 1843 GB/s（C500 理论 HBM 峰值）。")
    out.append("")
    return "\n".join(out)


def render_table(headers: list[str], body: list[list[str]]) -> str:
    """渲染 markdown 表格（表头 + 分隔行 + 数据行）。"""
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


# ─────────────────────────── 运行 pytest（可选）───────────────────────────
def _run_pytest(cwd: Path, run_args: str, run_log: Path) -> int:
    cmd = [sys.executable, "-m", "pytest", *shlex.split(run_args)]
    print(f"$ cd {cwd} && {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
    except FileNotFoundError:
        print(f"error: {sys.executable} not found", file=sys.stderr)
        return 127
    output = proc.stdout.decode(errors="replace")
    run_log.parent.mkdir(parents=True, exist_ok=True)
    run_log.write_text(output, encoding="utf-8")
    sys.stdout.write(output)
    return proc.returncode


# ─────────────────────────── 入口 ───────────────────────────
def _find_project_root() -> Path | None:
    """从当前目录或脚本目录向上查找 TileOPs 项目根目录。"""
    for start in (Path.cwd().resolve(), Path(__file__).resolve().parent):
        for candidate in (start, *start.parents):
            if ((candidate / "benchmarks/ops/bench_mhc.py").is_file()
                    and (candidate / "tileops").is_dir()):
                return candidate
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="解析 TileOPs benchmark 报告并生成单个 markdown 报告"
                    "（µs 换算 + 对比 + 资源/性能分析），可选输出分析 CSV。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--input", type=Path, help="输入报告（默认 <项目根>/profile_run.log）")
    ap.add_argument(
        "--output",
        type=Path,
        help=f"输出 markdown 路径（默认 <项目根>/{DEFAULT_OUTPUT_NAME}）",
    )
    ap.add_argument("--csv", type=Path, default=None, help="可选：分析宽表 CSV 输出路径")
    ap.add_argument("--run", metavar="PYTEST_ARGS",
                    help=f"benchmark 的 pytest 参数串（零参数模式默认 {DEFAULT_RUN!r}）")
    ap.add_argument("--cwd", type=Path, help="TileOPs 项目根（默认自动定位）")
    args = ap.parse_args()

    cwd = args.cwd.resolve() if args.cwd is not None else _find_project_root()
    if cwd is None:
        print("error: 无法自动定位 TileOPs 项目根目录，请使用 --cwd 指定", file=sys.stderr)
        return 2
    if not (cwd / "benchmarks").is_dir():
        print(f"error: {cwd} 不是 TileOPs 项目根目录（缺少 benchmarks/）", file=sys.stderr)
        return 2
    out_path = (args.output or (cwd / DEFAULT_OUTPUT_NAME)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 指定 --input 时只处理已有报告；实际跑 benchmark 时先通过 MHC Post 正确性。
    run_args = args.run if args.run is not None else (None if args.input else DEFAULT_RUN)
    if run_args:
        check_log = out_path.parent / (out_path.stem + ".correctness.pytest.log")
        rc = _run_pytest(cwd, DEFAULT_CHECK, check_log)
        if rc != 0:
            print(f"error: MHC Post 正确性测试失败，pytest 退出码 {rc}", file=sys.stderr)
            return rc

        rc = _run_pytest(cwd, run_args, out_path.parent / (out_path.stem + ".pytest.log"))
        if rc != 0:
            print(f"error: MHC Post benchmark 失败，pytest 退出码 {rc}", file=sys.stderr)
            return rc

    in_path = (args.input or (cwd / "profile_run.log")).resolve()
    if not in_path.is_file():
        print(f"error: 报告不存在: {in_path}", file=sys.stderr)
        return 2

    raw = in_path.read_text(encoding="utf-8", errors="replace")
    # 兼容已处理过的报告（旧脚本产物）：剥离生成注释与旧对比章节，避免重复；
    # 原始 profile_run.log（BenchmarkReport.dump 产物）不受影响。
    raw = "\n".join(l for l in raw.splitlines() if not l.startswith("> 本报告由"))
    cutoff = raw.find("\n## 对比信息")
    if cutoff != -1:
        raw = raw[:cutoff]
    raw = re.sub(r"(?m)^# TileOPs Benchmark Report$",
                 "## 原始数据（TileOPs Benchmark Report）", raw, count=1)
    report = parse_report(raw)
    mhc_post_tags = report.get(MHC_POST_OP_NAME)
    if mhc_post_tags is None:
        print(f"error: 输入报告不包含 {MHC_POST_OP_NAME}", file=sys.stderr)
        return 2
    report = {MHC_POST_OP_NAME: mhc_post_tags}

    # 统一 latency 口径（原始 ms 或已换算 µs 输入均可）
    for tags in report.values():
        for rows in tags.values():
            for row in rows:
                _normalize_latency(row)

    # 分析行：仅取 MHCPostOp 的 tileops 行，按实现口径计算指标。
    analysis_rows: list[dict] = []
    for tags in report.values():
        for row in tags.get("tileops", []):
            a = analyze_case(row)
            if a is not None:
                analysis_rows.append(a)

    sections = [
        "# MHC Post Benchmark 报告",
        f"数据来源：`{in_path}`（生成于 {raw.splitlines()[1] if len(raw.splitlines()) > 1 else '?'}）",
        "",
        convert_report(raw),
        "",
        build_comparison(report),
        "",
        build_analysis(report, analysis_rows),
    ]
    out_path.write_text(HEADER_NOTE + "\n".join(sections) + "\n", encoding="utf-8")
    print(f"报告已写入: {out_path}（{len(analysis_rows)} 个 case 分析）")

    if args.csv:
        cols = ["case", "kernel", "config", "grid", "blocks", "shared_bytes",
                "out_per_block", "out_per_thread", "xlo_reads", "h_post_reads",
                "x_res_reads", "x_out_writes", "total_hbm_accesses", "hbm_bytes",
                "fma", "real_flops", "latency_us", "tflops",
                "speedup", "actual_bw_gbs", "bw_util_pct"]
        csv_path = args.csv.resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(analysis_rows)
        print(f"CSV 已写入: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
