"""Bar-chart visualization for QCReportSchema.metrics (see agents_definition.py).

Renders one small-multiple panel per metric (units differ per metric — bp, Q
score, %, Gb — so a single shared axis would be misleading) showing the
user's value as a bar, colored by PASS/WARNING/FAIL, with the gold-standard
range drawn as a shaded band / dashed threshold line when available.
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Status palette (fixed, never themed) — see dataviz skill references/palette.md.
# PASS/WARNING/FAIL map onto the good/warning/critical status roles; "serious"
# is unused since this schema only has three statuses.
STATUS_COLORS = {
    "PASS": "#0ca30c",
    "WARNING": "#fab219",
    "FAIL": "#d03b3b",
}
DEFAULT_STATUS_COLOR = "#898781"  # muted ink, for an unrecognized status string

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BAND_FILL = "#c3c2b7"

_KOREAN_FONT_CANDIDATES = [
    "Malgun Gothic",   # Windows default Korean sans
    "AppleGothic",     # macOS fallback
    "NanumGothic",
    "Noto Sans CJK KR",
    "Noto Sans KR",
]


def _configure_korean_font() -> None:
    available = {f.name for f in fm.fontManager.ttflist}
    for name in _KOREAN_FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    else:
        plt.rcParams["font.family"] = plt.rcParams.get("font.family", "sans-serif")
    plt.rcParams["axes.unicode_minus"] = False


def _extract_numeric(value: str | None) -> float | None:
    """Pull the first number out of a free-text metric value, e.g. '12,400 bp' -> 12400.0."""
    if not value:
        return None
    match = re.search(r"-?\d[\d,]*\.?\d*", value)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _as_dict(metric: Any) -> dict:
    if hasattr(metric, "model_dump"):
        return metric.model_dump()
    if isinstance(metric, dict):
        return metric
    raise TypeError(f"Unsupported metric type: {type(metric)!r}")


def _draw_metric_panel(ax, metric: dict) -> None:
    name = metric.get("metric", "")
    user_value_text = metric.get("user_value", "")
    status = (metric.get("status") or "").upper()
    standard_min = metric.get("standard_min")
    standard_max = metric.get("standard_max")
    standard_text = metric.get("standard_text", "") or ""
    standard_source = metric.get("standard_source", "") or ""
    is_verified_source = bool(metric.get("is_verified_source"))

    color = STATUS_COLORS.get(status, DEFAULT_STATUS_COLOR)
    user_value = _extract_numeric(user_value_text)

    ax.set_title(name, fontsize=11, fontweight="bold", color=TEXT_PRIMARY, pad=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)

    if user_value is None:
        ax.text(
            0.5, 0.5, f"{user_value_text or '값 없음'}\n(수치로 변환할 수 없음)",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=9, color=TEXT_SECONDARY,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return

    bar_width = 0.45
    ax.bar([0], [user_value], width=bar_width, color=color, zorder=3)
    ax.set_xlim(-1, 1)
    ax.set_xticks([])  # no default tick label — the caption block below carries it

    scale_values = [user_value, 0.0]
    has_band = standard_min is not None and standard_max is not None
    has_single_threshold = (standard_min is not None) ^ (standard_max is not None)

    if has_band:
        ax.axhspan(standard_min, standard_max, color=BAND_FILL, alpha=0.25, zorder=1)
        ax.axhline(standard_min, color=TEXT_SECONDARY, linestyle="--", linewidth=1, zorder=2)
        ax.axhline(standard_max, color=TEXT_SECONDARY, linestyle="--", linewidth=1, zorder=2)
        scale_values += [standard_min, standard_max]
    elif has_single_threshold:
        threshold = standard_min if standard_min is not None else standard_max
        ax.axhline(threshold, color=TEXT_SECONDARY, linestyle="--", linewidth=1, zorder=2)
        scale_values.append(threshold)

    top = max(scale_values) * 1.25 if max(scale_values) > 0 else 1.0
    ax.set_ylim(0, top)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", colors=TEXT_MUTED, labelsize=8)

    # value-at-the-tip label
    ax.text(
        0, user_value + top * 0.03, f"{user_value:g}",
        ha="center", va="bottom", fontsize=9, color=TEXT_PRIMARY, fontweight="bold",
    )

    # Caption block below the axes, in axes-fraction coordinates so it never
    # collides with the bar/gridlines regardless of each panel's data scale.
    headline = f"{user_value_text or f'{user_value:g}'} · {status or '-'}"
    ax.text(
        0.5, -0.10, headline, transform=ax.transAxes,
        ha="center", va="top", fontsize=9, color=TEXT_PRIMARY,
    )
    # standard_text can wrap up to 3 lines, so the source block below it must shift
    # down by however many lines standard_text actually used — a fixed y position
    # (calibrated for ~2 lines) gets overrun and visually collides whenever a
    # threshold description is long enough to wrap to all 3.
    source_y = -0.20
    if standard_text:
        wrapped_lines = textwrap.wrap(standard_text, width=30, max_lines=3, placeholder="…")
        wrapped = "\n".join(wrapped_lines)
        ax.text(
            0.5, -0.20, wrapped, transform=ax.transAxes,
            ha="center", va="top", fontsize=7.5, color=TEXT_MUTED, linespacing=1.4,
        )
        source_y = -0.20 - len(wrapped_lines) * 0.05 - 0.03
    if standard_source:
        if is_verified_source:
            source_label = f"출처: {standard_source}"
            source_color = TEXT_MUTED
            source_weight = "normal"
        else:
            source_label = f"(추정치) {standard_source}"
            source_color = STATUS_COLORS["WARNING"]
            source_weight = "bold"
        wrapped_source = "\n".join(textwrap.wrap(source_label, width=34, max_lines=2, placeholder="…"))
        ax.text(
            0.5, source_y, wrapped_source, transform=ax.transAxes,
            ha="center", va="top", fontsize=7, color=source_color, fontweight=source_weight, linespacing=1.3,
        )


def save_qc_chart(metrics: list[Any], output_path: str | Path) -> Path:
    """Render one bar-chart panel per QC metric and save it as a PNG.

    Args:
        metrics: list of QCMetricResult (or dict-equivalents) with fields
            metric, user_value, standard_text, standard_min, standard_max,
            standard_source, is_verified_source, status, recommendation.
        output_path: destination PNG path; parent directories are created.

    Returns:
        The resolved output path.
    """
    if not metrics:
        raise ValueError("metrics list is empty")

    _configure_korean_font()

    metric_dicts = [_as_dict(m) for m in metrics]

    ncols = min(3, len(metric_dicts))
    nrows = (len(metric_dicts) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.8 * ncols, 5.4 * nrows), squeeze=False, facecolor="#fcfcfb"
    )

    for idx, metric in enumerate(metric_dicts):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        ax.set_facecolor("#fcfcfb")
        _draw_metric_panel(ax, metric)

    for idx in range(len(metric_dicts), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].axis("off")

    legend_handles = [
        Patch(facecolor=STATUS_COLORS["PASS"], label="PASS"),
        Patch(facecolor=STATUS_COLORS["WARNING"], label="WARNING"),
        Patch(facecolor=STATUS_COLORS["FAIL"], label="FAIL"),
        Patch(facecolor=BAND_FILL, alpha=0.25, edgecolor=TEXT_SECONDARY, linestyle="--", label="Gold Standard 기준"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=4, frameon=False,
        fontsize=9, labelcolor=TEXT_SECONDARY, bbox_to_anchor=(0.5, 0.0),
    )

    fig.suptitle("QC 지표 요약", fontsize=15, fontweight="bold", color=TEXT_PRIMARY, y=0.995)
    # Generous bottom margin per subplot (caption block) and hspace between
    # rows so one panel's caption never runs into the row below it.
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.subplots_adjust(hspace=1.35, wspace=0.35)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    return output_path
