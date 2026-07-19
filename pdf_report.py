"""Multi-page PDF QC report: summary page, chart pages, then a detail-table +
full report-text page (see agents_definition.py:QCReportSchema).

Reuses the metric-panel renderer from visualize.py so the chart pages here
look identical to the standalone PNG chart output.
"""
from __future__ import annotations

import textwrap
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from PIL import Image

# Vector PDF text embedding was tried first (Type 42/TrueType CID fonts, via
# matplotlib's pdf.fonttype=42), which renders correctly in matplotlib's own
# renderer and in MuPDF, but Chrome/Edge's built-in PDF viewer (opened via the
# web UI's "PDF 리포트 열기" -> target="_blank") shows Hangul glyphs as "?" or
# tofu boxes for this embedded CID font structure. Rasterizing each page to a
# PNG and assembling those into the PDF (see save_qc_report_pdf) sidesteps
# font-embedding compatibility entirely — every viewer just displays pixels —
# at the cost of non-selectable text, which is acceptable for a QC report.
PAGE_RASTER_DPI = 150

from visualize import (
    STATUS_COLORS,
    DEFAULT_STATUS_COLOR,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
    GRIDLINE,
    BAND_FILL,
    PANEL_WIDTH_IN,
    PANEL_HEIGHT_IN,
    _configure_korean_font,
    _as_dict,
    _draw_metric_panel,
    _draw_panel_captions,
    _required_bottom_margin,
    _required_hspace,
)

PAGE_FACECOLOR = "#fcfcfb"
CHARTS_PER_PAGE = 6
CHART_MAX_COLS = 3
TEXT_LINES_PER_PAGE = 55
# Chars-per-line wrap width for the verbatim report_agent text page. Measured
# against actual rendered glyph width (see visualize.py's renderer-based
# approach) rather than assumed: Korean text at the previous value (95, an
# ASCII-scale guess) ran ~10-15% past the page's right margin, since Hangul
# syllables render wider per character than Latin ones.
TEXT_WRAP_WIDTH = 70

LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"
# assets/logo.png is gitignored (company asset, not for public repo
# distribution — see .gitignore), so a fresh deploy checkout never has it on
# disk. Same fallback source static/index.html's masthead <img> uses.
LOGO_URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTbR_xY50mdZR-8yzlZTNcC62hPqcCdd7fL9kbUBS9NMA&s=10"
LOGO_X = 0.04
LOGO_TOP = 0.965
LOGO_WIDTH = 0.14

# Detail-table layout: each cell's text is capped at TABLE_MAX_LINES (with an
# ellipsis) so every row gets the same fixed height — matplotlib's own Table
# object centers/overflows multi-line cell text instead of growing the row,
# which made long standard_text/recommendation values bleed into the row below.
TABLE_MAX_LINES = 4
TABLE_LINE_HEIGHT = 0.024
TABLE_ROW_PADDING = 0.018
TABLE_HEADER_HEIGHT = 0.05
TABLE_ROW_HEIGHT = TABLE_MAX_LINES * TABLE_LINE_HEIGHT + TABLE_ROW_PADDING
TABLE_ROWS_PER_PAGE = max(1, int((1.0 - TABLE_HEADER_HEIGHT - 0.02) / TABLE_ROW_HEIGHT))

# (header label, metric dict key, column width fraction, wrap width in chars, max lines)
TABLE_COLUMNS = [
    ("지표", "metric", 0.15, 14, 2),
    ("사용자 값", "user_value", 0.13, 12, 2),
    ("기준", "standard_text", 0.24, 22, TABLE_MAX_LINES),
    ("상태", "status", 0.09, 10, 1),
    ("권고사항", "recommendation", 0.39, 36, TABLE_MAX_LINES),
]

VERDICT_COLORS = {
    "분석 진행 가능": STATUS_COLORS["PASS"],
    "조건부 진행": STATUS_COLORS["WARNING"],
    "재처리 권고": STATUS_COLORS["FAIL"],
}


def _wrap_bullets(ax, items: list[str], *, x: float, y_start: float, width: int,
                   fontsize: float, y_min: float = 0.06, line_step: float = 0.035) -> None:
    """Render a bullet list top-down until y_min is hit, then truncate with a counter."""
    if not items:
        ax.text(x, y_start, "- 해당 없음", fontsize=fontsize, color=TEXT_MUTED, transform=ax.transAxes)
        return

    y = y_start
    shown = 0
    for item in items:
        wrapped_lines = textwrap.wrap(f"- {item}", width=width) or [f"- {item}"]
        if y - len(wrapped_lines) * line_step < y_min:
            remaining = len(items) - shown
            if remaining > 0:
                ax.text(x, y, f"… 외 {remaining}건", fontsize=fontsize, color=TEXT_MUTED,
                         fontstyle="italic", transform=ax.transAxes)
            return
        for line in wrapped_lines:
            ax.text(x, y, line, fontsize=fontsize, color=TEXT_PRIMARY, transform=ax.transAxes)
            y -= line_step
        shown += 1


def _ensure_logo_cached() -> bool:
    """Download the logo to LOGO_PATH if it isn't already on disk (a fresh
    deploy checkout never has it — see LOGO_PATH's comment). Returns whether
    a usable logo file exists after this call."""
    if LOGO_PATH.exists():
        return True
    try:
        with urllib.request.urlopen(LOGO_URL, timeout=10) as response:
            data = response.read()
    except (urllib.error.URLError, TimeoutError):
        return False
    LOGO_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOGO_PATH.write_bytes(data)
    return True


def _add_logo(fig, run_date: str) -> None:
    """Draw the BioNexus logo top-left, with the report generation date under
    it — no presenter name/subtitle, per the report's letterhead spec."""
    if not _ensure_logo_cached():
        return
    img = mpimg.imread(LOGO_PATH)
    img_h, img_w = img.shape[0], img.shape[1]
    fig_w_in, fig_h_in = fig.get_size_inches()
    width_in = LOGO_WIDTH * fig_w_in
    height_in = width_in / (img_w / img_h)
    height_frac = height_in / fig_h_in

    logo_ax = fig.add_axes((LOGO_X, LOGO_TOP - height_frac, LOGO_WIDTH, height_frac))
    logo_ax.imshow(img)
    logo_ax.axis("off")

    fig.text(LOGO_X, LOGO_TOP - height_frac - 0.012, f"생성일: {run_date}",
              ha="left", va="top", fontsize=8, color=TEXT_MUTED)


def _draw_status_badges(ax, fig, *, y: float, counts: dict[str, int]) -> None:
    """Render 'PASS N개 · WARNING N개 · FAIL N개', each segment colored by
    status. Segment widths are measured via the renderer (same technique as
    the caption blocks in visualize.py) so the whole group centers exactly
    regardless of digit/font width, rather than a hand-tuned x offset."""
    renderer = fig.canvas.get_renderer()
    order = ["PASS", "WARNING", "FAIL"]
    segments = []
    for i, key in enumerate(order):
        if i > 0:
            segments.append((" · ", TEXT_MUTED, "normal"))
        segments.append((f"{key} {counts.get(key, 0)}개", STATUS_COLORS[key], "bold"))

    widths = []
    for text, color, weight in segments:
        t = ax.text(0, y, text, transform=ax.transAxes, fontsize=11,
                     fontweight=weight, color=color, ha="left", va="center")
        bbox = t.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
        widths.append(bbox.x1 - bbox.x0)
        t.remove()

    x = 0.5 - sum(widths) / 2
    for (text, color, weight), w in zip(segments, widths):
        ax.text(x, y, text, transform=ax.transAxes, fontsize=11,
                 fontweight=weight, color=color, ha="left", va="center")
        x += w


def _add_page_number(fig, page_num: int, total_pages: int) -> None:
    # Bottom-right, not bottom-center: chart pages already put the status
    # legend at bottom-center (loc="lower center"), and a centered page
    # number collided directly with it.
    fig.text(0.96, 0.014, f"{page_num} / {total_pages}", ha="right", va="bottom",
              fontsize=8.5, color=TEXT_MUTED)


def _draw_summary_page(fig, *, category: str, analysis_purpose: str, run_date: str,
                        verdict: str, summary: str, strengths: list[str], cautions: list[str],
                        status_counts: dict[str, int]) -> None:
    fig.patch.set_facecolor(PAGE_FACECOLOR)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")

    _add_logo(fig, run_date)

    ax.text(0.5, 0.94, "QC 분석 리포트", ha="center", va="top", fontsize=24,
             fontweight="bold", color=TEXT_PRIMARY, transform=ax.transAxes)

    meta = f"카테고리: {category}   |   분석 목적: {analysis_purpose or '-'}   |   실행 날짜: {run_date}"
    wrapped_meta = "\n".join(textwrap.wrap(meta, width=80))
    # y is low enough to clear the logo block (LOGO_TOP down to its date
    # label) in the top-left corner regardless of the centered line's width.
    ax.text(0.5, 0.83, wrapped_meta, ha="center", va="top", fontsize=10.5,
             color=TEXT_SECONDARY, transform=ax.transAxes, linespacing=1.5)

    verdict_color = VERDICT_COLORS.get(verdict, DEFAULT_STATUS_COLOR)
    ax.text(0.5, 0.76, "종합 판정", ha="center", va="top", fontsize=12,
             color=TEXT_MUTED, transform=ax.transAxes)
    ax.text(0.5, 0.70, verdict, ha="center", va="top", fontsize=28, fontweight="bold",
             color=verdict_color, transform=ax.transAxes,
             bbox=dict(boxstyle="round,pad=0.7", facecolor=verdict_color, alpha=0.12,
                        edgecolor=verdict_color, linewidth=1.5))

    _draw_status_badges(ax, fig, y=0.585, counts=status_counts)

    wrapped_summary = "\n".join(textwrap.wrap(summary or "-", width=68))
    ax.text(0.5, 0.52, wrapped_summary, ha="center", va="top", fontsize=10.5,
             color=TEXT_PRIMARY, transform=ax.transAxes, linespacing=1.6)

    ax.text(0.08, 0.40, "강점", fontsize=13, fontweight="bold",
             color=STATUS_COLORS["PASS"], transform=ax.transAxes)
    # width=32: each two-column block only has ~0.4 of the page width to work
    # with (column starts at x=0.08/0.56); measured against actual rendered
    # Hangul glyph width (see TEXT_WRAP_WIDTH), the previous 46 ran past the
    # right margin for Korean-heavy recommendation text.
    _wrap_bullets(ax, strengths, x=0.08, y_start=0.355, width=32, fontsize=9.5)

    ax.text(0.56, 0.40, "주의점", fontsize=13, fontweight="bold",
             color=STATUS_COLORS["FAIL"], transform=ax.transAxes)
    _wrap_bullets(ax, cautions, x=0.56, y_start=0.355, width=32, fontsize=9.5)


def _build_chart_page_figure(chunk: list[dict], page_label: str):
    ncols = min(CHART_MAX_COLS, len(chunk))
    nrows = (len(chunk) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(PANEL_WIDTH_IN * ncols, PANEL_HEIGHT_IN * nrows),
        squeeze=False, facecolor=PAGE_FACECOLOR,
    )

    panels = []
    for idx, metric in enumerate(chunk):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        ax.set_facecolor(PAGE_FACECOLOR)
        _draw_metric_panel(ax, metric)
        panels.append((ax, metric))

    for idx in range(len(chunk), nrows * ncols):
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

    fig.suptitle(f"QC 지표 상세 {page_label}".strip(), fontsize=15, fontweight="bold",
                 color=TEXT_PRIMARY, y=0.995)
    # Bottom margin and hspace are both sized to this page's worst-case
    # caption depth (see visualize._required_bottom_margin / _required_hspace)
    # instead of a fixed value, so long standard_text/source captions never
    # run into the row below or into the legend.
    bottom = _required_bottom_margin(chunk, nrows)
    fig.tight_layout(rect=(0, bottom, 1, 0.96))
    fig.subplots_adjust(hspace=_required_hspace(chunk), wspace=0.35)

    # Captions are drawn only now that every axes has its final size/position,
    # so the renderer-measured placements in _draw_caption_block are accurate.
    _draw_panel_captions(fig, panels)
    return fig


def _build_table_page_figure(rows: list[dict], page_label: str):
    """Draw the detail table by hand (fixed-height rows of wrapped, capped
    text) rather than via ax.table, which doesn't grow a row to fit
    multi-line cell text and lets long cells overlap the row below."""
    fig = plt.figure(figsize=(11.69, 8.27), facecolor=PAGE_FACECOLOR)
    ax = fig.add_axes((0.03, 0.04, 0.94, 0.86))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    col_x = []
    x = 0.0
    for _, _, width, _, _ in TABLE_COLUMNS:
        col_x.append(x)
        x += width

    header_top = 1.0
    header_bottom = header_top - TABLE_HEADER_HEIGHT
    ax.add_patch(Rectangle((0, header_bottom), 1, TABLE_HEADER_HEIGHT, facecolor=BAND_FILL, edgecolor="none"))
    for (label, _, _, _, _), cx in zip(TABLE_COLUMNS, col_x):
        ax.text(cx + 0.01, header_bottom + TABLE_HEADER_HEIGHT / 2, label, fontsize=9.5,
                 fontweight="bold", color=TEXT_PRIMARY, va="center", ha="left")

    y = header_bottom
    for row in rows:
        row_bottom = y - TABLE_ROW_HEIGHT
        status = (row.get("status") or "-").upper()
        status_color = STATUS_COLORS.get(status, DEFAULT_STATUS_COLOR)

        for (_, key, _, wrap_width, max_lines), cx in zip(TABLE_COLUMNS, col_x):
            raw = str(row.get(key) or "-")
            wrapped = "\n".join(textwrap.wrap(raw, width=wrap_width, max_lines=max_lines, placeholder="…")) or "-"
            is_status = key == "status"
            ax.text(
                cx + 0.01, y - 0.014, wrapped, fontsize=8.5,
                color=status_color if is_status else TEXT_PRIMARY,
                fontweight="bold" if is_status else "normal",
                va="top", ha="left", linespacing=1.4,
            )

        ax.axhline(row_bottom, xmin=0, xmax=1, color=GRIDLINE, linewidth=0.6)
        y = row_bottom

    for cx in [*col_x, 1.0]:
        ax.axvline(cx, ymin=y, ymax=header_top, color=GRIDLINE, linewidth=0.6)

    fig.suptitle(f"지표별 상세 표 {page_label}".strip(), fontsize=14, fontweight="bold", color=TEXT_PRIMARY)
    return fig


def _build_text_pages(full_text: str, title: str):
    lines: list[str] = []
    for paragraph in (full_text or "").splitlines():
        if not paragraph.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(paragraph, width=TEXT_WRAP_WIDTH) or [""])
    if not lines:
        lines = ["-"]

    page_chunks = [lines[i:i + TEXT_LINES_PER_PAGE] for i in range(0, len(lines), TEXT_LINES_PER_PAGE)]
    total = len(page_chunks)

    figs = []
    for idx, page_lines in enumerate(page_chunks, start=1):
        fig = plt.figure(figsize=(8.27, 11.69), facecolor=PAGE_FACECOLOR)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.axis("off")
        header = title if total == 1 else f"{title} ({idx}/{total})"
        ax.text(0.06, 0.965, header, fontsize=13, fontweight="bold", color=TEXT_PRIMARY,
                 transform=ax.transAxes, va="top")
        ax.text(0.06, 0.925, "\n".join(page_lines), fontsize=8.5, color=TEXT_PRIMARY,
                 transform=ax.transAxes, va="top", linespacing=1.5)
        figs.append(fig)
    return figs


def save_qc_report_pdf(
    *,
    category: str,
    analysis_purpose: str,
    run_date: str,
    report: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Render a multi-page QC PDF: 1) summary page, 2) chart pages (6/page),
    3) detail table page(s), 4) the report_agent verdict text verbatim.

    Args:
        category: omics category string (e.g. "ONT").
        analysis_purpose: user-entered analysis goal, may be empty.
        run_date: pre-formatted date string for display.
        report: dict with verdict, summary, metrics (QCMetricResult-like list),
            recommendations, and text (the full report_agent Markdown output).
        output_path: destination PDF path; parent directories are created.

    Returns:
        The resolved output path.
    """
    _configure_korean_font()

    metrics = [_as_dict(m) for m in (report.get("metrics") or [])]
    verdict = report.get("verdict") or "-"
    summary = report.get("summary") or ""
    recommendations = report.get("recommendations") or []
    full_text = report.get("text") or ""

    strengths = [
        f"{m.get('metric', '-')}: {m.get('user_value', '-')}"
        for m in metrics if (m.get("status") or "").upper() == "PASS"
    ]
    cautions = [
        f"{m.get('metric', '-')}: {m.get('recommendation') or m.get('user_value', '-')}"
        for m in metrics if (m.get("status") or "").upper() in ("WARNING", "FAIL")
    ]
    if not cautions and recommendations:
        cautions = list(recommendations)

    status_counts = {
        key: sum(1 for m in metrics if (m.get("status") or "").upper() == key)
        for key in ("PASS", "WARNING", "FAIL")
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build every page figure up front so the total page count is known
    # before stamping page numbers (each page needs "n / total").
    figs = []

    summary_fig = plt.figure(figsize=(8.27, 11.69))
    _draw_summary_page(
        summary_fig, category=category, analysis_purpose=analysis_purpose, run_date=run_date,
        verdict=verdict, summary=summary, strengths=strengths, cautions=cautions,
        status_counts=status_counts,
    )
    figs.append(summary_fig)

    if metrics:
        chart_chunks = [metrics[i:i + CHARTS_PER_PAGE] for i in range(0, len(metrics), CHARTS_PER_PAGE)]
        for idx, chunk in enumerate(chart_chunks, start=1):
            label = f"({idx}/{len(chart_chunks)})" if len(chart_chunks) > 1 else ""
            figs.append(_build_chart_page_figure(chunk, label))

        table_chunks = [metrics[i:i + TABLE_ROWS_PER_PAGE] for i in range(0, len(metrics), TABLE_ROWS_PER_PAGE)]
        for idx, chunk in enumerate(table_chunks, start=1):
            label = f"({idx}/{len(table_chunks)})" if len(table_chunks) > 1 else ""
            figs.append(_build_table_page_figure(chunk, label))

    figs.extend(_build_text_pages(full_text, "종합 판정 (report_agent 원문)"))

    total_pages = len(figs)
    page_images: list[Image.Image] = []
    for page_num, fig in enumerate(figs, start=1):
        _add_page_number(fig, page_num, total_pages)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=PAGE_RASTER_DPI, facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        page_images.append(Image.open(buf).convert("RGB"))

    first_page, remaining_pages = page_images[0], page_images[1:]
    first_page.save(
        output_path, format="PDF", save_all=True, append_images=remaining_pages,
        resolution=PAGE_RASTER_DPI,
    )

    return output_path
