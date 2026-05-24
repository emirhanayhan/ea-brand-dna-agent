"""Brand DNA PDF renderer.

Consumes a `BrandDnaReport` plus a mapping of `image_url -> bytes` (so the
cluster strips can embed the same images the LLM analyzed without re-fetching).
Layout is ported from the reference `brand_dna.py` script at the repo root.
"""

from __future__ import annotations

import io
import logging
from html import escape
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from src.models.brand_dna import (
    BrandDnaReport,
    BrandDnaSynthesis,
    BrandVisualAnalysis,
    ClusterRender,
    ColorSwatch,
)

logger = logging.getLogger("BrandDnaPdfRenderer")

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm

C_INK = colors.HexColor("#1A1A1A")
C_ACCENT = colors.HexColor("#B8966E")
C_LIGHT = colors.HexColor("#F5F3EF")
C_MID = colors.HexColor("#D6D0C8")
C_SUBHEAD = colors.HexColor("#5C5248")
C_WHITE = colors.white


def _styles() -> Dict[str, ParagraphStyle]:
    def s(name, **kwargs):
        return ParagraphStyle(name, **kwargs)

    return {
        "section_title": s(
            "section_title",
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=24,
            textColor=C_INK,
            spaceBefore=10,
            spaceAfter=6,
        ),
        "sub_title": s(
            "sub_title",
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=16,
            textColor=C_SUBHEAD,
            spaceBefore=8,
            spaceAfter=3,
        ),
        "body": s(
            "body",
            fontName="Helvetica",
            fontSize=10,
            leading=16,
            textColor=C_INK,
            spaceAfter=6,
        ),
        "body_sm": s(
            "body_sm",
            fontName="Helvetica",
            fontSize=9,
            leading=14,
            textColor=C_SUBHEAD,
            spaceAfter=4,
        ),
        "label": s(
            "label",
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=12,
            textColor=C_ACCENT,
            spaceAfter=2,
            spaceBefore=4,
        ),
        "cluster_title": s(
            "cluster_title",
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=18,
            textColor=C_INK,
            spaceAfter=3,
        ),
        "cluster_body": s(
            "cluster_body",
            fontName="Helvetica",
            fontSize=9.5,
            leading=15,
            textColor=C_INK,
            spaceAfter=4,
        ),
        "quote": s(
            "quote",
            fontName="Helvetica-Oblique",
            fontSize=13,
            leading=20,
            textColor=C_SUBHEAD,
            alignment=TA_CENTER,
            spaceBefore=12,
            spaceAfter=12,
        ),
        "tag": s(
            "tag",
            fontName="Helvetica",
            fontSize=8,
            leading=12,
            textColor=C_SUBHEAD,
            alignment=TA_LEFT,
        ),
    }


class _BrandDnaDoc(BaseDocTemplate):
    def __init__(self, filename: str, brand_name: str, **kwargs):
        super().__init__(filename, **kwargs)
        self.brand_name = brand_name

        content_frame = Frame(
            MARGIN,
            MARGIN,
            PAGE_W - 2 * MARGIN,
            PAGE_H - 2 * MARGIN,
            id="content",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=8 * mm,
        )
        cover_frame = Frame(
            0,
            0,
            PAGE_W,
            PAGE_H,
            id="cover",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(
            [
                PageTemplate(
                    id="cover", frames=[cover_frame], onPage=self._draw_cover_page
                ),
                PageTemplate(
                    id="content", frames=[content_frame], onPage=self._draw_footer
                ),
            ]
        )

    def _draw_cover_page(self, canvas, doc):
        _draw_cover(canvas, doc, self.brand_name)

    def _draw_footer(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(C_MID)
        canvas.drawString(
            MARGIN,
            10 * mm,
            f"{self.brand_name} — Brand DNA Report   ·   Confidential",
        )
        canvas.drawRightString(
            PAGE_W - MARGIN, 10 * mm, f"Page {doc.page - 1}"
        )
        canvas.setStrokeColor(C_MID)
        canvas.setLineWidth(0.4)
        canvas.line(MARGIN, 13 * mm, PAGE_W - MARGIN, 13 * mm)
        canvas.restoreState()


def _draw_cover(canvas, doc, brand_name: str) -> None:
    w, h = PAGE_W, PAGE_H
    canvas.saveState()
    canvas.setFillColor(C_INK)
    canvas.rect(0, 0, w, h, fill=1, stroke=0)

    canvas.setFillColor(C_ACCENT)
    canvas.rect(0, h * 0.35, w * 0.006, h * 0.65, fill=1, stroke=0)

    canvas.setFont("Helvetica-Bold", 42)
    canvas.setFillColor(C_WHITE)
    canvas.drawString(MARGIN + 6, h * 0.62, brand_name.upper())

    canvas.setFont("Helvetica", 16)
    canvas.setFillColor(C_ACCENT)
    canvas.drawString(MARGIN + 6, h * 0.55, "BRAND DNA REPORT")

    canvas.setStrokeColor(C_ACCENT)
    canvas.setLineWidth(0.8)
    canvas.line(MARGIN + 6, h * 0.525, w - MARGIN, h * 0.525)

    canvas.setFont("Helvetica", 10)
    canvas.setFillColor(colors.HexColor("#A09488"))
    canvas.drawString(
        MARGIN + 6,
        h * 0.495,
        "Visual Identity  ·  Brand Voice  ·  Audience Signals  ·  Aesthetic Clustering",
    )

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(C_SUBHEAD)
    canvas.drawString(MARGIN + 6, MARGIN, "Confidential — For Internal Use Only")
    canvas.drawRightString(
        w - MARGIN, MARGIN, "Synthesized by EA Brand Intelligence Agent"
    )

    canvas.restoreState()


def _section_header(title: str, styles) -> list:
    return [
        HRFlowable(width="100%", thickness=0.5, color=C_ACCENT, spaceAfter=4),
        Paragraph(title.upper(), styles["section_title"]),
        Spacer(1, 3 * mm),
    ]


def _color_swatches(palette: List[ColorSwatch], styles) -> Table:
    if not palette:
        palette = [ColorSwatch(hex="#CCCCCC", r=204, g=204, b=204)]
    swatch_w = (PAGE_W - 2 * MARGIN) / len(palette)
    swatch_h = 14 * mm

    color_cells = [""] * len(palette)
    hex_cells = [Paragraph(p.hex, styles["tag"]) for p in palette]

    tbl = Table(
        [color_cells, hex_cells],
        colWidths=[swatch_w] * len(palette),
        rowHeights=[swatch_h, 8 * mm],
    )
    style_cmds = [
        ("GRID", (0, 0), (-1, 0), 0, colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 1), (-1, 1), 7.5),
        ("TEXTCOLOR", (0, 1), (-1, 1), C_SUBHEAD),
    ]
    for i, swatch in enumerate(palette):
        try:
            c = colors.HexColor(swatch.hex)
        except Exception:  # noqa: BLE001
            c = colors.HexColor("#CCCCCC")
        style_cmds.append(("BACKGROUND", (i, 0), (i, 0), c))
    tbl.setStyle(TableStyle(style_cmds))
    return tbl


def _pill_tags(items: List[str], styles, max_width: Optional[float] = None):
    if not items:
        return Paragraph("—", styles["body_sm"])

    if max_width is None:
        max_width = PAGE_W - 2 * MARGIN

    # "tag" paragraph style is Helvetica @ 8pt; reserve 6pt left/right padding
    # plus a small cushion so glyphs don't wrap inside a single pill.
    h_pad = 12.0
    cushion = 4.0
    min_pill_w = 28.0
    row_gap = 2.0

    pills: List[tuple[str, float]] = []
    for item in items:
        text_w = stringWidth(item, "Helvetica", 8)
        w = max(text_w + h_pad + cushion, min_pill_w)
        w = min(w, max_width)
        pills.append((item, w))

    rows: List[List[tuple[str, float]]] = []
    current: List[tuple[str, float]] = []
    current_w = 0.0
    for text, w in pills:
        if current and current_w + w > max_width:
            rows.append(current)
            current = []
            current_w = 0.0
        current.append((text, w))
        current_w += w
    if current:
        rows.append(current)

    pill_style = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), C_LIGHT),
            ("BOX", (0, 0), (-1, 0), 0.4, C_MID),
            ("INNERGRID", (0, 0), (-1, 0), 0.4, C_MID),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]
    )

    row_tables = []
    for row in rows:
        cells = [Paragraph(escape(t), styles["tag"]) for t, _ in row]
        widths = [w for _, w in row]
        row_tbl = Table([cells], colWidths=widths, hAlign="LEFT")
        row_tbl.setStyle(pill_style)
        row_tables.append([row_tbl])

    if len(row_tables) == 1:
        return row_tables[0][0]

    outer = Table(row_tables, colWidths=[max_width], hAlign="LEFT")
    outer.setStyle(
        TableStyle(
            [
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), row_gap),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ]
        )
    )
    return outer


def _two_col_kv(pairs: List[tuple[str, str]], styles) -> Table:
    col_w = [(PAGE_W - 2 * MARGIN) * 0.3, (PAGE_W - 2 * MARGIN) * 0.7]
    rows = [
        [
            Paragraph(escape(k), styles["label"]),
            Paragraph(escape(v).replace("\n", "<br/>"), styles["body"]),
        ]
        for k, v in pairs
    ]
    tbl = Table(rows, colWidths=col_w)
    tbl.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, 0), (-1, -2), 0.3, C_LIGHT),
            ]
        )
    )
    return tbl


def _cluster_image_strip(
    image_bytes_list: List[bytes],
    max_w: float,
    max_h: float,
) -> Optional[Table]:
    valid: List[tuple[bytes, float]] = []
    for raw in image_bytes_list:
        try:
            pil = PILImage.open(io.BytesIO(raw))
            pil.load()
            aspect = pil.size[1] / pil.size[0] if pil.size[0] else 1.0
        except Exception as exc:  # noqa: BLE001
            logger.debug("skipping cluster image (decode failed): %s", exc)
            continue
        valid.append((raw, aspect))

    if not valid:
        return None

    n = len(valid)
    img_w = (max_w - (n - 1) * 3 * mm) / n
    cells = []
    for raw, aspect in valid:
        # Scale uniformly so the image fits inside both img_w and max_h
        # without distortion. `aspect` is height / width.
        if img_w * aspect <= max_h:
            actual_w = img_w
            actual_h = img_w * aspect
        else:
            actual_h = max_h
            actual_w = max_h / aspect if aspect else img_w
        cells.append(Image(io.BytesIO(raw), width=actual_w, height=actual_h))

    tbl = Table([cells], colWidths=[img_w] * n, rowHeights=max_h)
    tbl.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 1.5 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 1.5 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return tbl


def render_brand_dna_pdf(
    report: BrandDnaReport,
    image_bytes_by_url: Dict[str, bytes],
    output_path: Path | str,
) -> Path:
    """Build the PDF on disk and return the resolved path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = _styles()
    doc = _BrandDnaDoc(
        str(output_path),
        report.brand_name,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN + 8 * mm,
    )

    story: list = []
    _append_cover_placeholder(story)
    _append_executive_summary(story, report.synthesis, styles)
    _append_visual_identity(story, report.palette, report.visual, report.synthesis, styles)
    _append_brand_voice(story, report.synthesis, styles)
    _append_audience(story, report.synthesis, styles)
    _append_clusters(story, report.clusters, image_bytes_by_url, styles)
    _append_strategic_notes(story, report.synthesis, styles)

    doc.build(story)
    logger.info("brand DNA PDF written to %s", output_path)
    return output_path


def _append_cover_placeholder(story: list) -> None:
    story.append(NextPageTemplate("cover"))
    story.append(Spacer(PAGE_W, PAGE_H))
    story.append(NextPageTemplate("content"))
    story.append(PageBreak())


def _append_executive_summary(
    story: list, synthesis: BrandDnaSynthesis, styles
) -> None:
    story.extend(_section_header("Executive Summary", styles))
    if synthesis.positioning_statement:
        story.append(
            Paragraph(
                f'"{escape(synthesis.positioning_statement)}"',
                styles["quote"],
            )
        )
        story.append(Spacer(1, 3 * mm))
    if synthesis.brand_essence:
        story.append(Paragraph("Brand Essence", styles["sub_title"]))
        story.append(Paragraph(escape(synthesis.brand_essence), styles["body"]))
        story.append(Spacer(1, 4 * mm))
    story.append(PageBreak())


def _append_visual_identity(
    story: list,
    palette: List[ColorSwatch],
    visual: BrandVisualAnalysis,
    synthesis: BrandDnaSynthesis,
    styles,
) -> None:
    story.extend(_section_header("Visual Identity", styles))

    story.append(Paragraph("Dominant Color Palette", styles["sub_title"]))
    story.append(_color_swatches(palette, styles))
    story.append(Spacer(1, 5 * mm))

    if synthesis.visual_identity_summary:
        story.append(
            Paragraph(escape(synthesis.visual_identity_summary), styles["body"])
        )
        story.append(Spacer(1, 4 * mm))

    kv = []
    if visual.garment_categories:
        kv.append(("GARMENT MIX", ", ".join(visual.garment_categories)))
    if visual.silhouettes:
        kv.append(("SILHOUETTES", ", ".join(visual.silhouettes)))
    if visual.formality_spectrum:
        kv.append(("FORMALITY", visual.formality_spectrum))
    if visual.styling_notes:
        kv.append(("STYLING LANGUAGE", visual.styling_notes))
    if visual.recurring_motifs:
        kv.append(("RECURRING MOTIFS", ", ".join(visual.recurring_motifs)))

    if kv:
        story.append(_two_col_kv(kv, styles))
        story.append(Spacer(1, 4 * mm))

    if visual.mood_words:
        story.append(
            Paragraph("Mood & Aesthetic Vocabulary", styles["sub_title"])
        )
        story.append(_pill_tags(visual.mood_words, styles))
        story.append(Spacer(1, 3 * mm))

    story.append(PageBreak())


def _append_brand_voice(
    story: list, synthesis: BrandDnaSynthesis, styles
) -> None:
    story.extend(_section_header("Brand Voice & Textual Identity", styles))

    voice = synthesis.brand_voice
    v_pairs = []
    if voice.tone:
        v_pairs.append(("TONE", voice.tone))
    if voice.what_they_avoid:
        v_pairs.append(("WOULD NEVER", voice.what_they_avoid))
    if v_pairs:
        story.append(_two_col_kv(v_pairs, styles))
        story.append(Spacer(1, 4 * mm))

    if voice.vocabulary:
        story.append(Paragraph("Brand Vocabulary", styles["sub_title"]))
        story.append(_pill_tags(voice.vocabulary, styles))
        story.append(Spacer(1, 4 * mm))

    if synthesis.competitive_space:
        story.append(Paragraph("Competitive Positioning", styles["sub_title"]))
        story.append(
            Paragraph(escape(synthesis.competitive_space), styles["body"])
        )
        story.append(Spacer(1, 4 * mm))

    story.append(PageBreak())


def _append_audience(
    story: list, synthesis: BrandDnaSynthesis, styles
) -> None:
    story.extend(_section_header("Audience Signals", styles))
    aud = synthesis.audience
    pairs = []
    if aud.demographic_profile:
        pairs.append(("DEMOGRAPHIC", aud.demographic_profile))
    if aud.psychographic_profile:
        pairs.append(("PSYCHOGRAPHIC", aud.psychographic_profile))
    if aud.what_they_seek:
        pairs.append(("WHAT THEY SEEK", aud.what_they_seek))
    if pairs:
        story.append(_two_col_kv(pairs, styles))
        story.append(Spacer(1, 6 * mm))
    story.append(PageBreak())


def _append_clusters(
    story: list,
    clusters: List[ClusterRender],
    image_bytes_by_url: Dict[str, bytes],
    styles,
) -> None:
    story.extend(_section_header("Aesthetic Clustering", styles))
    if not clusters:
        story.append(
            Paragraph(
                "No aesthetic clusters were identified from the supplied imagery.",
                styles["body"],
            )
        )
        return

    story.append(
        Paragraph(
            f"The following {len(clusters)} clusters were identified through visual "
            "analysis of the brand's imagery. Each represents a distinct aesthetic "
            "territory within the collection.",
            styles["body"],
        )
    )
    story.append(Spacer(1, 5 * mm))

    usable_w = PAGE_W - 2 * MARGIN
    for i, cluster in enumerate(clusters):
        elements: list = [
            Paragraph(
                f"0{i + 1} — {escape(cluster.label)}", styles["cluster_title"]
            ),
            Paragraph(escape(cluster.description), styles["cluster_body"]),
        ]
        if cluster.key_elements:
            elements.append(_pill_tags(cluster.key_elements, styles))
            elements.append(Spacer(1, 3 * mm))

        raw_images = [
            image_bytes_by_url[url]
            for url in cluster.image_urls
            if url in image_bytes_by_url
        ]
        strip = _cluster_image_strip(raw_images, usable_w, 55 * mm)
        if strip is not None:
            elements.append(strip)
            elements.append(Spacer(1, 2 * mm))

        elements.append(
            HRFlowable(width="100%", thickness=0.3, color=C_MID, spaceAfter=6)
        )
        story.append(KeepTogether(elements))
        story.append(Spacer(1, 4 * mm))

    story.append(PageBreak())


def _append_strategic_notes(
    story: list, synthesis: BrandDnaSynthesis, styles
) -> None:
    story.extend(_section_header("Strategic Notes", styles))

    opp_col: list = [Paragraph("Opportunities", styles["sub_title"])]
    for o in synthesis.strategic_opportunities:
        opp_col.append(Paragraph(f"→  {escape(o)}", styles["body"]))

    watch_col: list = [Paragraph("Watch-outs", styles["sub_title"])]
    for w in synthesis.watchouts:
        watch_col.append(Paragraph(f"⚠  {escape(w)}", styles["body"]))

    usable_w = PAGE_W - 2 * MARGIN
    half = (usable_w - 8 * mm) / 2
    tbl = Table([[opp_col, watch_col]], colWidths=[half, half])
    tbl.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8 * mm),
            ]
        )
    )
    story.append(tbl)
