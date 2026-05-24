from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class ColorSwatch(BaseModel):
    """A single dominant color extracted from brand imagery."""

    hex: str = Field(description="Hex code like '#1A1A1A'.")
    r: int
    g: int
    b: int
    frequency_pct: float = Field(
        default=0.0,
        description="Share of analyzed pixels mapped to this swatch (0-100).",
    )


class ClusterTheme(BaseModel):
    """One predefined aesthetic territory from the Phase 1 discovery pass."""

    cluster_label: str = Field(
        description="Short evocative cluster name, e.g. 'Monochrome Architecture'."
    )
    description: str = Field(
        description="2-3 sentences describing what unites this aesthetic territory."
    )
    key_elements: List[str] = Field(
        default_factory=list,
        description="3-6 short tags capturing the territory's defining elements.",
    )
    seed_image_indices: List[int] = Field(
        default_factory=list,
        description=(
            "1-based indices into the discovery sample that exemplify this territory."
        ),
    )


class ClusterThemeDiscovery(BaseModel):
    """Structured output of Phase 1: global theme plan from a representative sample."""

    garment_categories: List[str] = Field(
        default_factory=list,
        description="Garment types observed (outerwear, knitwear, tailoring, ...).",
    )
    silhouettes: List[str] = Field(
        default_factory=list,
        description="Silhouette language (oversized, slim-fit, boxy, draped, ...).",
    )
    styling_notes: str = Field(
        default="",
        description="2-3 sentences on how looks are styled — layering, proportion, accessories.",
    )
    formality_spectrum: str = Field(
        default="",
        description="One of: Streetwear / Smart-casual / Contemporary / Elevated casual / Formal.",
    )
    recurring_motifs: List[str] = Field(
        default_factory=list,
        description="Patterns, textures, hardware, details seen repeatedly.",
    )
    mood_words: List[str] = Field(
        default_factory=list,
        description="5-8 adjectives capturing the visual mood.",
    )
    themes: List[ClusterTheme] = Field(
        default_factory=list,
        description=(
            "Exactly the requested number of mutually distinct aesthetic territories. "
            "Each label must differ on at least one axis (sport, audience, channel, "
            "color story, silhouette)."
        ),
    )


class ClusterObservation(BaseModel):
    """One aesthetic cluster identified by the vision LLM."""

    cluster_label: str = Field(
        description="Short evocative cluster name, e.g. 'Monochrome Architecture'."
    )
    image_indices: List[int] = Field(
        default_factory=list,
        description=(
            "1-based indices into the image batch the LLM was shown. Map back "
            "to actual image paths/URLs after the call returns."
        ),
    )
    description: str = Field(
        description="2-3 sentences describing what unites these looks aesthetically."
    )
    key_elements: List[str] = Field(
        default_factory=list,
        description="3-6 short tags capturing the cluster's defining elements.",
    )


class BrandVisualAnalysis(BaseModel):
    """Structured output of the vision LLM pass over brand imagery."""

    garment_categories: List[str] = Field(
        default_factory=list,
        description="Garment types observed (outerwear, knitwear, tailoring, ...).",
    )
    silhouettes: List[str] = Field(
        default_factory=list,
        description="Silhouette language (oversized, slim-fit, boxy, draped, ...).",
    )
    styling_notes: str = Field(
        default="",
        description="2-3 sentences on how looks are styled — layering, proportion, accessories.",
    )
    formality_spectrum: str = Field(
        default="",
        description="One of: Streetwear / Smart-casual / Contemporary / Elevated casual / Formal.",
    )
    recurring_motifs: List[str] = Field(
        default_factory=list,
        description="Patterns, textures, hardware, details seen repeatedly.",
    )
    mood_words: List[str] = Field(
        default_factory=list,
        description="5-8 adjectives capturing the visual mood.",
    )
    cluster_observations: List[ClusterObservation] = Field(
        default_factory=list,
        description="3-6 distinct aesthetic clusters with representative image indices.",
    )
    unassigned_image_indices: List[int] = Field(
        default_factory=list,
        description=(
            "1-based global indices the LLM did not place into any cluster — kept "
            "separate rather than dumped into the smallest cluster. Available as a "
            "backfill pool for thin clusters during rendering."
        ),
    )


class BatchClusterAssignment(BaseModel):
    """Structured output of Phase 2: assign batch images to fixed themes."""

    assignments: List[ClusterObservation] = Field(
        default_factory=list,
        description=(
            "One entry per fixed theme. cluster_label must match a provided theme "
            "exactly. image_indices are 1-based within the current batch."
        ),
    )


class BrandVoice(BaseModel):
    tone: str = Field(description="e.g. 'Confident and understated'.")
    vocabulary: List[str] = Field(
        default_factory=list,
        description="5-8 words or short phrases characteristic of this brand.",
    )
    what_they_avoid: str = Field(
        default="",
        description="One sentence on what this brand would never say or do.",
    )


class AudienceProfile(BaseModel):
    demographic_profile: str = Field(
        default="",
        description="Age range, lifestyle context, income bracket — 2 sentences.",
    )
    psychographic_profile: str = Field(
        default="",
        description="Values, motivations, cultural reference points — 2 sentences.",
    )
    what_they_seek: str = Field(
        default="",
        description="What this customer wants from a fashion brand — 1-2 sentences.",
    )


class BrandDnaSynthesis(BaseModel):
    """Final strategist-grade synthesis assembled from visual + textual signals."""

    positioning_statement: str = Field(
        description=(
            "One crisp sentence: '[Brand] is the [category] for [audience] who [insight].'"
        )
    )
    brand_essence: str = Field(
        description="2-3 sentences capturing the core of the brand — poetic but grounded."
    )
    brand_voice: BrandVoice
    visual_identity_summary: str = Field(
        description=(
            "3-4 sentences synthesizing garment mix, silhouette language, color "
            "philosophy, and overall aesthetic direction."
        )
    )
    audience: AudienceProfile
    competitive_space: str = Field(
        default="",
        description="1-2 sentences situating the brand among peers.",
    )
    strategic_opportunities: List[str] = Field(
        default_factory=list,
        description="3 bullet-point opportunities the brand could lean into.",
    )
    watchouts: List[str] = Field(
        default_factory=list,
        description="2 risks or tensions observed in the brand presentation.",
    )


class ClusterRender(BaseModel):
    """A cluster paired with the actual image URLs chosen to represent it."""

    label: str
    description: str
    key_elements: List[str] = []
    image_urls: List[str] = []


class BrandDnaReport(BaseModel):
    """Full payload handed to the PDF renderer."""

    brand_name: str
    brand_url: Optional[str] = None
    generated_at: datetime
    palette: List[ColorSwatch]
    visual: BrandVisualAnalysis
    synthesis: BrandDnaSynthesis
    clusters: List[ClusterRender]
    image_count: int
    text_snippet_count: int
