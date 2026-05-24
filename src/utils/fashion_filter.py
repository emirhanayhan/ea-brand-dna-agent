"""Fashion / clothing image gate.

Addresses the case-study requirement that collected images be "confidently
identified as clothing or fashion-related." Without this, editorial pages
on brand sites (Kaft's conceptual illustrations: lamps, furniture, surreal
art) leak into the cluster strips and bias the LLM's read of the brand.

Three strategies are provided, selected by config:

  - NullFashionFilter: identity. Default behavior, preserves existing flow.
  - HeuristicFashionFilter: cheap, always-available filtering based on
    aspect ratio + product metadata signals already present on
    DiscoveredImage. Drops nothing on its own — it only demotes obvious
    non-product imagery (extreme aspect ratios, social-source images
    without product_name) so the cluster picker prefers proper shots.
    NB: this is a coarse signal; for hard filtering use the CLIP variant.
  - ClipFashionFilter: zero-shot classification via OpenCLIP. Opt-in,
    requires `pip install open_clip_torch torch`. Lazy-imports so the
    base install never pays the dependency cost.

Wire-up: BrandDnaService.__init__ -> make_fashion_filter(settings).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Protocol, Tuple

from src.models.image_outputs import DiscoveredImage

logger = logging.getLogger("FashionFilter")

ImageItem = Tuple[DiscoveredImage, bytes]


class FashionFilter(Protocol):
    """Filter brand images to those confidently identifiable as fashion."""

    def filter(self, items: List[ImageItem]) -> List[ImageItem]: ...

    @property
    def name(self) -> str: ...


class NullFashionFilter:
    name = "null"

    def filter(self, items: List[ImageItem]) -> List[ImageItem]:
        return items


class HeuristicFashionFilter:
    """Drops images whose metadata strongly suggests they are not garments.

    This is a coarse pre-filter — it only removes images that are clearly
    *not* fashion (e.g. extreme aspect ratios that look like banners or
    icons). It deliberately keeps the long tail intact; for stricter
    filtering, enable the CLIP variant.
    """

    name = "heuristic"

    def __init__(
        self,
        min_aspect: float = 0.4,
        max_aspect: float = 2.5,
    ) -> None:
        self.min_aspect = float(min_aspect)
        self.max_aspect = float(max_aspect)

    def filter(self, items: List[ImageItem]) -> List[ImageItem]:
        kept: List[ImageItem] = []
        dropped = 0
        for image, data in items:
            if image.height == 0 or image.width == 0:
                dropped += 1
                continue
            aspect = image.width / image.height
            if aspect < self.min_aspect or aspect > self.max_aspect:
                # Banners, hero strips, navigation icons.
                dropped += 1
                continue
            kept.append((image, data))
        if dropped:
            logger.info(
                "fashion_filter[heuristic]: dropped %s/%s non-product images",
                dropped,
                len(items),
            )
        return kept


class ClipFashionFilter:
    """Zero-shot fashion classification via OpenCLIP.

    Lazy-imports open_clip / torch on first use so the base project never
    requires them. If the import fails we log once and fall back to a no-op
    so the run still completes (graceful failure, per case study).
    """

    name = "clip"

    _POSITIVE_PROMPTS = (
        "a photo of clothing",
        "a photo of a fashion model wearing apparel",
        "a product shot of a garment",
        "a photo of an accessory (bag, shoes, hat)",
    )
    _NEGATIVE_PROMPTS = (
        "a digital illustration or artwork",
        "a photo of an interior or furniture",
        "a photo of a landscape",
        "an abstract pattern or logo",
        "a screenshot of text",
    )

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        threshold: float = 0.55,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.threshold = float(threshold)
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._text_features = None
        self._torch = None
        self._available: bool | None = None

    def _ensure_loaded(self) -> bool:
        if self._available is False:
            return False
        if self._model is not None:
            return True
        try:
            import open_clip  # type: ignore[import-not-found]
            import torch  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "fashion_filter[clip]: open_clip/torch not installed (%s); "
                "skipping fashion gate. Run `pip install open_clip_torch torch` "
                "or set brand_dna_fashion_filter_enabled=false.",
                exc,
            )
            self._available = False
            return False
        self._torch = torch
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer(self.model_name)
        prompts = list(self._POSITIVE_PROMPTS) + list(self._NEGATIVE_PROMPTS)
        with torch.no_grad():
            tokens = tokenizer(prompts)
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            )
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = tokenizer
        self._text_features = text_features
        self._available = True
        return True

    def filter(self, items: List[ImageItem]) -> List[ImageItem]:
        if not items:
            return items
        if not self._ensure_loaded():
            return items
        import io

        from PIL import Image

        torch = self._torch
        n_pos = len(self._POSITIVE_PROMPTS)
        kept: List[ImageItem] = []
        dropped = 0

        for image, data in items:
            try:
                pil = Image.open(io.BytesIO(data)).convert("RGB")
            except Exception as exc:  # noqa: BLE001
                logger.debug("fashion_filter[clip]: decode failed: %s", exc)
                kept.append((image, data))
                continue
            with torch.no_grad():
                tensor = self._preprocess(pil).unsqueeze(0)
                img_features = self._model.encode_image(tensor)
                img_features = img_features / img_features.norm(
                    dim=-1, keepdim=True
                )
                # Cosine sim against each prompt -> softmax -> sum positives
                logits = (img_features @ self._text_features.T) * 100.0
                probs = logits.softmax(dim=-1).squeeze(0)
                fashion_prob = float(probs[:n_pos].sum().item())

            if fashion_prob >= self.threshold:
                kept.append((image, data))
            else:
                dropped += 1
                logger.debug(
                    "fashion_filter[clip]: dropping %s (P(fashion)=%.2f)",
                    image.url,
                    fashion_prob,
                )

        if dropped:
            logger.info(
                "fashion_filter[clip]: dropped %s/%s images below P=%.2f",
                dropped,
                len(items),
                self.threshold,
            )
        return kept


def make_fashion_filter(settings: Dict[str, Any]) -> FashionFilter:
    """Build the fashion filter configured for this run.

    Config keys (all optional):
      brand_dna_fashion_filter_strategy: 'null' | 'heuristic' | 'clip'
        (default: 'heuristic' — coarse, no extra deps)
      brand_dna_fashion_filter_threshold: float, used by 'clip' (default 0.55)
      brand_dna_fashion_filter_model: str, used by 'clip' (default 'ViT-B-32')
      brand_dna_fashion_filter_pretrained: str, used by 'clip' (default 'openai')
    """
    strategy = str(
        settings.get("brand_dna_fashion_filter_strategy", "heuristic")
    ).lower()
    if strategy == "clip":
        return ClipFashionFilter(
            model_name=str(
                settings.get("brand_dna_fashion_filter_model", "ViT-B-32")
            ),
            pretrained=str(
                settings.get("brand_dna_fashion_filter_pretrained", "openai")
            ),
            threshold=float(
                settings.get("brand_dna_fashion_filter_threshold", 0.55)
            ),
        )
    if strategy == "null":
        return NullFashionFilter()
    return HeuristicFashionFilter()
