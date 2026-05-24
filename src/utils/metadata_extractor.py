import json
import re
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urldefrag, urljoin, urlparse

from lxml import html

from src.models.metadata_extraction_config import (
    JsonSourceRule,
    MetadataExtractionConfig,
    ProductCardRule,
)

_SRCSET_URL_PATTERN = re.compile(r"(https?://\S+?)(?:\s+\d+[wx])?(?:,|$)")
_RELATIVE_SRCSET_ENTRY = re.compile(
    r"(/?[\w./%-]+\.(?:jpg|jpeg|png|webp|gif))\s+\d+[wx]",
    re.IGNORECASE,
)
_IMAGE_EXTENSION_PATTERN = re.compile(r"\.(?:jpg|jpeg|png|webp|gif)(?:\?|$)", re.IGNORECASE)
_BROKEN_TRANSFORM_PATH_PATTERN = re.compile(
    r"/w/(?:f_auto|fl_|c_scale|t_product|h_1\.0|w_1\.0|fl_layer_apply)",
    re.IGNORECASE,
)
_EMBEDDED_HTTPS_IMAGE_PATTERN = re.compile(
    r'https://[^\s"\'<>\\]+?\.(?:jpg|jpeg|png|webp|gif)(?:\?[^\s"\'<>\\]*)?',
    re.IGNORECASE,
)
_NEXT_DATA_PATTERN = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
    re.DOTALL,
)
_IMAGE_ID_PATTERN = re.compile(r"^[a-f0-9-]{8,}$", re.IGNORECASE)
_INLINE_PRODUCT_BLOCK = re.compile(
    r'"copy"\s*:\s*\{\s*"title"\s*:\s*"((?:\\.|[^"\\])*)".*?'
    r'"portraitURL"\s*:\s*"(?P<portrait>https://[^"]+)".*?'
    r'"squarishURL"\s*:\s*"(?P<squarish>https://[^"]+)"',
    re.DOTALL,
)


@dataclass(frozen=True)
class ImageCandidate:
    url: str
    source_page: str
    product_name: Optional[str] = None
    sku_id: Optional[str] = None


@dataclass(frozen=True)
class ProductMetadata:
    name: Optional[str] = None
    sku_id: Optional[str] = None
    image_urls: Tuple[str, ...] = ()


def normalize_url(url: str) -> str:
    normalized, _ = urldefrag(url)
    return normalized


def is_plausible_image_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False

    path = parsed.path.lower()
    if not _IMAGE_EXTENSION_PATTERN.search(path):
        return False
    if _BROKEN_TRANSFORM_PATH_PATTERN.search(path):
        return False
    return True


def _build_json_image_field_pattern(field_names: list[str]) -> re.Pattern[str]:
    fields = "|".join(re.escape(name) for name in field_names)
    return re.compile(rf'"(?:{fields})"\s*:\s*"(https://[^"]+)"', re.IGNORECASE)


def extract_embedded_image_urls(
    page_html: str,
    field_names: Optional[list[str]] = None,
) -> Set[str]:
    urls: Set[str] = set()
    names = field_names or [
        "portraitURL",
        "squarishURL",
        "portraitImg",
        "squarishImg",
        "imageUrl",
        "image_url",
        "thumbnailUrl",
        "src",
    ]

    for match in _build_json_image_field_pattern(names).finditer(page_html):
        raw_url = match.group(1).strip()
        if is_plausible_image_url(raw_url):
            urls.add(normalize_url(raw_url))

    for match in _EMBEDDED_HTTPS_IMAGE_PATTERN.finditer(page_html):
        raw_url = match.group(0).strip()
        if is_plausible_image_url(raw_url):
            urls.add(normalize_url(raw_url))

    return urls


def merge_candidate_metadata(
    existing: ImageCandidate,
    incoming: ImageCandidate,
) -> ImageCandidate:
    return replace(
        existing,
        product_name=existing.product_name or incoming.product_name,
        sku_id=existing.sku_id or incoming.sku_id,
    )


def _parse_srcset(value: str) -> Iterable[str]:
    seen: Set[str] = set()
    for match in _SRCSET_URL_PATTERN.finditer(value):
        url = match.group(1)
        if url not in seen:
            seen.add(url)
            yield url
    for match in _RELATIVE_SRCSET_ENTRY.finditer(value):
        url = match.group(1)
        if url not in seen:
            seen.add(url)
            yield url


def _absolute_image_url(raw_url: Optional[str], base_url: str) -> Optional[str]:
    if not raw_url:
        return None
    raw_url = raw_url.strip()
    if not raw_url or raw_url.startswith(("data:", "blob:", "javascript:")):
        return None
    if raw_url.startswith(("http://", "https://")):
        absolute = normalize_url(raw_url)
    elif raw_url.startswith("//"):
        absolute = normalize_url(f"https:{raw_url}")
    elif raw_url.startswith("/"):
        absolute = normalize_url(urljoin(base_url, raw_url))
    else:
        return None

    if is_plausible_image_url(absolute):
        return absolute
    return None


def _coerce_str(value: Any) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (int, float)):
        return str(value)
    return None


def _unescape_json_string(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        return value.replace('\\"', '"').replace("\\/", "/")


def _get_nested_value(node: Dict[str, Any], path: str) -> Any:
    current: Any = node
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


class ConfigDrivenMetadataExtractor:
    def __init__(self, config: MetadataExtractionConfig):
        self.config = config
        pattern = config.sku_style_pattern or r"^[A-Z0-9]*\d[A-Z0-9]*-\d+$"
        self._style_code_pattern = re.compile(pattern, re.IGNORECASE)

    def is_style_code_sku(self, value: Optional[str]) -> bool:
        return bool(value and self._style_code_pattern.match(value.strip()))

    def sku_from_url(self, raw_url: Optional[str]) -> Optional[str]:
        if not raw_url:
            return None
        for part in reversed(urlparse(raw_url).path.strip("/").split("/")):
            if self.is_style_code_sku(part):
                return part.upper()
        return None

    def parse_page_products(self, page_html: str, base_url: str) -> List[ProductMetadata]:
        products: List[ProductMetadata] = []
        for rule in self.config.json_sources:
            if rule.source == "json_ld":
                products.extend(self._parse_json_ld_products(page_html, base_url, rule))
            elif rule.source == "next_data":
                products.extend(self._parse_next_data_products(page_html, base_url, rule))
            elif rule.source == "inline_json":
                products.extend(self._parse_inline_json_products(page_html, base_url, rule))
            elif rule.source == "algolia_state":
                products.extend(self._parse_algolia_state_products(page_html, base_url, rule))

        if self.config.product_card:
            products.extend(
                self._parse_product_card_products(page_html, base_url, self.config.product_card)
            )

        return self._dedupe_products(products)

    def extract_image_candidates(
        self,
        page_html: str,
        base_url: str,
        source_page: str,
    ) -> List[ImageCandidate]:
        tree = html.fromstring(page_html)
        products = self.parse_page_products(page_html, base_url)

        raw_candidates: List[ImageCandidate] = []
        seen_urls: Set[str] = set()

        def add_candidate(
            raw_url: Optional[str],
            dom_element: Optional[html.HtmlElement] = None,
        ) -> None:
            absolute = _absolute_image_url(raw_url, base_url)
            if not absolute or absolute in seen_urls:
                return
            seen_urls.add(absolute)

            dom_name: Optional[str] = None
            dom_sku: Optional[str] = None
            if dom_element is not None:
                dom_name, dom_sku = self._dom_metadata_for_element(dom_element)

            raw_candidates.append(
                self._candidate_from_url(absolute, source_page, dom_name, dom_sku, products)
            )

        for element in tree.xpath("//img"):
            for attr in ("src", "data-src", "data-lazy-src"):
                add_candidate(element.get(attr), element)
            srcset = element.get("srcset")
            if srcset:
                for candidate_url in _parse_srcset(srcset):
                    add_candidate(candidate_url, element)

        for element in tree.xpath("//picture//source"):
            srcset = element.get("srcset")
            if srcset:
                for candidate_url in _parse_srcset(srcset):
                    add_candidate(candidate_url, element)
            add_candidate(element.get("src"), element)

        for element in tree.xpath("//meta[@property='og:image']"):
            add_candidate(element.get("content"))

        embedded_urls = extract_embedded_image_urls(
            page_html,
            self.config.embedded_image_json_fields,
        )
        for embedded_url in embedded_urls:
            if embedded_url in seen_urls:
                continue
            seen_urls.add(embedded_url)
            raw_candidates.append(
                self._candidate_from_url(embedded_url, source_page, None, None, products)
            )

        return self._apply_single_product_fallback(raw_candidates, products, source_page)

    def _extract_name(self, node: Dict[str, Any], rule: JsonSourceRule) -> Optional[str]:
        for key in rule.name_keys:
            value = _get_nested_value(node, key) if "." in key else node.get(key)
            coerced = _coerce_str(value)
            if coerced:
                return coerced
        return None

    def _extract_sku(self, node: Dict[str, Any], rule: JsonSourceRule) -> Optional[str]:
        if rule.sku_prefer_pdp_url:
            for key in rule.pdp_url_keys:
                raw_pdp = _get_nested_value(node, key) if "." in key else node.get(key)
                sku = self.sku_from_url(_coerce_str(raw_pdp))
                if sku:
                    return sku

        for key in rule.sku_keys:
            value = _get_nested_value(node, key) if "." in key else node.get(key)
            coerced = _coerce_str(value)
            if coerced and self.is_style_code_sku(coerced):
                return coerced.upper()

        if rule.source in ("json_ld", "algolia_state"):
            for key in rule.sku_keys:
                value = _get_nested_value(node, key) if "." in key else node.get(key)
                coerced = _coerce_str(value)
                if coerced:
                    return coerced.upper() if rule.source == "algolia_state" else coerced
        return None

    def _collect_image_urls(
        self,
        node: Dict[str, Any],
        base_url: str,
        rule: JsonSourceRule,
    ) -> Set[str]:
        urls: Set[str] = set()
        for key in rule.image_keys:
            if key.startswith("colorwayImages."):
                sub_key = key.split(".", 1)[1]
                colorway = node.get("colorwayImages")
                if isinstance(colorway, dict):
                    absolute = _absolute_image_url(_coerce_str(colorway.get(sub_key)), base_url)
                    if absolute:
                        urls.add(absolute)
                continue

            value = _get_nested_value(node, key) if "." in key else node.get(key)
            if isinstance(value, dict) and "url" in value:
                absolute = _absolute_image_url(_coerce_str(value.get("url")), base_url)
                if absolute:
                    urls.add(absolute)
            elif isinstance(value, str):
                absolute = _absolute_image_url(value, base_url)
                if absolute:
                    urls.add(absolute)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        absolute = _absolute_image_url(item, base_url)
                        if absolute:
                            urls.add(absolute)
                    elif isinstance(item, dict):
                        for nested_key in ("url", "contentUrl"):
                            absolute = _absolute_image_url(_coerce_str(item.get(nested_key)), base_url)
                            if absolute:
                                urls.add(absolute)

        if isinstance(node.get("colorwayImages"), list):
            for item in node["colorwayImages"]:
                if isinstance(item, dict):
                    urls.update(self._collect_image_urls(item, base_url, rule))

        return urls

    def _product_from_node(
        self,
        node: Dict[str, Any],
        base_url: str,
        rule: JsonSourceRule,
    ) -> Optional[ProductMetadata]:
        image_urls = self._collect_image_urls(node, base_url, rule)
        name = self._extract_name(node, rule)
        sku_id = self._extract_sku(node, rule)

        if not image_urls:
            return None
        if not sku_id and not name:
            return None

        return ProductMetadata(
            name=name,
            sku_id=sku_id,
            image_urls=tuple(sorted(image_urls)),
        )

    def _walk_product_nodes(
        self,
        obj: Any,
        base_url: str,
        rule: JsonSourceRule,
    ) -> List[ProductMetadata]:
        products: List[ProductMetadata] = []

        if isinstance(obj, dict):
            product = self._product_from_node(obj, base_url, rule)
            if product:
                products.append(product)

            if isinstance(obj.get("products"), dict):
                for item in obj["products"].values():
                    if isinstance(item, dict):
                        nested = self._product_from_node(item, base_url, rule)
                        if nested:
                            products.append(nested)

            for value in obj.values():
                products.extend(self._walk_product_nodes(value, base_url, rule))
        elif isinstance(obj, list):
            for item in obj:
                products.extend(self._walk_product_nodes(item, base_url, rule))

        return products

    def _parse_next_data_products(
        self,
        page_html: str,
        base_url: str,
        rule: JsonSourceRule,
    ) -> List[ProductMetadata]:
        match = _NEXT_DATA_PATTERN.search(page_html)
        if not match:
            return []

        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            return []

        return self._dedupe_products(self._walk_product_nodes(payload, base_url, rule))

    @staticmethod
    def _extract_script_object(page_html: str, marker: str) -> Optional[Dict[str, Any]]:
        idx = page_html.find(marker)
        if idx < 0:
            return None

        start = page_html.find("{", page_html.find("=", idx))
        if start < 0:
            return None

        depth = 0
        for pos in range(start, len(page_html)):
            char = page_html[pos]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        payload = json.loads(page_html[start : pos + 1])
                    except json.JSONDecodeError:
                        return None
                    return payload if isinstance(payload, dict) else None
        return None

    def _parse_algolia_state_products(
        self,
        page_html: str,
        base_url: str,
        rule: JsonSourceRule,
    ) -> List[ProductMetadata]:
        payload = self._extract_script_object(page_html, "window.__ALGOLIA_STATE__")
        if not payload:
            return []

        products: List[ProductMetadata] = []
        for value in payload.values():
            if not isinstance(value, dict):
                continue
            results = value.get("results")
            if not isinstance(results, list):
                continue
            for result in results:
                if not isinstance(result, dict):
                    continue
                hits = result.get("hits")
                if not isinstance(hits, list):
                    continue
                for hit in hits:
                    if not isinstance(hit, dict):
                        continue
                    product = self._product_from_node(hit, base_url, rule)
                    if product:
                        products.append(product)

        return self._dedupe_products(products)

    def _parse_inline_json_products(
        self,
        page_html: str,
        base_url: str,
        rule: JsonSourceRule,
    ) -> List[ProductMetadata]:
        products: List[ProductMetadata] = []

        for match in _INLINE_PRODUCT_BLOCK.finditer(page_html):
            product_name = _unescape_json_string(match.group(1))
            block = page_html[match.start(): match.end() + 500]
            sku_id = self._extract_sku_from_block(block, rule)
            image_urls: Set[str] = set()

            for raw_url in (match.group("portrait"), match.group("squarish")):
                absolute = _absolute_image_url(raw_url, base_url)
                if absolute:
                    image_urls.add(absolute)

            if not image_urls:
                continue

            products.append(
                ProductMetadata(
                    name=product_name,
                    sku_id=sku_id,
                    image_urls=tuple(sorted(image_urls)),
                )
            )

        return self._dedupe_products(products)

    def _extract_sku_from_block(self, block: str, rule: JsonSourceRule) -> Optional[str]:
        if rule.sku_prefer_pdp_url:
            pdp_match = re.search(
                r'"pdpUrl"\s*:\s*\{\s*"url"\s*:\s*"(https://[^"]+)"',
                block,
            )
            if pdp_match:
                sku = self.sku_from_url(pdp_match.group(1))
                if sku:
                    return sku

        for key in rule.sku_keys:
            field = key.split(".")[-1]
            sku_match = re.search(rf'"{re.escape(field)}"\s*:\s*"([^"]+)"', block)
            if sku_match and self.is_style_code_sku(sku_match.group(1)):
                return sku_match.group(1).upper()
        return None

    def _parse_product_card_products(
        self,
        page_html: str,
        base_url: str,
        card_rule: ProductCardRule,
    ) -> List[ProductMetadata]:
        tree = html.fromstring(page_html)
        products: List[ProductMetadata] = []

        for card in tree.xpath(card_rule.card_selector):
            links = card.xpath(f"{card_rule.link_selector}/@href")
            if not links:
                continue

            sku_id = self.sku_from_url(links[0]) if card_rule.sku_from_link_url else None
            if card_rule.sku_from_link_url and not sku_id:
                continue

            product_name = None
            if card_rule.name_from_link_text:
                name_values = card.xpath(f"{card_rule.link_selector}/text()")
                product_name = name_values[0].strip() if name_values else None

            image_urls: Set[str] = set()
            if card_rule.image_within_card:
                for element in card.xpath(".//img"):
                    for attr in ("src", "data-src", "data-lazy-src"):
                        absolute = _absolute_image_url(element.get(attr), base_url)
                        if absolute:
                            image_urls.add(absolute)
                    srcset = element.get("srcset")
                    if srcset:
                        for candidate_url in _parse_srcset(srcset):
                            absolute = _absolute_image_url(candidate_url, base_url)
                            if absolute:
                                image_urls.add(absolute)

            if not image_urls:
                continue

            products.append(
                ProductMetadata(
                    name=product_name,
                    sku_id=sku_id,
                    image_urls=tuple(sorted(image_urls)),
                )
            )

        return self._dedupe_products(products)

    def _is_product_type(self, value: Any) -> bool:
        product_types = {"product", "productgroup"}
        if isinstance(value, str):
            return value.casefold() in product_types
        if isinstance(value, list):
            return any(
                isinstance(item, str) and item.casefold() in product_types
                for item in value
            )
        return False

    def _walk_json_ld_nodes(self, payload: Any) -> Iterable[Dict[str, Any]]:
        if isinstance(payload, dict):
            yield payload
            graph = payload.get("@graph")
            if isinstance(graph, list):
                for item in graph:
                    yield from self._walk_json_ld_nodes(item)
        elif isinstance(payload, list):
            for item in payload:
                yield from self._walk_json_ld_nodes(item)

    def _parse_json_ld_products(
        self,
        page_html: str,
        base_url: str,
        rule: JsonSourceRule,
    ) -> List[ProductMetadata]:
        tree = html.fromstring(page_html)
        products: List[ProductMetadata] = []

        for script in tree.xpath("//script[@type='application/ld+json']"):
            raw = script.text
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue

            for node in self._walk_json_ld_nodes(payload):
                if not isinstance(node, dict) or not self._is_product_type(node.get("@type")):
                    continue
                product = self._product_from_node(node, base_url, rule)
                if product:
                    products.append(product)

        return products

    def _merge_product_metadata(
        self,
        existing: ProductMetadata,
        incoming: ProductMetadata,
    ) -> ProductMetadata:
        return ProductMetadata(
            name=existing.name or incoming.name,
            sku_id=existing.sku_id or incoming.sku_id,
            image_urls=tuple(sorted(set(existing.image_urls) | set(incoming.image_urls))),
        )

    def _dedupe_products(self, products: List[ProductMetadata]) -> List[ProductMetadata]:
        deduped: List[ProductMetadata] = []
        for product in products:
            merged = False
            for index, existing in enumerate(deduped):
                if existing.sku_id and product.sku_id and existing.sku_id == product.sku_id:
                    deduped[index] = self._merge_product_metadata(existing, product)
                    merged = True
                    break
            if not merged:
                deduped.append(product)
        return deduped

    def _image_match_keys(self, url: str) -> Set[str]:
        keys: Set[str] = set()
        path = urlparse(url).path
        filename = path.rsplit("/", 1)[-1].lower()
        if filename:
            keys.add(filename)
        for part in path.split("/"):
            if _IMAGE_ID_PATTERN.match(part):
                keys.add(part.lower())
        return keys

    def _lookup_product_metadata(
        self,
        url: str,
        products: List[ProductMetadata],
    ) -> Optional[ProductMetadata]:
        image_map: Dict[str, ProductMetadata] = {}
        for product in products:
            for image_url in product.image_urls:
                image_map[image_url] = product
        if url in image_map:
            return image_map[url]

        url_keys = self._image_match_keys(url)
        if not url_keys:
            return None

        uuid_keys = {key for key in url_keys if _IMAGE_ID_PATTERN.match(key)}
        matches: List[ProductMetadata] = []

        for product in products:
            for image_url in product.image_urls:
                image_keys = self._image_match_keys(image_url)
                if uuid_keys:
                    if uuid_keys & image_keys:
                        matches.append(product)
                        break
                elif url_keys & image_keys:
                    matches.append(product)
                    break

        if not matches:
            return None

        style_code_matches = [product for product in matches if self.is_style_code_sku(product.sku_id)]
        if style_code_matches:
            return style_code_matches[0]
        return matches[0]

    def _text_content(self, element: html.HtmlElement) -> Optional[str]:
        text = element.text_content().strip()
        return text or None

    def _card_ancestor_selector(self) -> Optional[str]:
        selector = self.config.dom_context.card_selector
        if not selector and self.config.product_card:
            selector = self.config.product_card.card_selector
        if not selector:
            return None
        if selector.startswith("//"):
            return f"ancestor-or-self::{selector[2:]}"
        return selector

    def _dom_metadata_for_element(
        self,
        element: html.HtmlElement,
    ) -> Tuple[Optional[str], Optional[str]]:
        dom = self.config.dom_context
        product_name: Optional[str] = None
        sku_id: Optional[str] = None
        current = element
        card_selector = self._card_ancestor_selector()

        for _ in range(dom.ancestor_limit):
            is_card = bool(card_selector and current.xpath(card_selector))
            if is_card:
                link_selector = dom.card_link_selector
                if not link_selector and self.config.product_card:
                    link_selector = self.config.product_card.link_selector
                if link_selector:
                    links = current.xpath(f"{link_selector}/@href")
                    if links and sku_id is None:
                        sku_id = self.sku_from_url(links[0])
                    if product_name is None:
                        name_values = current.xpath(f"{link_selector}/text()")
                        if name_values:
                            product_name = name_values[0].strip()

            if product_name is None:
                for attr in dom.name_attrs:
                    if attr.startswith("itemprop="):
                        prop = attr.split("=", 1)[1]
                        for name_element in current.xpath(f'.//*[@itemprop="{prop}"]'):
                            product_name = self._text_content(name_element)
                            if product_name:
                                break
                    else:
                        value = current.get(attr)
                        if value:
                            product_name = value.strip()
                            break
                    if product_name:
                        break

            if sku_id is None:
                for attr in dom.sku_attrs:
                    if attr.startswith("itemprop="):
                        prop = attr.split("=", 1)[1]
                        for sku_element in current.xpath(f'.//*[@itemprop="{prop}"]'):
                            sku_value = self._text_content(sku_element)
                            if sku_value:
                                sku_id = sku_value
                                break
                    else:
                        value = current.get(attr)
                        if value:
                            sku_id = value.strip()
                            break
                    if sku_id:
                        break

            if product_name and sku_id:
                break

            parent = current.getparent()
            if parent is None:
                break
            current = parent

        return product_name, sku_id

    def _candidate_from_url(
        self,
        url: str,
        source_page: str,
        dom_name: Optional[str],
        dom_sku: Optional[str],
        products: List[ProductMetadata],
    ) -> ImageCandidate:
        product_name = dom_name
        sku_id = dom_sku

        matched_product = self._lookup_product_metadata(url, products)
        if matched_product:
            product_name = product_name or matched_product.name
            if not self.is_style_code_sku(sku_id):
                sku_id = sku_id or matched_product.sku_id

        if self.is_style_code_sku(sku_id):
            sku_id = sku_id.upper()

        return ImageCandidate(
            url=url,
            source_page=source_page,
            product_name=product_name,
            sku_id=sku_id,
        )

    def _apply_single_product_fallback(
        self,
        candidates: List[ImageCandidate],
        products: List[ProductMetadata],
        source_page: str,
    ) -> List[ImageCandidate]:
        path_hint = self.config.single_product_page_path_contains
        if len(products) != 1 or not path_hint or path_hint not in source_page:
            return candidates

        product = products[0]
        enriched: List[ImageCandidate] = []
        for candidate in candidates:
            enriched.append(
                replace(
                    candidate,
                    product_name=candidate.product_name or product.name,
                    sku_id=candidate.sku_id or product.sku_id,
                )
            )
        return enriched


def extract_image_candidates(
    page_html: str,
    base_url: str,
    source_page: str,
    extraction_config: Optional[MetadataExtractionConfig] = None,
) -> List[ImageCandidate]:
    config = extraction_config or MetadataExtractionConfig.generic()
    return ConfigDrivenMetadataExtractor(config).extract_image_candidates(
        page_html,
        base_url,
        source_page,
    )


__all__ = [
    "ConfigDrivenMetadataExtractor",
    "ImageCandidate",
    "ProductMetadata",
    "extract_embedded_image_urls",
    "extract_image_candidates",
    "is_plausible_image_url",
    "merge_candidate_metadata",
    "normalize_url",
]
