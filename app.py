from __future__ import annotations

import email
import email.policy
import io
import plistlib
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlparse

import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from PIL import Image
from streamlit_drawable_canvas import st_canvas


OUTPUT_COLUMNS = [
    "query",
    "browser",
    "source_file",
    "serp_block_type",
    "serp_block_rank_tb",
    "serp_block_rank_lr",
    "item_type",
    "item_rank_tb",
    "item_rank_lr",
    "is_expandable",
    "title",
    "description",
    "url",
    "domain",
    "notes",
]


@dataclass
class SerpItem:
    serp_block_type: str
    serp_block_rank_tb: int
    serp_block_rank_lr: int
    item_type: str
    item_rank_tb: int
    item_rank_lr: int
    is_expandable: str
    title: str
    description: str
    url: str
    notes: str


def guess_browser(filename: str) -> str:
    lowered = filename.lower()
    if "firefox" in lowered:
        return "firefox"
    if "chrome" in lowered:
        return "chrome"
    if "safari" in lowered:
        return "safari"
    if "opera" in lowered:
        return "opera"
    return "unknown"


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def extract_domain(url: str) -> str:
    if not url:
        return ""
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def decode_mht(data: bytes) -> str:
    message = email.message_from_bytes(data, policy=email.policy.default)

    if message.is_multipart():
        for part in message.walk():
            ctype = part.get_content_type()
            if ctype == "text/html":
                payload = part.get_payload(decode=True) or b""
                charset = part.get_content_charset() or "utf-8"
                return payload.decode(charset, errors="replace")
    else:
        payload = message.get_payload(decode=True) or b""
        charset = message.get_content_charset() or "utf-8"
        if payload:
            return payload.decode(charset, errors="replace")

    # Fallback when parsing fails: try plain decode.
    return data.decode("utf-8", errors="replace")


def _walk_webarchive(obj: object) -> bytes | None:
    if isinstance(obj, dict):
        if "WebMainResource" in obj:
            main = obj["WebMainResource"]
            if isinstance(main, dict):
                data = main.get("WebResourceData")
                if isinstance(data, (bytes, bytearray)):
                    return bytes(data)
        data = obj.get("WebResourceData")
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        for value in obj.values():
            found = _walk_webarchive(value)
            if found:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _walk_webarchive(value)
            if found:
                return found
    return None


def decode_webarchive(data: bytes) -> str:
    try:
        parsed = plistlib.loads(data)
        html_bytes = _walk_webarchive(parsed)
        if html_bytes:
            return html_bytes.decode("utf-8", errors="replace")
    except Exception:
        pass

    # Robust fallback for partially parseable binaries.
    text = data.decode("utf-8", errors="replace")
    match = re.search(r"<!DOCTYPE html>.*</html>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0)
    return text


def load_html_from_upload(filename: str, data: bytes) -> str:
    lower = filename.lower()
    if lower.endswith((".mht", ".mhtml")):
        return decode_mht(data)
    if lower.endswith(".webarchive"):
        return decode_webarchive(data)
    return data.decode("utf-8", errors="replace")


def guess_query(soup: BeautifulSoup) -> str:
    title = clean_text(soup.title.get_text(" ")) if soup.title else ""
    if " - " in title:
        return clean_text(title.split(" - ", 1)[0])

    comment_text = str(soup)[:4000]
    m = re.search(r"[?&]q=([^&\"]+)", comment_text)
    if m:
        try:
            return clean_text(urlparse(f"https://x.test/?q={m.group(1)}").query)
        except Exception:
            pass
    return ""


def detect_block_type(block: BeautifulSoup) -> str:
    text = clean_text(block.get_text(" ")).lower()
    html = str(block)

    if "related-question-pair" in html or "people also ask" in text or "le persone hanno chiesto anche" in text:
        return "people_also_ask"
    if "youtube.com/shorts" in html or "youtube.com/watch" in html or "VIDEO_RESULT" in html:
        return "video_pack"
    if "encrypted-tbn0.gstatic.com/images?q=tbn" in html:
        return "image_pack"
    if block.select("a.zReHs h3"):
        return "organic"
    return "other"


def extract_organic_items(block: BeautifulSoup, block_rank: int) -> list[SerpItem]:
    items: list[SerpItem] = []
    seen: set[tuple[str, str]] = set()
    for idx, anchor in enumerate(block.select("a.zReHs"), start=1):
        h3 = anchor.select_one("h3")
        title = clean_text(h3.get_text(" ")) if h3 else clean_text(anchor.get_text(" "))
        url = clean_text(anchor.get("href"))
        key = (title, url)
        if not title and not url:
            continue
        if key in seen:
            continue
        seen.add(key)

        card = anchor.find_parent("div", class_=re.compile(r"(tF2Cxc|wHYlTd|N54PNb)"))
        description = ""
        if card:
            desc_node = card.select_one("div.VwiC3b") or card.select_one("div.kb0PBd span")
            description = clean_text(desc_node.get_text(" ")) if desc_node else ""

        items.append(
            SerpItem(
                serp_block_type="organic",
                serp_block_rank_tb=block_rank,
                serp_block_rank_lr=1,
                item_type="result",
                item_rank_tb=idx,
                item_rank_lr=1,
                is_expandable="FALSE",
                title=title,
                description=description,
                url=url,
                notes="",
            )
        )
    return items


def extract_paa_items(block: BeautifulSoup, block_rank: int) -> list[SerpItem]:
    rows: list[SerpItem] = []
    pairs = block.select(".related-question-pair")
    for idx, pair in enumerate(pairs, start=1):
        q = (
            pair.select_one(".CSkcDe")
            or pair.select_one(".JCzEY")
            or pair.select_one("[data-q]")
        )
        title = ""
        if q and q.has_attr("data-q"):
            title = clean_text(q.get("data-q"))
        if not title and q:
            title = clean_text(q.get_text(" "))

        if not title:
            title = clean_text(pair.get("data-q"))

        rows.append(
            SerpItem(
                serp_block_type="people_also_ask",
                serp_block_rank_tb=block_rank,
                serp_block_rank_lr=1,
                item_type="question",
                item_rank_tb=idx,
                item_rank_lr=1,
                is_expandable="TRUE",
                title=title,
                description="",
                url="",
                notes="PAA item",
            )
        )
    return rows


def extract_video_items(block: BeautifulSoup, block_rank: int) -> list[SerpItem]:
    rows: list[SerpItem] = []
    seen: set[str] = set()
    idx = 1
    for anchor in block.select("a[href]"):
        url = clean_text(anchor.get("href"))
        if "youtube.com/watch" not in url and "youtube.com/shorts" not in url:
            continue
        if url in seen:
            continue
        seen.add(url)

        title_node = anchor.select_one("h3, h1, .JGD2rd")
        title = clean_text(title_node.get_text(" ")) if title_node else clean_text(anchor.get_text(" "))
        rows.append(
            SerpItem(
                serp_block_type="video_pack",
                serp_block_rank_tb=block_rank,
                serp_block_rank_lr=1,
                item_type="video",
                item_rank_tb=idx,
                item_rank_lr=1,
                is_expandable="FALSE",
                title=title,
                description="",
                url=url,
                notes="",
            )
        )
        idx += 1
    return rows


def extract_image_items(block: BeautifulSoup, block_rank: int) -> list[SerpItem]:
    rows: list[SerpItem] = []
    seen: set[str] = set()
    idx = 1
    for image in block.select("img[src]"):
        src = clean_text(image.get("src"))
        if "encrypted-tbn0.gstatic.com/images?q=tbn" not in src:
            continue
        if src in seen:
            continue
        seen.add(src)
        rows.append(
            SerpItem(
                serp_block_type="image_pack",
                serp_block_rank_tb=block_rank,
                serp_block_rank_lr=1,
                item_type="image",
                item_rank_tb=idx,
                item_rank_lr=idx,
                is_expandable="FALSE",
                title=clean_text(image.get("alt")),
                description="",
                url=src,
                notes="",
            )
        )
        idx += 1
    return rows


def extract_wjd_items(html: str, start_block_rank: int) -> list[SerpItem]:
    """
    Fallback extraction from serialized Google payloads (W_jd/js snippets).
    Some saved SERP files include many results only inside script data.
    """
    rows: list[SerpItem] = []
    rank = start_block_rank

    # Rich pattern: [title, description, source, data:image...], [null,1,[..., title, ..., url], ...]
    rich_pattern = re.compile(
        r'\["(?P<t1>[^"]+)","(?P<desc>[^"]*)","(?P<src>[^"]*)","data:image[^"]*"\],\s*'
        r'\[null,1,\[null,null,5,null,"(?P<t2>[^"]+)",null,"(?P<url>https://[^"]+)"\]',
        flags=re.DOTALL,
    )
    simple_pattern = re.compile(
        r'\[null,1,\[null,null,5,null,"(?P<title>[^"]+)",null,"(?P<url>https://[^"]+)"\]',
        flags=re.DOTALL,
    )

    seen_urls: set[str] = set()

    for match in rich_pattern.finditer(html):
        url = clean_text(match.group("url"))
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        title = clean_text(match.group("t2") or match.group("t1"))
        desc = clean_text(match.group("desc"))
        if "youtube.com/shorts" in url or "youtube.com/watch" in url:
            block_type = "video_pack"
            item_type = "video"
        else:
            block_type = "organic"
            item_type = "result"
        rows.append(
            SerpItem(
                serp_block_type=block_type,
                serp_block_rank_tb=rank,
                serp_block_rank_lr=1,
                item_type=item_type,
                item_rank_tb=0,
                item_rank_lr=1,
                is_expandable="FALSE",
                title=title,
                description=desc,
                url=url,
                notes="parsed from serialized payload",
            )
        )

    for match in simple_pattern.finditer(html):
        url = clean_text(match.group("url"))
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        title = clean_text(match.group("title"))
        if "youtube.com/shorts" in url or "youtube.com/watch" in url:
            block_type = "video_pack"
            item_type = "video"
        else:
            block_type = "organic"
            item_type = "result"
        rows.append(
            SerpItem(
                serp_block_type=block_type,
                serp_block_rank_tb=rank,
                serp_block_rank_lr=1,
                item_type=item_type,
                item_rank_tb=0,
                item_rank_lr=1,
                is_expandable="FALSE",
                title=title,
                description="",
                url=url,
                notes="parsed from serialized payload",
            )
        )

    return rows


def parse_serp(html: str) -> tuple[str, list[SerpItem]]:
    soup = BeautifulSoup(html, "html.parser")
    query = guess_query(soup)

    rso = soup.select_one("#rso")
    blocks = rso.select(":scope > div") if rso else []
    if not blocks:
        blocks = soup.select("div.MjjYud")

    items: list[SerpItem] = []
    seen_rows: set[tuple[str, str, str]] = set()
    for block_rank, block in enumerate(blocks, start=1):
        # Do not classify blocks exclusively: Google often mixes multiple
        # result families inside the same top-level DOM container.
        for row in extract_organic_items(block, block_rank):
            key = (row.serp_block_type, row.title, row.url)
            if key not in seen_rows:
                seen_rows.add(key)
                items.append(row)
        for row in extract_paa_items(block, block_rank):
            key = (row.serp_block_type, row.title, row.url)
            if key not in seen_rows:
                seen_rows.add(key)
                items.append(row)
        for row in extract_video_items(block, block_rank):
            key = (row.serp_block_type, row.title, row.url)
            if key not in seen_rows:
                seen_rows.add(key)
                items.append(row)
        for row in extract_image_items(block, block_rank):
            key = (row.serp_block_type, row.title, row.url)
            if key not in seen_rows:
                seen_rows.add(key)
                items.append(row)

    # Fallback for saved SERPs where results are present only in serialized JS payloads.
    max_rank = max((x.serp_block_rank_tb for x in items), default=0)
    first_organic_block_rank = min(
        (x.serp_block_rank_tb for x in items if x.serp_block_type == "organic"),
        default=max_rank + 1,
    )
    first_video_block_rank = min(
        (x.serp_block_rank_tb for x in items if x.serp_block_type == "video_pack"),
        default=max_rank + 1,
    )
    next_item_rank: dict[tuple[str, int], int] = {}
    for x in items:
        key = (x.serp_block_type, x.serp_block_rank_tb)
        next_item_rank[key] = max(next_item_rank.get(key, 0), x.item_rank_tb)

    for row in extract_wjd_items(html, start_block_rank=max_rank + 1):
        # Keep fallback organic/video rows in their existing first blocks, then append.
        if row.serp_block_type == "organic":
            row.serp_block_rank_tb = first_organic_block_rank
        elif row.serp_block_type == "video_pack":
            row.serp_block_rank_tb = first_video_block_rank

        rank_key = (row.serp_block_type, row.serp_block_rank_tb)
        row.item_rank_tb = next_item_rank.get(rank_key, 0) + 1
        row.item_rank_lr = 1 if row.serp_block_type != "image_pack" else row.item_rank_tb
        next_item_rank[rank_key] = row.item_rank_tb

        key = (row.serp_block_type, row.title, row.url)
        if key not in seen_rows:
            seen_rows.add(key)
            items.append(row)

    return query, normalize_section_ranks(items)


def normalize_section_ranks(items: list[SerpItem]) -> list[SerpItem]:
    """
    Recompute section ranks from the actual extraction order:
    contiguous items of the same type belong to the same block.
    """
    if not items:
        return items

    current_block_rank = 0
    previous_type = ""
    item_counter: dict[tuple[str, int], int] = {}

    for item in items:
        if item.serp_block_type != previous_type:
            current_block_rank += 1
            previous_type = item.serp_block_type

        item.serp_block_rank_tb = current_block_rank
        item.serp_block_rank_lr = 1

        key = (item.serp_block_type, current_block_rank)
        item_counter[key] = item_counter.get(key, 0) + 1
        item.item_rank_tb = item_counter[key]
        item.item_rank_lr = item.item_rank_tb if item.serp_block_type == "image_pack" else 1

    return items


def rows_to_dataframe(query: str, browser: str, source_file: str, items: Iterable[SerpItem]) -> pd.DataFrame:
    data = []
    for item in items:
        data.append(
            {
                "query": query,
                "browser": browser,
                "source_file": source_file,
                "serp_block_type": item.serp_block_type,
                "serp_block_rank_tb": item.serp_block_rank_tb,
                "serp_block_rank_lr": item.serp_block_rank_lr,
                "item_type": item.item_type,
                "item_rank_tb": item.item_rank_tb,
                "item_rank_lr": item.item_rank_lr,
                "is_expandable": item.is_expandable,
                "title": item.title,
                "description": item.description,
                "url": item.url,
                "domain": extract_domain(item.url),
                "notes": item.notes,
            }
        )

    if not data:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.DataFrame(data)
    return df[OUTPUT_COLUMNS]


def normalize_match_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def extract_pdf_lines(pdf_data: bytes) -> list[tuple[str, float, float]]:
    """
    Extract rough visual lines as (text, y, x) from first PDF page.
    """
    try:
        import pdfplumber
    except Exception as exc:
        raise RuntimeError("Missing dependency: pdfplumber") from exc

    lines: list[tuple[str, float, float]] = []
    with pdfplumber.open(io.BytesIO(pdf_data)) as pdf:
        if not pdf.pages:
            return lines
        page = pdf.pages[0]
        words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []
        if not words:
            return lines

        buckets: dict[int, list[dict]] = {}
        for word in words:
            top = float(word.get("top", 0.0))
            key = int(top // 6)
            buckets.setdefault(key, []).append(word)

        for key in sorted(buckets):
            row_words = sorted(buckets[key], key=lambda w: float(w.get("x0", 0.0)))
            text = clean_text(" ".join(str(w.get("text", "")) for w in row_words))
            if not text:
                continue
            y = float(sum(float(w.get("top", 0.0)) for w in row_words)) / max(len(row_words), 1)
            x = float(min(float(w.get("x0", 0.0)) for w in row_words))
            lines.append((text, y, x))
    return lines


def find_visual_match(html_name: str, visual_uploads: list[object]) -> object | None:
    if not visual_uploads:
        return None
    if len(visual_uploads) == 1:
        return visual_uploads[0]

    html_lower = html_name.lower()
    html_browser = guess_browser(html_name)

    for visual in visual_uploads:
        visual_lower = visual.name.lower()
        if html_browser != "unknown" and html_browser in visual_lower:
            return visual
        if visual_lower.split(".")[0] in html_lower:
            return visual
    return None


def apply_visual_ranking_from_pdf(
    items: list[SerpItem], pdf_data: bytes, visible_only: bool = True
) -> tuple[list[SerpItem], int, int]:
    """
    Reorder extracted items using approximate positions recovered from a PDF snapshot.
    """
    if not items:
        return items, 0, 0

    pdf_lines = extract_pdf_lines(pdf_data)
    norm_lines = [(normalize_match_text(text), y, x) for text, y, x in pdf_lines]
    positions: dict[int, tuple[float, float]] = {}
    matched_indexes: set[int] = set()

    for idx, item in enumerate(items):
        ntitle = normalize_match_text(item.title)
        if len(ntitle) < 6:
            continue

        best: tuple[int, float, float] | None = None
        for nline, y, x in norm_lines:
            if not nline:
                continue
            if ntitle in nline:
                score = len(ntitle)
            elif len(nline) >= 8 and nline in ntitle:
                score = len(nline)
            else:
                continue
            if best is None or score > best[0]:
                best = (score, y, x)
        if best:
            positions[idx] = (best[1], best[2])
            matched_indexes.add(idx)

    working_indices = list(range(len(items)))
    if visible_only:
        filtered_indices: list[int] = []
        for idx, _item in enumerate(items):
            # Keep rows that were matched on visual text.
            if idx in matched_indexes:
                filtered_indices.append(idx)
        # Safety fallback: if matching failed, do not drop everything.
        if filtered_indices:
            working_indices = filtered_indices

    decorated = []
    for idx in working_indices:
        item = items[idx]
        if idx in positions:
            y, x = positions[idx]
            decorated.append((0, y, x, idx, item))
        else:
            decorated.append((1, float(idx), 0.0, idx, item))
    decorated.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    ranked_items = [row[4] for row in decorated]
    ranked_items = normalize_section_ranks(ranked_items)
    return ranked_items, len(matched_indexes), len(working_indices)


def contiguous_blocks_from_items(items: list[SerpItem]) -> list[dict[str, object]]:
    if not items:
        return []
    blocks: list[dict[str, object]] = []
    prev = ""
    count = 0
    for item in items:
        if item.serp_block_type != prev:
            if prev:
                blocks.append({"serp_block_type": prev, "item_count": count})
            prev = item.serp_block_type
            count = 1
        else:
            count += 1
    if prev:
        blocks.append({"serp_block_type": prev, "item_count": count})
    return blocks


def initial_canvas_objects(blocks: list[dict[str, object]], width: int, height: int) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    if not blocks:
        return objects

    margin_x = 24
    top = 24
    block_gap = 14
    usable_w = max(120, width - margin_x * 2)
    avg_h = max(80, int((height - 48 - block_gap * max(len(blocks) - 1, 0)) / max(len(blocks), 1)))

    for block in blocks:
        item_count = int(block.get("item_count", 1))
        h = max(70, min(240, avg_h + (item_count - 1) * 10))
        if top + h > height - 12:
            h = max(50, height - top - 12)
        objects.append(
            {
                "type": "rect",
                "left": float(margin_x),
                "top": float(top),
                "width": float(usable_w),
                "height": float(h),
                "fill": "rgba(30, 136, 229, 0.15)",
                "stroke": "#1e88e5",
                "strokeWidth": 2,
            }
        )
        top += h + block_gap
    return objects


def load_visual_image(upload: object) -> tuple[Image.Image, str]:
    lower = upload.name.lower()
    data = upload.getvalue()
    if lower.endswith(".pdf"):
        try:
            import pypdfium2 as pdfium
        except Exception as exc:
            raise RuntimeError("Missing dependency: pypdfium2 (needed to render PDF for annotation)") from exc

        pdf = pdfium.PdfDocument(io.BytesIO(data))
        if len(pdf) == 0:
            raise RuntimeError("PDF has no pages")
        page = pdf[0]
        pil_image = page.render(scale=2).to_pil()
        return pil_image.convert("RGB"), "pdf"

    pil_image = Image.open(io.BytesIO(data)).convert("RGB")
    return pil_image, "image"


def assign_block_ranks_from_geometry(blocks: list[dict[str, object]], image_height: int) -> list[dict[str, object]]:
    if not blocks:
        return blocks

    blocks_sorted = sorted(blocks, key=lambda b: (float(b["top"]), float(b["left"])))
    row_threshold = max(24.0, image_height * 0.03)
    rows: list[dict[str, object]] = []

    for block in blocks_sorted:
        top = float(block["top"])
        assigned_row = None
        for row in rows:
            if abs(top - float(row["top_ref"])) <= row_threshold:
                assigned_row = row
                break
        if assigned_row is None:
            assigned_row = {"top_ref": top, "blocks": []}
            rows.append(assigned_row)
        assigned_row["blocks"].append(block)

    rows.sort(key=lambda r: float(r["top_ref"]))
    for tb_rank, row in enumerate(rows, start=1):
        row_blocks = sorted(row["blocks"], key=lambda b: float(b["left"]))
        for lr_rank, block in enumerate(row_blocks, start=1):
            block["serp_block_rank_tb"] = tb_rank
            block["serp_block_rank_lr"] = lr_rank

    return blocks_sorted


def default_item_type(block_type: str) -> str:
    if block_type == "people_also_ask":
        return "question"
    if block_type == "video_pack":
        return "video"
    if block_type == "image_pack":
        return "image"
    return "result"


def build_items_from_annotated_blocks(
    parsed_items: list[SerpItem], annotated_blocks: list[dict[str, object]], image_height: int
) -> list[SerpItem]:
    ranked_blocks = assign_block_ranks_from_geometry(annotated_blocks, image_height=image_height)
    pools: dict[str, deque[SerpItem]] = defaultdict(deque)
    for item in parsed_items:
        pools[item.serp_block_type].append(item)

    output: list[SerpItem] = []
    for block in ranked_blocks:
        block_type = str(block["serp_block_type"])
        item_count = max(1, int(block["item_count"]))
        tb_rank = int(block["serp_block_rank_tb"])
        lr_rank = int(block["serp_block_rank_lr"])

        for idx in range(1, item_count + 1):
            if pools[block_type]:
                base = pools[block_type].popleft()
                row = SerpItem(
                    serp_block_type=block_type,
                    serp_block_rank_tb=tb_rank,
                    serp_block_rank_lr=lr_rank,
                    item_type=base.item_type,
                    item_rank_tb=idx,
                    item_rank_lr=idx if block_type == "image_pack" else 1,
                    is_expandable=base.is_expandable,
                    title=base.title,
                    description=base.description,
                    url=base.url,
                    notes=clean_text(f"{base.notes}; visual_assisted"),
                )
            else:
                row = SerpItem(
                    serp_block_type=block_type,
                    serp_block_rank_tb=tb_rank,
                    serp_block_rank_lr=lr_rank,
                    item_type=default_item_type(block_type),
                    item_rank_tb=idx,
                    item_rank_lr=idx if block_type == "image_pack" else 1,
                    is_expandable="TRUE" if block_type == "people_also_ask" else "FALSE",
                    title="",
                    description="",
                    url="",
                    notes="visual_assisted; no_matching_html_item",
                )
            output.append(row)
    return output


def render_auto_mode() -> None:
    st.subheader("Auto Mode")
    st.caption("Automatic extraction from source files. Optional PDF-based re-ranking can be enabled.")
    html_uploads = st.file_uploader(
        "Upload SERP source files (required)",
        type=["html", "htm", "mht", "mhtml", "webarchive"],
        accept_multiple_files=True,
        key="auto_html_uploads",
    )
    visual_uploads = st.file_uploader(
        "Upload visual files for better ranking (optional)",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="auto_visual_uploads",
    )
    use_visual_ranking = st.checkbox(
        "Use visual ranking when a matching PDF is available",
        value=True,
        help="Uses PDF text coordinates to improve top-bottom / left-right ordering.",
        key="auto_use_visual_ranking",
    )
    visible_only_mode = st.checkbox(
        "Keep only items visible in PDF text layer",
        value=True,
        help="Removes rows found only in serialized HTML payload but not visible in the provided PDF snapshot.",
        key="auto_visible_only_mode",
    )

    if not html_uploads:
        st.info("Upload at least one SERP source file to start.")
        return

    all_frames: list[pd.DataFrame] = []
    problems: list[str] = []

    if use_visual_ranking and visual_uploads:
        has_pdf = any(v.name.lower().endswith(".pdf") for v in visual_uploads)
        if not has_pdf:
            st.warning("Visual ranking currently supports PDF only. Images are uploaded but not parsed yet.")

    for upload in html_uploads:
        try:
            file_bytes = upload.getvalue()
            html = load_html_from_upload(upload.name, file_bytes)
            query, items = parse_serp(html)
            if use_visual_ranking:
                matched_visual = find_visual_match(upload.name, visual_uploads or [])
                if matched_visual and matched_visual.name.lower().endswith(".pdf"):
                    items, matched_count, kept_count = apply_visual_ranking_from_pdf(
                        items,
                        matched_visual.getvalue(),
                        visible_only=visible_only_mode,
                    )
                    visual_used = matched_visual.name
                    if matched_count == 0:
                        for item in items:
                            item.notes = clean_text(f"{item.notes}; visual pdf matched but no title positions found")
                    else:
                        for item in items:
                            item.notes = clean_text(
                                f"{item.notes}; visual ranking from {visual_used}; matched_titles={matched_count}; kept_items={kept_count}"
                            )

            browser = guess_browser(upload.name)
            frame = rows_to_dataframe(query=query, browser=browser, source_file=upload.name, items=items)
            all_frames.append(frame)
        except Exception as exc:
            problems.append(f"{upload.name}: {exc}")

    if problems:
        st.warning("Some files were not processed correctly:")
        for problem in problems:
            st.write(f"- {problem}")

    if not all_frames:
        st.error("No data extracted from uploaded files.")
        return

    result = pd.concat(all_frames, ignore_index=True)
    st.subheader("Preview")
    st.dataframe(result, use_container_width=True, hide_index=True)

    csv_data = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="serp_output.csv",
        mime="text/csv",
        key="auto_download_csv",
    )


def render_visual_assisted_mode() -> None:
    st.subheader("Visual Assisted Mode")
    st.caption("Upload HTML + screenshot/PDF. The app proposes blocks and you can manually adjust them.")

    html_upload = st.file_uploader(
        "Upload one SERP source file (required)",
        type=["html", "htm", "mht", "mhtml", "webarchive"],
        accept_multiple_files=False,
        key="va_html_upload",
    )
    visual_upload = st.file_uploader(
        "Upload one screenshot/PDF (required)",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=False,
        key="va_visual_upload",
    )

    if not html_upload or not visual_upload:
        st.info("Upload both one source file and one visual file to annotate blocks.")
        return

    try:
        html = load_html_from_upload(html_upload.name, html_upload.getvalue())
        query, parsed_items = parse_serp(html)
        image, _kind = load_visual_image(visual_upload)
    except Exception as exc:
        st.error(f"Failed to prepare visual mode: {exc}")
        return

    image_w, image_h = image.size
    base_blocks = contiguous_blocks_from_items(parsed_items)
    if not base_blocks:
        st.warning("No extracted items from HTML. You can still draw blocks manually and fill counts.")
        base_blocks = [{"serp_block_type": "organic", "item_count": 1}]

    default_objects = initial_canvas_objects(base_blocks, image_w, image_h)
    initial_drawing = {"version": "4.4.0", "objects": default_objects}

    st.write("1) Draw/edit rectangles on the screenshot. 2) Edit labels/counts in the table. 3) Generate CSV.")
    canvas = st_canvas(
        fill_color="rgba(30, 136, 229, 0.15)",
        stroke_width=2,
        stroke_color="#1e88e5",
        background_image=image,
        update_streamlit=True,
        height=image_h,
        width=image_w,
        drawing_mode="rect",
        initial_drawing=initial_drawing,
        key="va_canvas",
    )

    objects = []
    if canvas.json_data and canvas.json_data.get("objects"):
        for obj in canvas.json_data["objects"]:
            if obj.get("type") != "rect":
                continue
            objects.append(obj)

    if not objects:
        st.warning("Draw at least one rectangle block on the visual.")
        return

    block_rows = []
    for idx, obj in enumerate(objects, start=1):
        fallback_type = base_blocks[idx - 1]["serp_block_type"] if idx - 1 < len(base_blocks) else "organic"
        fallback_count = int(base_blocks[idx - 1]["item_count"]) if idx - 1 < len(base_blocks) else 1
        block_rows.append(
            {
                "block_id": idx,
                "serp_block_type": fallback_type,
                "item_count": fallback_count,
                "left": round(float(obj.get("left", 0.0)), 1),
                "top": round(float(obj.get("top", 0.0)), 1),
                "width": round(float(obj.get("width", 0.0) * float(obj.get("scaleX", 1.0))), 1),
                "height": round(float(obj.get("height", 0.0) * float(obj.get("scaleY", 1.0))), 1),
            }
        )

    block_df = pd.DataFrame(block_rows)
    edited_df = st.data_editor(
        block_df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "serp_block_type": st.column_config.SelectboxColumn(
                "serp_block_type",
                options=["organic", "people_also_ask", "video_pack", "image_pack", "other"],
            ),
            "item_count": st.column_config.NumberColumn("item_count", min_value=1, step=1),
        },
        key="va_blocks_editor",
    )

    if st.button("Generate CSV from annotations", key="va_generate_csv"):
        try:
            annotated_blocks = edited_df.to_dict(orient="records")
            items = build_items_from_annotated_blocks(parsed_items, annotated_blocks, image_height=image_h)
            browser = guess_browser(html_upload.name)
            result = rows_to_dataframe(query=query, browser=browser, source_file=html_upload.name, items=items)
            st.subheader("Annotated Output Preview")
            st.dataframe(result, use_container_width=True, hide_index=True)
            csv_data = result.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Annotated CSV",
                data=csv_data,
                file_name="serp_output_annotated.csv",
                mime="text/csv",
                key="va_download_csv",
            )
        except Exception as exc:
            st.error(f"Failed to generate annotated output: {exc}")


def main() -> None:
    st.set_page_config(page_title="SERP Parser", layout="wide")
    st.title("SERP Parser (Streamlit)")
    tab_auto, tab_visual = st.tabs(["Auto", "Visual Assisted"])
    with tab_auto:
        render_auto_mode()
    with tab_visual:
        render_visual_assisted_mode()


if __name__ == "__main__":
    main()
