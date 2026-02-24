from __future__ import annotations

import email
import email.policy
import io
import plistlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlparse

import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup


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


def main() -> None:
    st.set_page_config(page_title="SERP Parser", layout="wide")
    st.title("SERP Parser (Streamlit)")
    st.caption("Extract blocks and contents from saved Google SERP snapshots.")

    st.subheader("Input files")
    html_uploads = st.file_uploader(
        "Upload SERP source files (required)",
        type=["html", "htm", "mht", "mhtml", "webarchive"],
        accept_multiple_files=True,
    )
    visual_uploads = st.file_uploader(
        "Upload visual files for better ranking (optional)",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    use_visual_ranking = st.checkbox(
        "Use visual ranking when a matching PDF is available",
        value=True,
        help="Uses PDF text coordinates to improve top-bottom / left-right ordering.",
    )
    visible_only_mode = st.checkbox(
        "Keep only items visible in PDF text layer",
        value=True,
        help="Removes rows found only in serialized HTML payload but not visible in the provided PDF snapshot.",
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
            visual_used = ""
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
    )


if __name__ == "__main__":
    main()
