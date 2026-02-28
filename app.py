from __future__ import annotations

import email
import email.policy
import plistlib
import re
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse

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
    return ""


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
        q = pair.select_one(".CSkcDe") or pair.select_one(".JCzEY") or pair.select_one("[data-q]")
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


def extract_ai_summary_items(block: BeautifulSoup, block_rank: int) -> list[SerpItem]:
    """
    Heuristic extraction for AI-generated summaries / overviews.
    """
    rows: list[SerpItem] = []
    block_html = str(block)
    block_html_l = block_html.lower()
    block_text = clean_text(block.get_text(" "))
    block_text_l = block_text.lower()

    markers = [
        "ai overview",
        "overview by ai",
        "panoramica dell'ia",
        "panoramica ia",
        "riepilogo ia",
        "sintesi ia",
    ]
    has_marker = any(marker in block_text_l or marker in block_html_l for marker in markers)
    has_wa_description = "wa:/description" in block_html_l

    # Keep this conservative to reduce false positives.
    if not has_marker and not (has_wa_description and (" ai " in f" {block_text_l} " or "overview" in block_text_l)):
        return rows

    summary_nodes = block.select('[data-attrid*="wa:/description"], .hgKElc, .pOOWX, .ILfuVd')
    seen_texts: set[str] = set()
    idx = 1

    for node in summary_nodes:
        desc = clean_text(node.get_text(" "))
        if len(desc) < 80:
            continue
        norm = desc.lower()
        if norm in seen_texts:
            continue
        seen_texts.add(norm)

        link = node.find("a", href=True) or block.find("a", href=True)
        url = clean_text(link.get("href")) if link else ""

        rows.append(
            SerpItem(
                serp_block_type="ai_summary",
                serp_block_rank_tb=block_rank,
                serp_block_rank_lr=1,
                item_type="summary",
                item_rank_tb=idx,
                item_rank_lr=1,
                is_expandable="FALSE",
                title="AI Summary",
                description=desc,
                url=url,
                notes="heuristic_ai_summary; verify manually",
            )
        )
        idx += 1

    if not rows and has_marker and len(block_text) >= 100:
        rows.append(
            SerpItem(
                serp_block_type="ai_summary",
                serp_block_rank_tb=block_rank,
                serp_block_rank_lr=1,
                item_type="summary",
                item_rank_tb=1,
                item_rank_lr=1,
                is_expandable="FALSE",
                title="AI Summary",
                description=block_text[:1000],
                url="",
                notes="heuristic_ai_summary_text; verify manually",
            )
        )

    return rows


def extract_wjd_items(html: str, start_block_rank: int) -> list[SerpItem]:
    rows: list[SerpItem] = []
    rank = start_block_rank

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


def normalize_section_ranks(items: list[SerpItem]) -> list[SerpItem]:
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
        for extractor in (
            extract_ai_summary_items,
            extract_organic_items,
            extract_paa_items,
            extract_video_items,
            extract_image_items,
        ):
            for row in extractor(block, block_rank):
                key = (row.serp_block_type, row.title, row.url)
                if key in seen_rows:
                    continue
                seen_rows.add(key)
                items.append(row)

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
        if row.serp_block_type == "organic":
            row.serp_block_rank_tb = first_organic_block_rank
        elif row.serp_block_type == "video_pack":
            row.serp_block_rank_tb = first_video_block_rank

        rank_key = (row.serp_block_type, row.serp_block_rank_tb)
        row.item_rank_tb = next_item_rank.get(rank_key, 0) + 1
        row.item_rank_lr = 1 if row.serp_block_type != "image_pack" else row.item_rank_tb
        next_item_rank[rank_key] = row.item_rank_tb

        key = (row.serp_block_type, row.title, row.url)
        if key in seen_rows:
            continue
        seen_rows.add(key)
        items.append(row)

    return query, normalize_section_ranks(items)


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


def main() -> None:
    st.set_page_config(page_title="SERP Parser", layout="wide")
    st.title("SERP Parser (HTML/source only)")
    st.caption("Extracts a CSV table from saved Google SERP source files.")

    st.warning(
        "Hierarchy fields are heuristic and MUST be manually checked/edited before final use: "
        "`serp_block_rank_tb`, `serp_block_rank_lr`, `item_rank_tb`, `item_rank_lr` (and optionally `serp_block_type`)."
    )
    st.info(
        "The table below is editable. Review all ranking fields manually, then download the edited CSV. "
        "`ai_summary` extraction is heuristic and may require manual correction."
    )

    uploads = st.file_uploader(
        "Upload one or more source files",
        type=["html", "htm", "mht", "mhtml", "webarchive"],
        accept_multiple_files=True,
    )

    if not uploads:
        st.info("Upload at least one source file to start.")
        return

    all_frames: list[pd.DataFrame] = []
    problems: list[str] = []

    for upload in uploads:
        try:
            html = load_html_from_upload(upload.name, upload.getvalue())
            query, items = parse_serp(html)
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

    st.subheader("Editable Table")
    edited = st.data_editor(
        result,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "serp_block_rank_tb": st.column_config.NumberColumn("serp_block_rank_tb", min_value=1, step=1),
            "serp_block_rank_lr": st.column_config.NumberColumn("serp_block_rank_lr", min_value=1, step=1),
            "item_rank_tb": st.column_config.NumberColumn("item_rank_tb", min_value=1, step=1),
            "item_rank_lr": st.column_config.NumberColumn("item_rank_lr", min_value=1, step=1),
            "serp_block_type": st.column_config.SelectboxColumn(
                "serp_block_type",
                options=["organic", "ai_summary", "people_also_ask", "video_pack", "image_pack", "other"],
            ),
            "is_expandable": st.column_config.SelectboxColumn(
                "is_expandable",
                options=["TRUE", "FALSE"],
            ),
        },
    )

    csv_data = edited.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Edited CSV",
        data=csv_data,
        file_name="serp_output.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
