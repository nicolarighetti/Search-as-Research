from __future__ import annotations

import email
import email.policy
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
        title = clean_text(anchor.get_text(" "))
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
                item_rank_tb=1,
                item_rank_lr=1,
                is_expandable="FALSE",
                title=title,
                description=desc,
                url=url,
                notes="parsed from serialized payload",
            )
        )
        rank += 1

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
                item_rank_tb=1,
                item_rank_lr=1,
                is_expandable="FALSE",
                title=title,
                description="",
                url=url,
                notes="parsed from serialized payload",
            )
        )
        rank += 1

    return rows


def parse_serp(html: str) -> tuple[str, list[SerpItem]]:
    soup = BeautifulSoup(html, "html.parser")
    query = guess_query(soup)

    rso = soup.select_one("#rso")
    blocks = rso.select(":scope > div") if rso else []
    if not blocks:
        blocks = soup.select("div.MjjYud")

    items: list[SerpItem] = []
    seen_rows: set[tuple[str, str, str, int]] = set()
    for block_rank, block in enumerate(blocks, start=1):
        # Do not classify blocks exclusively: Google often mixes multiple
        # result families inside the same top-level DOM container.
        for row in extract_organic_items(block, block_rank):
            key = (row.serp_block_type, row.title, row.url, row.serp_block_rank_tb)
            if key not in seen_rows:
                seen_rows.add(key)
                items.append(row)
        for row in extract_paa_items(block, block_rank):
            key = (row.serp_block_type, row.title, row.url, row.serp_block_rank_tb)
            if key not in seen_rows:
                seen_rows.add(key)
                items.append(row)
        for row in extract_video_items(block, block_rank):
            key = (row.serp_block_type, row.title, row.url, row.serp_block_rank_tb)
            if key not in seen_rows:
                seen_rows.add(key)
                items.append(row)
        for row in extract_image_items(block, block_rank):
            key = (row.serp_block_type, row.title, row.url, row.serp_block_rank_tb)
            if key not in seen_rows:
                seen_rows.add(key)
                items.append(row)

    # Fallback for saved SERPs where results are present only in serialized JS payloads.
    max_rank = max((x.serp_block_rank_tb for x in items), default=0)
    for row in extract_wjd_items(html, start_block_rank=max_rank + 1):
        key = (row.serp_block_type, row.title, row.url, row.serp_block_rank_tb)
        if key not in seen_rows:
            seen_rows.add(key)
            items.append(row)

    return query, items


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
    st.title("SERP Parser (Streamlit)")
    st.caption("Estrae blocchi e contenuti da snapshot Google SERP (HTML, MHT/MHTML, WebArchive).")

    uploads = st.file_uploader(
        "Carica uno o più file",
        type=["html", "htm", "mht", "mhtml", "webarchive"],
        accept_multiple_files=True,
    )

    if not uploads:
        st.info("Carica almeno un file per iniziare.")
        return

    all_frames: list[pd.DataFrame] = []
    problems: list[str] = []

    for upload in uploads:
        try:
            file_bytes = upload.getvalue()
            html = load_html_from_upload(upload.name, file_bytes)
            query, items = parse_serp(html)
            browser = guess_browser(upload.name)
            frame = rows_to_dataframe(query=query, browser=browser, source_file=upload.name, items=items)
            all_frames.append(frame)
        except Exception as exc:
            problems.append(f"{upload.name}: {exc}")

    if problems:
        st.warning("Alcuni file non sono stati processati correttamente:")
        for problem in problems:
            st.write(f"- {problem}")

    if not all_frames:
        st.error("Nessun dato estratto dai file caricati.")
        return

    result = pd.concat(all_frames, ignore_index=True)

    st.subheader("Anteprima")
    st.dataframe(result, use_container_width=True, hide_index=True)

    csv_data = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Scarica CSV",
        data=csv_data,
        file_name="serp_output.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
