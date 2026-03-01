# Search as Research

Streamlit app that extracts a structured CSV table from saved Google SERP source files.

## Online App
The app is currently online at:

- [https://search-as-research.streamlit.app](https://search-as-research.streamlit.app)

## Scope
- Input files: `.html`, `.htm`, `.mht`, `.mhtml`, `.webarchive`
- Output: editable tabular data + CSV export
- Visual-assisted mode and PDF/screenshot processing are intentionally removed for stability.

## Manual Validation Requirement
Hierarchy fields are generated heuristically and must be manually checked and edited:
- `serp_block_rank_tb`
- `serp_block_rank_lr`
- `item_rank_tb`
- `item_rank_lr`
- optionally `serp_block_type`

This is a **mandatory** step before using the CSV in analysis.
The numeric hierarchy must be reviewed and corrected manually for each `source_file` and each block/item.
In particular, make sure the numeric ordering reflects the intended source hierarchy in the page:
- top-to-bottom block order (`serp_block_rank_tb`)
- left-to-right block order (`serp_block_rank_lr`)
- top-to-bottom item order inside each block (`item_rank_tb`)
- left-to-right item order inside each block (`item_rank_lr`)

The app exposes an editable table before CSV download for this exact manual validation step.

## AI Summary Extraction
The parser attempts to capture AI-generated summary content under `serp_block_type = ai_summary`.
This extraction is heuristic and may require manual correction.

## CSV Data Dictionary

| Field | Type | Description | Example |
|---|---|---|---|
| `query` | string | Search query inferred from page metadata/title. | `IA` |
| `browser` | string | Browser inferred from file name when possible. | `chrome` |
| `source_file` | string | Uploaded source file used for extraction. | `chrome html.html` |
| `serp_block_type` | string | SERP block family. Current values include `organic`, `ai_summary`, `people_also_ask`, `video_pack`, `image_pack`, `other`. | `ai_summary` |
| `serp_block_rank_tb` | integer | Block order from top to bottom (heuristic; manual check required). | `1` |
| `serp_block_rank_lr` | integer | Block order from left to right (heuristic; manual check required). | `1` |
| `item_type` | string | Item category inside a block. | `summary` |
| `item_rank_tb` | integer | Item order from top to bottom inside the block (heuristic; manual check required). | `1` |
| `item_rank_lr` | integer | Item order from left to right inside the block (heuristic; manual check required). | `1` |
| `is_expandable` | boolean-like string | Whether the item is expandable in SERP interaction terms (`TRUE` / `FALSE`). | `FALSE` |
| `title` | string | Main item title text. | `AI Summary` |
| `description` | string | Item snippet/summary text, when available. | `...` |
| `url` | string | Destination URL extracted from the item, when available. | `https://openai.com/` |
| `domain` | string | Host/domain parsed from `url`. | `openai.com` |
| `notes` | string | Additional extraction notes and heuristics metadata. | `heuristic_ai_summary; verify manually` |

## Local Run (Optional)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Web Deployment (No User-Side Install)
You can publish this app from GitHub to Streamlit Community Cloud (free):

1. Push this repository to GitHub (public is recommended).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
3. Click `New app` and select this repository.
4. Set `Main file path` to `app.py`.
5. Deploy.

Once deployed, end users only need the app URL. They do not need Python or local dependencies.
