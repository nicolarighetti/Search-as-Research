# Search as Research

Streamlit app that extracts a structured CSV table from saved Google SERP snapshots.

## Web Deployment (No User-Side Install)
You can publish this app from GitHub to Streamlit Community Cloud (free):

1. Push this repository to GitHub (public is recommended).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
3. Click `New app` and select this repository.
4. Set `Main file path` to `app.py`.
5. Deploy.

Once deployed, end users only need the app URL. They do not need Python or local dependencies.

## Inputs
- SERP source files (required): `.html`, `.htm`, `.mht`, `.mhtml`, `.webarchive`
- Visual files (optional): `.pdf`, `.png`, `.jpg`, `.jpeg`

## Current Visual Ranking Support
- `PDF`: supported (extracts approximate text coordinates from page 1 and reorders SERP rows by visual position)
- `PNG/JPG`: accepted as upload, but not parsed yet for automatic ranking

Recommended mode for better `top-bottom` / `left-right` ordering: upload both HTML and the matching PDF snapshot.

## Features
- Content extraction from SERP source files:
  - `serp_block_type`
  - `title`, `description`, `url`, `domain`
  - `is_expandable` (`TRUE` / `FALSE`)
- Rank assignment:
  - `serp_block_rank_tb` and `item_rank_tb`
  - `serp_block_rank_lr` and `item_rank_lr`
- Optional visual refinement from PDF for more reliable ordering
- Table preview and CSV export

## Local Run (Optional)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## CSV Data Dictionary

| Field | Type | Description | Example |
|---|---|---|---|
| `query` | string | Search query inferred from page metadata/title. | `IA` |
| `browser` | string | Browser inferred from file name when possible. | `chrome` |
| `source_file` | string | Uploaded source file used for extraction. | `chrome html.html` |
| `serp_block_type` | string | SERP block family. Current values: `organic`, `people_also_ask`, `video_pack`, `image_pack`, `other`. | `organic` |
| `serp_block_rank_tb` | integer | Block order from top to bottom in the final scan sequence. | `1` |
| `serp_block_rank_lr` | integer | Block order from left to right in the same vertical band. | `1` |
| `item_type` | string | Item category inside a block. Current values: `result`, `question`, `video`, `image`. | `result` |
| `item_rank_tb` | integer | Item order from top to bottom inside the block. | `2` |
| `item_rank_lr` | integer | Item order from left to right inside the block (notably used in image rows). | `1` |
| `is_expandable` | boolean-like string | Whether the item is expandable in SERP interaction terms (`TRUE` / `FALSE`). | `TRUE` |
| `title` | string | Main item title text. | `ChatGPT` |
| `description` | string | Item snippet/description text, when available. | `...` |
| `url` | string | Destination URL extracted from the item, when available. | `https://openai.com/it-IT/` |
| `domain` | string | Host/domain parsed from `url`. | `openai.com` |
| `notes` | string | Additional extraction notes (fallback path, visual ranking metadata, etc.). | `visual ranking from ...` |

## Notes
- Expanded content is not reconstructed from non-expanded snapshots.
- Visual ranking from PDF is heuristic and depends on title matching quality.
