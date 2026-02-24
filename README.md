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

## v1 Features
- File upload support: `.html`, `.htm`, `.mht`, `.mhtml`, `.webarchive`
- SERP extraction includes:
  - SERP blocks
  - top-to-bottom and left-to-right ordering (v1 heuristic)
  - title, description, URL, domain
  - `is_expandable` (`TRUE` / `FALSE`)
- Table preview and CSV download

## Local Run (Optional)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## CSV Data Dictionary
The app exports a CSV with the following fields:

| Field | Type | Description | Example |
|---|---|---|---|
| `query` | string | Search query inferred from page metadata/title. | `IA` |
| `browser` | string | Browser inferred from file name when possible. | `chrome` |
| `source_file` | string | Uploaded file name used for extraction. | `chrome html.html` |
| `serp_block_type` | string | SERP block family. Current values: `organic`, `people_also_ask`, `video_pack`, `image_pack`, `other`. | `people_also_ask` |
| `serp_block_rank_tb` | integer | Block order from top to bottom in the page. | `3` |
| `serp_block_rank_lr` | integer | Block order from left to right within the same level (v1 heuristic). | `1` |
| `item_type` | string | Item category inside a block. Current values: `result`, `question`, `video`, `image`. | `question` |
| `item_rank_tb` | integer | Item order from top to bottom inside a block. | `2` |
| `item_rank_lr` | integer | Item order from left to right inside a block. | `1` |
| `is_expandable` | boolean-like string | Whether the item is expandable in SERP interaction terms (`TRUE` / `FALSE`). | `TRUE` |
| `title` | string | Main item title text. | `How can I use AI for free?` |
| `description` | string | Item snippet/description text, when available. | `...` |
| `url` | string | Destination URL extracted from the item, when available. | `https://openai.com/it-IT/` |
| `domain` | string | Host/domain parsed from `url`. | `openai.com` |
| `notes` | string | Additional extraction notes (for traceability/fallback paths). | `PAA item` |

## Notes
- Expanded content is not reconstructed from non-expanded snapshots.
- v1 uses conservative heuristics; output is designed for iterative validation and refinement.
