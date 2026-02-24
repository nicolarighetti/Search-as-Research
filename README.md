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

## App Modes

### 1) Auto Mode
Automatic extraction from source files:
- Input source files (required): `.html`, `.htm`, `.mht`, `.mhtml`, `.webarchive`
- Optional visual files: `.pdf`, `.png`, `.jpg`, `.jpeg`
- Optional PDF-based ranking refinement and visible-only filtering

### 2) Visual Assisted Mode
Human-in-the-loop annotation workflow:
- Upload one source file (`html/mht/webarchive`) and one screenshot (`pdf/png/jpg`)
- The app proposes block rectangles
- You can manually adjust rectangles and edit `serp_block_type` + `item_count`
- The app computes `serp_block_rank_tb` / `serp_block_rank_lr` from geometry
- Items are enriched from HTML and exported to CSV

This mode is recommended when automatic structure is not satisfactory.

## CSV Data Dictionary

| Field | Type | Description | Example |
|---|---|---|---|
| `query` | string | Search query inferred from page metadata/title. | `IA` |
| `browser` | string | Browser inferred from file name when possible. | `chrome` |
| `source_file` | string | Uploaded source file used for extraction. | `chrome html.html` |
| `serp_block_type` | string | SERP block family. Current values: `organic`, `people_also_ask`, `video_pack`, `image_pack`, `other`. | `organic` |
| `serp_block_rank_tb` | integer | Block order from top to bottom; in visual mode based on block Y coordinates. | `1` |
| `serp_block_rank_lr` | integer | Block order from left to right within the same vertical row. | `1` |
| `item_type` | string | Item category inside a block. Current values: `result`, `question`, `video`, `image`. | `result` |
| `item_rank_tb` | integer | Item order from top to bottom inside the block. | `2` |
| `item_rank_lr` | integer | Item order from left to right inside the block (notably used in image rows). | `1` |
| `is_expandable` | boolean-like string | Whether the item is expandable in SERP interaction terms (`TRUE` / `FALSE`). | `TRUE` |
| `title` | string | Main item title text. | `ChatGPT` |
| `description` | string | Item snippet/description text, when available. | `...` |
| `url` | string | Destination URL extracted from the item, when available. | `https://openai.com/it-IT/` |
| `domain` | string | Host/domain parsed from `url`. | `openai.com` |
| `notes` | string | Additional extraction notes (fallback path, visual metadata, manual-assisted output notes). | `visual_assisted` |

## Local Run (Optional)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dependency Note
The interactive visual canvas requires a compatible Streamlit/components pair.
This repository pins:
- `streamlit==1.31.1`
- `streamlit-drawable-canvas==0.9.3`

## Notes
- Expanded content is not reconstructed from non-expanded snapshots.
- PDF visual ranking and visual-assisted annotation are heuristic but significantly improve block structure control.
