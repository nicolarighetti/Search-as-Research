# Search as Research

Tool Streamlit per estrarre una tabella CSV strutturata da snapshot SERP Google.

## Funzioni v1
- Upload file: `.html`, `.htm`, `.mht`, `.mhtml`, `.webarchive`
- Estrazione campi principali:
  - blocchi SERP
  - ordine alto/basso e sinistra/destra (v1: sinistra/destra centrata su layout base)
  - titolo, descrizione, URL, dominio
  - `is_expandable` (`TRUE`/`FALSE`)
- Anteprima tabellare e download CSV

## Avvio locale
1. Crea e attiva un virtual environment Python 3.10+
2. Installa dipendenze:
   ```bash
   pip install -r requirements.txt
   ```
3. Avvia app:
   ```bash
   streamlit run app.py
   ```

## Note
- La dimensione `espanso` non viene ricostruita da snapshot non espansi.
- In v1 i blocchi sono classificati con euristiche robuste ma conservative.
