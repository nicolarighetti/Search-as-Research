# Search as Research

Tool Streamlit per estrarre una tabella CSV strutturata da snapshot SERP Google.

## Uso web (senza installare nulla)
Puoi pubblicare l'app da GitHub su Streamlit Community Cloud (gratis):

1. Pusha il repository su GitHub (pubblico consigliato).
2. Vai su [share.streamlit.io](https://share.streamlit.io) e fai login.
3. `New app` -> seleziona questo repo.
4. Imposta `Main file path` = `app.py`.
5. Deploy.

Dopo il deploy gli utenti usano solo il link web, senza Python o dipendenze locali.

## Funzioni v1
- Upload file: `.html`, `.htm`, `.mht`, `.mhtml`, `.webarchive`
- Estrazione campi principali:
  - blocchi SERP
  - ordine alto/basso e sinistra/destra (v1: sinistra/destra centrata su layout base)
  - titolo, descrizione, URL, dominio
  - `is_expandable` (`TRUE`/`FALSE`)
- Anteprima tabellare e download CSV

## Avvio locale (opzionale)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Note
- La dimensione `espanso` non viene ricostruita da snapshot non espansi.
- In v1 i blocchi sono classificati con euristiche robuste ma conservative.
