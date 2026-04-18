# PlateScan 🚗

Sistema di schedatura targhe per comprensori — riconoscimento AI tramite Claude Vision.

## Stack
- **React 18** + **Vite 5**
- **Claude Sonnet** (vision) per il riconoscimento targhe
- `localStorage` per la persistenza dati (nessun backend)

## Setup locale

```bash
npm install
npm run dev
```

## Deploy su Netlify (via GitHub)

1. Esegui `git init && git add . && git commit -m "init"` in questa cartella
2. Crea un repo su GitHub e fai il push
3. Su [app.netlify.com](https://app.netlify.com) → **Add new site → Import from Git**
4. Seleziona il repo e usa queste impostazioni:
   - **Build command:** `npm run build`
   - **Publish directory:** `dist`
5. In **Site configuration → Environment variables** aggiungi:
   ```
   VITE_ANTHROPIC_API_KEY = sk-ant-...
   ```
6. Clicca **Deploy site** ✓

> ⚠️ Il file `netlify.toml` è già configurato con build e redirect SPA.

## Funzionalità

| Pagina | Descrizione |
|--------|-------------|
| 📷 Scansione | Apre fotocamera, scatta foto, riconosce targhe via AI |
| 📋 Archivio | Lista targhe con data primo/ultimo avvistamento e contatore passaggi. Esportabile in CSV |
| ✏️ Gestione | Correggi o elimina targhe riconosciute erroneamente |
