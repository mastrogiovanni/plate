import { useState, useRef, useEffect } from "react";
import * as ort from "onnxruntime-web";

// ── Storage helpers ──────────────────────────────────────────────────────────
const STORAGE_KEY = "comprensorio_targhe";

function loadPlates() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"); }
  catch { return []; }
}
function savePlates(plates) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(plates));
}

// ── OCR (ONNX) ───────────────────────────────────────────────────────────────
const CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
let _session = null;

async function getSession() {
  if (!_session) {
    _session = await ort.InferenceSession.create("/models/cct_ocr.onnx");
  }
  return _session;
}

function preprocessCanvas(sourceCanvas, width = 160, height = 32) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(sourceCanvas, 0, 0, width, height);
  const { data } = ctx.getImageData(0, 0, width, height);
  const input = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    input[i] = (data[i * 4] + data[i * 4 + 1] + data[i * 4 + 2]) / 3 / 255.0;
  }
  return new ort.Tensor("float32", input, [1, 1, height, width]);
}

function ctcDecode(output) {
  const seq = output.data;
  const vocabSize = CHARS.length;
  const steps = seq.length / vocabSize;
  let result = "";
  let prev = -1;
  for (let i = 0; i < steps; i++) {
    let maxIdx = 0;
    let maxVal = -Infinity;
    for (let j = 0; j < vocabSize; j++) {
      const val = seq[i * vocabSize + j];
      if (val > maxVal) { maxVal = val; maxIdx = j; }
    }
    if (maxIdx !== prev) {
      result += CHARS[maxIdx] ?? "";
      prev = maxIdx;
    }
  }
  return result.trim();
}

async function recognizePlate(base64Image) {
  const session = await getSession();

  // Decode base64 into an ImageBitmap via a canvas
  const img = await new Promise((resolve, reject) => {
    const i = new Image();
    i.onload = () => resolve(i);
    i.onerror = reject;
    i.src = "data:image/jpeg;base64," + base64Image;
  });

  const srcCanvas = document.createElement("canvas");
  srcCanvas.width = img.naturalWidth;
  srcCanvas.height = img.naturalHeight;
  srcCanvas.getContext("2d").drawImage(img, 0, 0);

  const inputTensor = preprocessCanvas(srcCanvas);
  const results = await session.run({ input: inputTensor });
  const outputKey = Object.keys(results)[0];
  const plate = ctcDecode(results[outputKey]);

  if (!plate || plate.length < 4) {
    return { plates: [], confidence: "bassa", note: "Nessuna targa rilevata" };
  }
  return { plates: [plate], confidence: "alta", note: "" };
}

// ── Helpers ──────────────────────────────────────────────────────────────────
const confColor = { alta: "#00e5a0", media: "#f5c518", bassa: "#ff5c5c" };

function formatDate(iso) {
  if (!iso) return "—";
  return new Date(iso).toLocaleString("it-IT", {
    day: "2-digit", month: "2-digit", year: "numeric",
    hour: "2-digit", minute: "2-digit",
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
export default function App() {
  const [page, setPage]           = useState("scan");
  const [plates, setPlates]       = useState(loadPlates);
  const [cameraOn, setCameraOn]   = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [scanResult, setScanResult] = useState(null);
  const [flash, setFlash]         = useState(false);
  const [toast, setToast]         = useState(null);
  const [editTarget, setEditTarget] = useState(null);
  const [editValue, setEditValue]   = useState("");
  const [search, setSearch]         = useState("");

  const videoRef  = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => { savePlates(plates); }, [plates]);

  // ── Toast ─────────────────────────────────────────────────────────────────
  function showToast(msg, type = "ok") {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3000);
  }

  // ── Camera ────────────────────────────────────────────────────────────────
  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1920 } },
      });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setCameraOn(true);
    } catch {
      showToast("Impossibile accedere alla fotocamera", "err");
    }
  }

  function stopCamera() {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setCameraOn(false);
    setScanResult(null);
  }

  // ── Capture & analyse ─────────────────────────────────────────────────────
  async function capture() {
    if (!videoRef.current || !canvasRef.current || analyzing || capturing) return;
    setCapturing(true);
    setFlash(true);
    setTimeout(() => setFlash(false), 220);

    const video  = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);

    const photoUrl = canvas.toDataURL("image/jpeg", 0.92);
    const base64   = photoUrl.split(",")[1];

    setAnalyzing(true);
    setScanResult(null);

    try {
      const result = await recognizePlate(base64);
      setScanResult({ ...result, photoUrl });

      if (result.plates.length > 0) {
        const now = new Date().toISOString();
        setPlates((prev) => {
          const updated = [...prev];
          result.plates.forEach((rawP) => {
            const plate = rawP.toUpperCase().replace(/\s/g, "");
            const idx = updated.findIndex((x) => x.plate === plate);
            if (idx >= 0) {
              updated[idx] = {
                ...updated[idx],
                lastSeen: now,
                sightings: (updated[idx].sightings || 1) + 1,
              };
            } else {
              updated.unshift({ plate, firstSeen: now, lastSeen: now, sightings: 1 });
            }
          });
          return updated;
        });
        showToast(`${result.plates.length} targa/e registrata/e ✓`);
      } else {
        showToast("Nessuna targa trovata nell'immagine", "warn");
      }
    } catch {
      showToast("Errore durante l'analisi OCR", "err");
    } finally {
      setAnalyzing(false);
      setCapturing(false);
    }
  }

  // ── Edit / Delete ─────────────────────────────────────────────────────────
  function startEdit(p) { setEditTarget(p); setEditValue(p.plate); }

  function saveEdit() {
    const newPlate = editValue.toUpperCase().replace(/\s/g, "");
    if (!newPlate) return;
    setPlates((prev) =>
      prev.map((p) => p.plate === editTarget.plate ? { ...p, plate: newPlate } : p)
    );
    setEditTarget(null);
    showToast("Targa aggiornata ✓");
  }

  function deletePlate(plate) {
    if (!window.confirm(`Eliminare la targa ${plate}?`)) return;
    setPlates((prev) => prev.filter((p) => p.plate !== plate));
    setEditTarget(null);
    showToast("Targa eliminata");
  }

  function exportCSV() {
    const header = "Targa,Primo avvistamento,Ultimo avvistamento,Passaggi";
    const rows = plates.map((p) =>
      `${p.plate},${formatDate(p.firstSeen)},${formatDate(p.lastSeen)},${p.sightings || 1}`
    );
    const blob = new Blob([[header, ...rows].join("\n")], { type: "text/csv;charset=utf-8;" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href = url; a.download = "targhe.csv"; a.click();
    URL.revokeObjectURL(url);
  }

  // ── Filtered list ─────────────────────────────────────────────────────────
  const filtered = plates.filter((p) =>
    p.plate.toLowerCase().includes(search.toLowerCase())
  );

  // ═══════════════════════════════════════════════════════════════════════════
  return (
    <div style={S.root}>
      <div style={S.grid} />

      {/* Toast */}
      {toast && (
        <div style={{
          ...S.toast,
          background: toast.type === "err" ? "#ff5c5c" : toast.type === "warn" ? "#f5c518" : "#00e5a0",
          color: "#000",
        }}>
          {toast.msg}
        </div>
      )}

      {/* Header */}
      <header style={S.header}>
        <div style={S.logo}>
          <span style={S.logoIcon}>⬡</span>
          <span style={S.logoText}>PLATE<span style={S.logoAccent}>SCAN</span></span>
          <span style={S.logoBadge}>COMPRENSORIO</span>
        </div>
        <nav style={S.nav}>
          {[
            { id: "scan", label: "📷 SCANSIONE" },
            { id: "list", label: `📋 ARCHIVIO (${plates.length})` },
            { id: "edit", label: "✏️ GESTIONE" },
          ].map((n) => (
            <button
              key={n.id}
              style={{ ...S.navBtn, ...(page === n.id ? S.navBtnActive : {}) }}
              onClick={() => { setPage(n.id); if (n.id !== "scan") stopCamera(); }}
            >
              {n.label}
            </button>
          ))}
        </nav>
      </header>

      {/* ── PAGE: SCAN ── */}
      {page === "scan" && (
        <div style={S.page}>
          <div style={S.card}>
            <div style={S.cardHead}>
              <span style={S.sectionTitle}>RILEVAMENTO TARGHE</span>
              <span style={{ ...S.pill, ...(cameraOn ? S.pillGreen : {}) }}>
                {cameraOn ? "● LIVE" : "○ STANDBY"}
              </span>
            </div>

            {/* Viewfinder */}
            <div style={S.viewfinder}>
              {flash && <div style={S.flashOverlay} />}
              <video
                ref={videoRef} autoPlay playsInline muted
                style={{ ...S.video, display: cameraOn ? "block" : "none" }}
              />
              <canvas ref={canvasRef} style={{ display: "none" }} />

              {!cameraOn && (
                <div style={S.placeholder}>
                  <div style={S.reticle}>
                    <div style={S.reticleH} /><div style={S.reticleV} />
                    <div style={S.reticleCircle} />
                  </div>
                  <p style={S.placeholderText}>FOTOCAMERA INATTIVA</p>
                  <p style={S.placeholderSub}>Premi AVVIA per iniziare la scansione</p>
                </div>
              )}

              {/* Corner brackets */}
              {["tl","tr","bl","br"].map((c) => (
                <div key={c} style={{ ...S.corner, ...S[`c_${c}`] }} />
              ))}
            </div>

            {/* Controls */}
            <div style={S.controls}>
              {!cameraOn ? (
                <button style={S.btnPrimary} onClick={startCamera}>
                  ▶ &nbsp;AVVIA FOTOCAMERA
                </button>
              ) : (
                <>
                  <button style={S.btnDanger} onClick={stopCamera}>■ &nbsp;STOP</button>
                  <button
                    style={{ ...S.btnCapture, ...(analyzing ? S.btnDisabled : {}) }}
                    onClick={capture}
                    disabled={analyzing || capturing}
                  >
                    {analyzing
                      ? <><span style={S.spin}>◌</span> &nbsp;ANALISI OCR...</>
                      : "◉ &nbsp;SCATTA E ANALIZZA"}
                  </button>
                </>
              )}
            </div>

            {/* Result */}
            {scanResult && (
              <div style={S.result}>
                <div style={S.resultHead}>
                  <span style={S.resultLabel}>RISULTATO SCANSIONE</span>
                  <span style={{
                    ...S.confBadge,
                    background: confColor[scanResult.confidence] || "#888",
                  }}>
                    CONFIDENZA: {scanResult.confidence?.toUpperCase()}
                  </span>
                </div>

                {scanResult.plates.length > 0 ? (
                  <div style={S.platesRow}>
                    {scanResult.plates.map((p, i) => (
                      <div key={i} style={S.plateChip}>{p.toUpperCase()}</div>
                    ))}
                  </div>
                ) : (
                  <p style={S.noPlate}>⚠ Nessuna targa rilevata nell'immagine</p>
                )}

                {scanResult.note && <p style={S.note}>ℹ {scanResult.note}</p>}

                {scanResult.photoUrl && (
                  <img src={scanResult.photoUrl} style={S.thumb} alt="scatto" />
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── PAGE: LIST ── */}
      {page === "list" && (
        <div style={S.page}>
          <div style={S.card}>
            <div style={S.cardHead}>
              <span style={S.sectionTitle}>ARCHIVIO AVVISTAMENTI</span>
              <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                <span style={S.pill}>{plates.length} targhe</span>
                {plates.length > 0 && (
                  <button style={S.btnExport} onClick={exportCSV}>⬇ CSV</button>
                )}
              </div>
            </div>

            {/* Search */}
            {plates.length > 0 && (
              <div style={S.searchBar}>
                <input
                  style={S.searchInput}
                  placeholder="🔍  Cerca targa..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                />
              </div>
            )}

            {plates.length === 0 ? (
              <div style={S.empty}>
                <div style={S.emptyIcon}>◈</div>
                <p>Nessuna targa registrata.<br />Vai in Scansione per iniziare.</p>
              </div>
            ) : filtered.length === 0 ? (
              <div style={S.empty}><p>Nessuna targa corrisponde alla ricerca.</p></div>
            ) : (
              <>
                <div style={S.tableHead}>
                  <span>TARGA</span>
                  <span>PRIMO AVVISTAMENTO</span>
                  <span>ULTIMO AVVISTAMENTO</span>
                  <span style={{ textAlign: "center" }}>N°</span>
                </div>
                {filtered.map((p) => (
                  <div key={p.plate} style={S.tableRow}>
                    <span style={S.plateTag}>{p.plate}</span>
                    <span style={S.cell}>{formatDate(p.firstSeen)}</span>
                    <span style={S.cell}>{formatDate(p.lastSeen)}</span>
                    <span style={{ ...S.cell, textAlign: "center", color: "#00e5a0", fontWeight: 700 }}>
                      {p.sightings || 1}
                    </span>
                  </div>
                ))}
              </>
            )}
          </div>
        </div>
      )}

      {/* ── PAGE: EDIT ── */}
      {page === "edit" && (
        <div style={S.page}>
          <div style={S.card}>
            <div style={S.cardHead}>
              <span style={S.sectionTitle}>GESTIONE TARGHE</span>
              <span style={S.pill}>correggi / elimina</span>
            </div>

            {plates.length === 0 ? (
              <div style={S.empty}>
                <div style={S.emptyIcon}>◈</div>
                <p>Nessuna targa da gestire.</p>
              </div>
            ) : (
              <div style={S.editGrid}>
                {plates.map((p) => (
                  <div key={p.plate} style={S.editCard}>
                    {editTarget?.plate === p.plate ? (
                      <div style={S.editForm}>
                        <label style={S.editLabel}>MODIFICA TARGA</label>
                        <input
                          style={S.editInput}
                          value={editValue}
                          onChange={(e) => setEditValue(e.target.value.toUpperCase())}
                          maxLength={10}
                          autoFocus
                        />
                        <div style={S.editBtns}>
                          <button style={S.btnSave} onClick={saveEdit}>✓ SALVA</button>
                          <button style={S.btnDel} onClick={() => deletePlate(p.plate)}>✕ ELIMINA</button>
                          <button style={S.btnCancel} onClick={() => setEditTarget(null)}>✗ ANNULLA</button>
                        </div>
                      </div>
                    ) : (
                      <div style={S.editCardInner}>
                        <div style={S.plateTag}>{p.plate}</div>
                        <div style={S.editMeta}>
                          <span>Avvistamenti: <b style={{ color: "#00e5a0" }}>{p.sightings || 1}</b></span>
                          <span>Primo: {formatDate(p.firstSeen)}</span>
                          <span>Ultimo: {formatDate(p.lastSeen)}</span>
                        </div>
                        <button style={S.btnEdit} onClick={() => startEdit(p)}>✏ MODIFICA</button>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Footer */}
      <footer style={S.footer}>
        PlateScan — Sistema di schedatura targhe &nbsp;|&nbsp; Dati salvati localmente nel browser
      </footer>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// STYLES
// ═══════════════════════════════════════════════════════════════════════════════
const C = {
  bg:      "#0a0c10",
  surface: "#12161e",
  border:  "#1e2535",
  accent:  "#00e5a0",
  warn:    "#f5c518",
  danger:  "#ff5c5c",
  text:    "#e8eaf0",
  muted:   "#5a6478",
};

const S = {
  root: {
    minHeight: "100vh", background: C.bg, color: C.text,
    fontFamily: "'Courier New', 'Lucida Console', monospace",
    position: "relative", overflowX: "hidden",
    display: "flex", flexDirection: "column",
  },
  grid: {
    position: "fixed", inset: 0, pointerEvents: "none", zIndex: 0,
    backgroundImage: `linear-gradient(${C.border} 1px,transparent 1px),linear-gradient(90deg,${C.border} 1px,transparent 1px)`,
    backgroundSize: "40px 40px", opacity: 0.35,
  },
  toast: {
    position: "fixed", top: 20, left: "50%", transform: "translateX(-50%)",
    padding: "10px 24px", borderRadius: 4, fontWeight: 700, fontSize: 13,
    zIndex: 9999, letterSpacing: "0.1em", boxShadow: "0 4px 20px rgba(0,0,0,.5)",
    whiteSpace: "nowrap",
  },
  header: {
    position: "sticky", top: 0, zIndex: 100,
    display: "flex", alignItems: "center", justifyContent: "space-between",
    padding: "14px 24px",
    background: `${C.bg}ee`,
    borderBottom: `1px solid ${C.border}`,
    backdropFilter: "blur(10px)",
    flexWrap: "wrap", gap: 12,
  },
  logo: { display: "flex", alignItems: "center", gap: 10 },
  logoIcon: { fontSize: 22, color: C.accent },
  logoText: { fontSize: 20, fontWeight: 700, letterSpacing: "0.2em" },
  logoAccent: { color: C.accent },
  logoBadge: {
    fontSize: 9, letterSpacing: "0.2em", color: C.muted,
    border: `1px solid ${C.border}`, padding: "2px 7px", borderRadius: 3,
  },
  nav: { display: "flex", gap: 6, flexWrap: "wrap" },
  navBtn: {
    background: "transparent", border: `1px solid ${C.border}`,
    color: C.muted, padding: "7px 14px", cursor: "pointer",
    fontSize: 11, letterSpacing: "0.1em", borderRadius: 3,
    fontFamily: "inherit", transition: "all .2s",
  },
  navBtnActive: { background: `${C.accent}18`, borderColor: C.accent, color: C.accent },

  page: {
    position: "relative", zIndex: 1,
    maxWidth: 960, margin: "0 auto", padding: "24px 16px", flex: 1,
    width: "100%",
  },
  card: {
    background: C.surface, border: `1px solid ${C.border}`,
    borderRadius: 8, overflow: "hidden",
  },
  cardHead: {
    display: "flex", alignItems: "center", justifyContent: "space-between",
    padding: "14px 20px", borderBottom: `1px solid ${C.border}`,
  },
  sectionTitle: { fontSize: 13, fontWeight: 700, letterSpacing: "0.18em", color: C.accent },
  pill: {
    fontSize: 11, letterSpacing: "0.1em",
    background: `${C.accent}18`, color: C.accent,
    padding: "4px 10px", borderRadius: 20, border: `1px solid ${C.accent}40`,
  },
  pillGreen: { background: `${C.accent}30`, animation: "pulse 2s infinite" },

  // Viewfinder
  viewfinder: {
    position: "relative", width: "100%", aspectRatio: "16/9",
    background: "#000", overflow: "hidden", maxHeight: 500,
  },
  video: { width: "100%", height: "100%", objectFit: "cover", display: "block" },
  flashOverlay: {
    position: "absolute", inset: 0, background: "#fff",
    opacity: 0.85, zIndex: 10, pointerEvents: "none",
    animation: "flashOut .22s ease-out forwards",
  },
  placeholder: {
    position: "absolute", inset: 0,
    display: "flex", flexDirection: "column",
    alignItems: "center", justifyContent: "center", gap: 12,
    background: "#050709",
  },
  reticle: { position: "relative", width: 70, height: 70 },
  reticleH: { position: "absolute", top: "50%", left: 0, right: 0, height: 1, background: `${C.accent}50` },
  reticleV: { position: "absolute", left: "50%", top: 0, bottom: 0, width: 1, background: `${C.accent}50` },
  reticleCircle: {
    position: "absolute", inset: 16, borderRadius: "50%",
    border: `1px solid ${C.accent}40`,
  },
  placeholderText: { fontSize: 11, letterSpacing: "0.22em", color: C.muted, margin: 0 },
  placeholderSub: { fontSize: 10, letterSpacing: "0.1em", color: `${C.muted}80`, margin: 0 },

  corner: { position: "absolute", width: 22, height: 22, zIndex: 5 },
  c_tl: { top: 14, left: 14, borderTop: `2px solid ${C.accent}`, borderLeft: `2px solid ${C.accent}` },
  c_tr: { top: 14, right: 14, borderTop: `2px solid ${C.accent}`, borderRight: `2px solid ${C.accent}` },
  c_bl: { bottom: 14, left: 14, borderBottom: `2px solid ${C.accent}`, borderLeft: `2px solid ${C.accent}` },
  c_br: { bottom: 14, right: 14, borderBottom: `2px solid ${C.accent}`, borderRight: `2px solid ${C.accent}` },

  controls: {
    display: "flex", gap: 12, padding: "16px 20px",
    borderBottom: `1px solid ${C.border}`, flexWrap: "wrap",
  },
  btnPrimary: {
    flex: 1, padding: "13px 24px", background: C.accent, color: "#000",
    border: "none", borderRadius: 4, cursor: "pointer",
    fontSize: 13, fontWeight: 700, letterSpacing: "0.15em", fontFamily: "inherit",
    minWidth: 200,
  },
  btnDanger: {
    padding: "13px 20px", background: "transparent",
    border: `1px solid ${C.danger}`, color: C.danger,
    borderRadius: 4, cursor: "pointer", fontSize: 13,
    fontWeight: 700, letterSpacing: "0.12em", fontFamily: "inherit",
  },
  btnCapture: {
    flex: 1, padding: "13px 24px", background: C.accent, color: "#000",
    border: "none", borderRadius: 4, cursor: "pointer",
    fontSize: 13, fontWeight: 700, letterSpacing: "0.15em", fontFamily: "inherit",
    display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
    minWidth: 220,
  },
  btnDisabled: { opacity: 0.5, cursor: "not-allowed" },
  spin: { display: "inline-block", animation: "spin 1s linear infinite" },

  // Result panel
  result: { padding: 20 },
  resultHead: { display: "flex", alignItems: "center", gap: 12, marginBottom: 14, flexWrap: "wrap" },
  resultLabel: { fontSize: 11, letterSpacing: "0.18em", color: C.muted },
  confBadge: {
    fontSize: 10, fontWeight: 700, letterSpacing: "0.12em",
    padding: "3px 10px", borderRadius: 20, color: "#000",
  },
  platesRow: { display: "flex", flexWrap: "wrap", gap: 10, marginBottom: 14 },
  plateChip: {
    background: "#000", border: `2px solid ${C.accent}`,
    color: C.accent, padding: "10px 22px", borderRadius: 4,
    fontSize: 24, fontWeight: 700, letterSpacing: "0.35em",
  },
  noPlate: { color: C.muted, fontSize: 14, marginBottom: 12 },
  note: { color: C.muted, fontSize: 12, marginBottom: 12, letterSpacing: "0.05em" },
  thumb: {
    maxWidth: 320, width: "100%", borderRadius: 4,
    border: `1px solid ${C.border}`, display: "block", marginTop: 8,
  },

  // Search
  searchBar: { padding: "14px 20px", borderBottom: `1px solid ${C.border}` },
  searchInput: {
    width: "100%", background: "#000", border: `1px solid ${C.border}`,
    color: C.text, padding: "9px 14px", borderRadius: 4,
    fontSize: 13, fontFamily: "inherit", outline: "none",
    letterSpacing: "0.05em",
  },

  // Table
  tableHead: {
    display: "grid", gridTemplateColumns: "1fr 1.4fr 1.4fr 0.4fr",
    padding: "10px 20px", fontSize: 10, letterSpacing: "0.15em",
    color: C.muted, borderBottom: `1px solid ${C.border}`,
  },
  tableRow: {
    display: "grid", gridTemplateColumns: "1fr 1.4fr 1.4fr 0.4fr",
    padding: "12px 20px", borderBottom: `1px solid ${C.border}22`,
    alignItems: "center",
  },
  plateTag: { fontWeight: 700, fontSize: 15, letterSpacing: "0.22em", color: C.accent },
  cell: { fontSize: 12, color: C.muted },

  // Edit grid
  editGrid: {
    display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(270px,1fr))",
    gap: 12, padding: 16,
  },
  editCard: {
    background: C.bg, border: `1px solid ${C.border}`,
    borderRadius: 6, padding: 16,
  },
  editCardInner: { display: "flex", flexDirection: "column", gap: 10 },
  editMeta: {
    display: "flex", flexDirection: "column", gap: 3,
    fontSize: 11, color: C.muted,
  },
  editForm: { display: "flex", flexDirection: "column", gap: 10 },
  editLabel: { fontSize: 10, letterSpacing: "0.18em", color: C.muted },
  editInput: {
    background: "#000", border: `1px solid ${C.accent}`,
    color: C.accent, padding: "10px 14px", borderRadius: 4,
    fontSize: 20, fontWeight: 700, letterSpacing: "0.3em",
    fontFamily: "inherit", width: "100%", outline: "none",
  },
  editBtns: { display: "flex", gap: 6, flexWrap: "wrap" },
  btnEdit: {
    background: "transparent", border: `1px solid ${C.border}`,
    color: C.muted, padding: "7px 14px", borderRadius: 4,
    cursor: "pointer", fontSize: 11, letterSpacing: "0.1em",
    fontFamily: "inherit", alignSelf: "flex-start",
  },
  btnSave: {
    flex: 1, background: C.accent, color: "#000", border: "none",
    padding: "8px 12px", borderRadius: 4, cursor: "pointer",
    fontSize: 11, fontWeight: 700, letterSpacing: "0.1em", fontFamily: "inherit",
  },
  btnDel: {
    flex: 1, background: "transparent", border: `1px solid ${C.danger}`,
    color: C.danger, padding: "8px 12px", borderRadius: 4,
    cursor: "pointer", fontSize: 11, fontWeight: 700, fontFamily: "inherit",
  },
  btnCancel: {
    flex: 1, background: "transparent", border: `1px solid ${C.border}`,
    color: C.muted, padding: "8px 12px", borderRadius: 4,
    cursor: "pointer", fontSize: 11, fontFamily: "inherit",
  },
  btnExport: {
    background: "transparent", border: `1px solid ${C.border}`,
    color: C.muted, padding: "5px 12px", borderRadius: 4,
    cursor: "pointer", fontSize: 11, fontFamily: "inherit", letterSpacing: "0.08em",
  },

  // Empty
  empty: {
    textAlign: "center", padding: "60px 20px",
    color: C.muted, fontSize: 14, letterSpacing: "0.05em", lineHeight: 1.8,
  },
  emptyIcon: { fontSize: 48, marginBottom: 16, opacity: 0.3 },

  footer: {
    position: "relative", zIndex: 1,
    textAlign: "center", padding: "20px",
    fontSize: 10, letterSpacing: "0.12em", color: C.muted,
    borderTop: `1px solid ${C.border}`,
  },
};
