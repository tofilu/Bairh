# ──────────────────────────────────────────────────────────────────────────────
# Emoji Recommender – Backend (FastAPI + ChromaDB + optional Ollama)
# ──────────────────────────────────────────────────────────────────────────────
# Dieses Backend lädt eine CSV mit Emojis, erzeugt Embeddings via
# Sentence-Transformer ("all-MiniLM-L6-v2") und speichert sie in einer
# lokalen ChromaDB. Ein POST-Endpoint /recommend sucht die ähnlichsten
# Emojis zu einem gegebenen Satz. Falls der Top-Score < 0.8 ist, wird
# optional ein LLM (qwen2.5:0.5b über Ollama) zur Verfeinerung genutzt.
# ──────────────────────────────────────────────────────────────────────────────

import logging
import os
import re
import sys

import chromadb
import pandas as pd
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Logging ──────────────────────────────────────────────────────────────────
# Loggt auf der Konsole mit Zeitstempel, Level und Nachricht.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Pfade ────────────────────────────────────────────────────────────────────
# BASE_DIR   = backend/                  (wo main.py liegt)
# PROJECT_DIR = emoji_app/               (ein Level über backend/)
# CSV_PATH   = backend/emoji_dataset.csv (muss vom Nutzer bereitgestellt werden)
# CHROMA_DB  = emoji_db/                 (wird von ChromaDB automatisch erstellt)
# FRONTEND   = frontend/index.html       (wird unter / ausgeliefert)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
CSV_PATH = os.path.join(BASE_DIR, "emoji_dataset.csv")
CHROMA_DB_PATH = os.path.join(PROJECT_DIR, "emoji_db")
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")

# ── FastAPI-App ──────────────────────────────────────────────────────────────
# CORS ist freigegeben ("*"), damit das Frontend aus dem Browser
# (auch per file://) auf das Backend zugreifen kann.
app = FastAPI(title="Emoji Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Frontend ausliefern ──────────────────────────────────────────────────────
# Die index.html wird direkt unter der Root-URL (/) serviert.
@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# ── Embedding-Funktion ───────────────────────────────────────────────────────
# Nutzt Sentence-Transformers ("all-MiniLM-L6-v2") – ein leichtes,
# CPU-only Modell, das Sätze in 384-dimensionale Vektoren umwandelt.
embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

# ── ChromaDB-Client ──────────────────────────────────────────────────────────
# PersistentClient speichert die Vektordatenbank im Verzeichnis emoji_db/.
# Die Collection "emojis" wird mit der Embedding-Funktion verknüpft,
# sodass Query-Text automatisch geembedded wird.
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(
    name="emojis",
    embedding_function=embedding_function,  # type: ignore[arg-type]
)


# ── Pydantic-Modelle (API-Vertrag) ───────────────────────────────────────────
# Definieren das JSON-Format für Request und Response.

class RecommendRequest(BaseModel):
    text: str
    use_llm: bool = True

class EmojiCandidate(BaseModel):
    emoji: str       # Das Emoji-Zeichen, z. B. "🍕"
    description: str  # Bereinigte Beschreibung, z. B. "pizza"
    score: float      # Ähnlichkeitsscore (0.0 – 1.0)

class RecommendResponse(BaseModel):
    recommended: str            # Das finale, empfohlene Emoji
    all_candidates: list[EmojiCandidate]  # Top-3 Kandidaten mit Scores
    llm_used: bool              # Wurde das LLM zur Verfeinerung genutzt?


# ── Hilfsfunktionen ──────────────────────────────────────────────────────────

def _clean_description(desc: str) -> str:
    """
    Entfernt den Unicode-Version-Präfix aus den Beschreibungen.
    Beispiel: "E0.6 pizza" → "pizza",  "E1.0 grinning face" → "grinning face"
    """
    return re.sub(r"^E[\d.]+\s*", "", desc).strip()


def _init_database():
    """
    Wird beim Start aufgerufen. Prüft, ob die Collection bereits Daten enthält.
    Falls nicht, wird die CSV geladen und zeilenweise in ChromaDB eingefügt.
    Die Emoji-Beschreibungen werden als Documents (zu embedden) verwendet,
    das Emoji-Zeichen selbst wird als Metadatum gespeichert.
    """
    # Bereits initialisiert? Dann nichts tun.
    if collection.count() > 0:
        log.info("Collection enthält bereits %d Einträge", collection.count())
        return

    # CSV vorhanden? Sonst klares Fehler-Log + Abbruch.
    if not os.path.isfile(CSV_PATH):
        log.error(
            "CSV-Datei nicht gefunden: %s\n"
            "Bitte lege die Datei 'emoji_dataset.csv' im backend/ Verzeichnis ab.",
            CSV_PATH,
        )
        sys.exit(1)

    # CSV einlesen
    log.info("Lade CSV: %s", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    log.info("CSV geladen: %d Zeilen", len(df))

    # IDs: Zeilenindex als String (eindeutig)
    ids = [str(i) for i in range(len(df))]
    # Documents: die bereinigten Beschreibungen (werden geembedded)
    documents = df["description"].apply(_clean_description).tolist()
    # Metadaten: das Emoji-Zeichen (wird später im Ergebnis verwendet)
    metadatas = [{"emoji": emoji} for emoji in df["emoji"].tolist()]

    # Batch-Weise einfügen (je 100 Einträge), um Speicher zu schonen
    batch_size = 100
    for i in range(0, len(df), batch_size):
        end = min(i + batch_size, len(df))
        collection.add(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],  # type: ignore[arg-type]
        )
        log.info("Batch %d\u2013%d von %d hinzugef\u00fcgt", i, end, len(df))

    log.info("Datenbank initialisiert: %d Einträge", collection.count())


def _extract_emoji(text: str) -> str | None:
    """
    Extrahiert das erste Emoji-Zeichen aus einem beliebigen String.
    Nutzt Unicode-Bereiche für die wichtigsten Emoji-Blöcke.
    Gibt None zurück, wenn kein Emoji gefunden wird.
    """
    # Emoji-Regex: erfasst einzelne Emoji-Zeichen inkl. ZWJ-Sequenzen.
    # Erklärung der Blöcke:
    #   \U0001F300-\U0001F9FF  – Haupt-Emoji-Blöcke (Symbole, Smileys, Aktivitäten)
    #   \U00002600-\U000027BF  – Diverse Unicode-Symbole (⭐☀☁☂ etc.)
    #   \U0000FE00-\U0000FE0F  – Variationsselektoren (farbige Darstellung)
    #   \U0000200D             – Zero-Width-Joiner (für ZWJ-Sequenzen wie 👨‍👩‍👧)
    emoji_pattern = re.compile(
        "[\U0001F300-\U0001F9FF"
        "\U00002600-\U000027BF"
        "\U0000FE00-\U0000FE0F"
        "\U0000200D"
        "\U0001F1E0-\U0001F1FF"
        "]+",
        re.UNICODE,
    )
    match = emoji_pattern.search(text)
    return match.group() if match else None


def _query_ollama(prompt: str) -> str | None:
    """
    Sendet einen Prompt an das lokale Ollama-Modell (qwen2.5:0.5b).
    Erwartet ein einzelnes Emoji-Zeichen als Antwort.
    Bei Fehlern (Ollama nicht installiert, Modell fehlt, Timeout)
    wird None zurückgegeben und die App fällt auf Embedding-Ergebnisse zurück.
    """
    try:
        import ollama

        response = ollama.chat(
            model="qwen2.5:0.5b",
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 10},
        )
        raw = response["message"]["content"].strip()
        log.info("Ollama rohe Antwort: %s", raw)

        # Versuche, ein Emoji aus der Antwort zu extrahieren
        # (das Modell hält sich nicht immer an die "NUR Emoji"-Anweisung)
        emoji = _extract_emoji(raw)
        return emoji if emoji else None
    except Exception as e:
        log.warning("Ollama nicht verfügbar (%s). Verwende Embedding-Ergebnisse.", e)
        return None


# ── Startup-Event ────────────────────────────────────────────────────────────
# Beim ersten Start wird die Datenbank initialisiert.
# Bei Folgestarts ist die Collection bereits gefüllt und der Schritt wird
# übersprungen (siehe Prüfung in _init_database).
@app.on_event("startup")
def startup():
    _init_database()


# ── Endpunkt: Health-Check ───────────────────────────────────────────────────
# Einfacher GET-Endpunkt, um zu prüfen, ob der Server läuft
# und wie viele Einträge in der Datenbank sind.
@app.get("/health")
def health():
    return {"status": "ok", "collection_size": collection.count()}


# ── Endpunkt: Emoji-Empfehlung ───────────────────────────────────────────────
# Der Kern der Anwendung.
#
# 1. Stufe (Embedding):
#   Der eingegebene Text wird mit dem Sentence-Transformer geembedded
#   und in ChromaDB die 3 ähnlichsten Emojis gesucht.
#   Die ChromaDB-Distanz wird in einen Score umgerechnet: 1/(1+distance).
#
# 2. Stufe (LLM, optional, per `use_llm`-Flag schaltbar):
#   Wenn der Nutzer es wünscht und der beste Score < 0.84 ist, wird das
#   LLM gebeten, aus den drei Kandidaten das passendste auszuwählen.
#   Bei Score >= 0.84 gilt die Embedding-Suche als zuverlässig genug.
#   Falls Ollama nicht erreichbar ist, werden die Embedding-Ergebnisse
#   unverändert verwendet.
@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    # Leere Eingabe abfangen
    if not req.text.strip():
        return RecommendResponse(
            recommended="🤷",
            all_candidates=[],
            llm_used=False,
        )

    # 1. Stufe: Embedding-basierte Suche in ChromaDB
    # ChromaDB embeded den Query-Text automatisch über die eingestellte
    # embedding_function und sucht per Cosine-Distance.
    results = collection.query(
        query_texts=[req.text],
        n_results=3,
    )

    # Ergebnisse aufbereiten: Distanz in Score umrechnen
    if results["metadatas"] is None or results["documents"] is None or results["distances"] is None:
        return RecommendResponse(recommended="🤷", all_candidates=[], llm_used=False)
    candidates: list[EmojiCandidate] = []
    for meta, desc, dist in zip(
        results["metadatas"][0],
        results["documents"][0],
        results["distances"][0],
    ):
        # Umrechnung: Distanz 0 → Score 1.0, Distanz 1 → Score 0.5, ...
        score = 1.0 / (1.0 + dist)
        candidates.append(
            EmojiCandidate(
                emoji=str(meta["emoji"]),
                description=desc,
                score=round(score, 4),
            )
        )

    # Sortieren, damit der beste Kandidat an Position 0 steht
    candidates.sort(key=lambda c: c.score, reverse=True)
    top = candidates[0]
    llm_used = False

    # 2. Stufe: LLM-Verfeinerung (optional, per `use_llm` schaltbar)
    # Nur aktiv wenn der Nutzer es wünscht und der Score unter 0.84 liegt.
    if req.use_llm and top.score < 0.84:
        kandidaten_str = "; ".join(
            f"{c.emoji} ({c.description})" for c in candidates
        )
        prompt = (
            f"Wähle das passendste Emoji für: \"{req.text}\".\n"
            f"Kandidaten: {kandidaten_str}\n"
            f"Antworte NUR mit dem Emoji-Zeichen."
        )
        llm_answer = _query_ollama(prompt)

        if llm_answer:
            found = next(
                (c for c in candidates if c.emoji == llm_answer),
                None,
            )
            if found:
                top = found
            else:
                top = EmojiCandidate(
                    emoji=llm_answer,
                    description="(LLM-Vorschlag)",
                    score=1.0,
                )
            llm_used = True
            log.info("LLM hat %s ausgewählt", top.emoji)
        else:
            log.info("Ollama nicht erreichbar – verwende Embedding-Ergebnis.")

    return RecommendResponse(
        recommended=top.emoji,
        all_candidates=candidates,
        llm_used=llm_used,
    )
