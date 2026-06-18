# Bairh
# Ziele:
# 1. Fehler werfen wenn etwas nicht stimmt
# 2. csv neu generieren wenn mehr dateien existieren bzw wenn die csv mehr inhalte enthält als in der aktuellen fassung
# 3. mehrere csv's in eine chromadb zusammenfassen
import os  # os for pathing
import pathlib  # pathlib for pathing
import uuid  # uuid for unique identifiers
import chromadb
import pandas as pd  # for reading the csv
import logging

from chromadb.errors import (
    InternalError,
)  # to catch the specific error that gets thrown
from pydantic import BaseModel
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.collection, app.state.feedback = init_db()
    yield


app = FastAPI(lifespan=lifespan)

# absolute Pfade
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "emoji_dataset.csv")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "emoji_db")
FRONTEND_PATH = os.path.join(BASE_DIR, "..", "frontend")
FRONTEND_PATH = os.path.abspath(FRONTEND_PATH)

DEFAULT_FEEDBACK_WEIGHT = 0.7  # 1.0 = gleich stark wie Basis; zum Testen hochdrehen
FEEDBACK_DISTANCE_THRESHOLD = 0.7  # Feedback, das weiter von der Anfrage entfernt ist als dieser Schwellenwert, wird ignoriert

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

# frontend Ordner mounten
app.mount("/static", StaticFiles(directory=FRONTEND_PATH), name="static")


@app.get("/")
def home():
    return FileResponse(os.path.join(FRONTEND_PATH, "index.html"))


@app.post("/generate")
def generate(
    request: Request,
    input: str = Form(...),
    weight: float = Form(DEFAULT_FEEDBACK_WEIGHT),
):
    return recommend(
        request.app.state.collection,
        request.app.state.feedback,
        input,
        weight,
    )


@app.post("/feedback")
def feedback(request: Request, emoji: str = Form(...), emoji_feedback: str = Form(...)):
    print(f"Feedback received for emoji {emoji}: {emoji_feedback}")
    request.app.state.feedback.add(
        ids=[f"fb-{uuid.uuid4()}"],
        documents=[emoji_feedback],
        metadatas=[{"emoji": emoji}],
    )
    return {"status": "ok"}


# Für debug Zwecke
@app.get("/debug/feedback")
def debug_feedback(request: Request):
    fb = request.app.state.feedback
    return {"count": fb.count(), "data": fb.get()}

class ListEntry(BaseModel):
    emoji: str
    description: str
    score: float


# chroma
def get_chroma_client():
    logging.info("looking for PersistentClient of ChromaDB")
    try:
        return chromadb.PersistentClient(path=CHROMA_DB_PATH)
    except InternalError as e:
        raise RuntimeError(
            "ChromaDB unter " + CHROMA_DB_PATH + " nicht erreichbar: " + str(e)
        )


def load_csv(
    path,
):  # csv ist ein typ namens DATAFRAME, ist eigentlich immer noch eine tabelle
    logging.info("checking for a csv")
    if not pathlib.Path(path).exists():
        raise FileNotFoundError("emoji_dataset.csv nicht gefunden unter " + path)
    csv = pd.read_csv(path)
    return csv


def fill_collection(collection, csv):
    # wir brauchen die anzahl der zeilen als id, in chromadb werden nur strings angenommen deshalb durchiterieren und alles dann zu strings machen
    logging.info("generating ids")
    ids = [str(index) for index in range(len(csv))]

    descriptions = csv["description"].to_list()
    emojis = [{"emoji": emoji} for emoji in csv["emoji"].to_list()]

    logging.info("filling the chromadb bit by bit with new info")
    batch_size = 100
    length = len(csv)
    for i in range(0, length, batch_size):
        end = min(i + batch_size, length)
        collection.add(
            ids=ids[i:end],
            documents=descriptions[i:end],
            metadatas=emojis[i:end],  # type: ignore[arg-type]
        )


def init_db():
    logging.info("Initializing DB")
    chroma_client = get_chroma_client()

    collection = chroma_client.get_or_create_collection(
        name="emojis",
        embedding_function=embedding_function,  # type: ignore[arg-type]
    )

    logging.info("doing some feedback stuff")
    feedback = chroma_client.get_or_create_collection(
        name="feedback",
        embedding_function=embedding_function,  # type: ignore[arg-type]
    )

    csv = load_csv(CSV_PATH)
    fill_collection(collection, csv)

    return collection, feedback


def recommend(collection, feedback, text, weight, n_results=4):
    scores: dict[str, float] = {}
    descriptions: dict[str, str] = {}

    def add_emojis(results, factor, is_feedback=False):
        if len(results["ids"][0]) == 0:  # leere Collection -> nichts tun
            return
        for metadata, description, cosine_distance in zip(
            results["metadatas"][0], results["documents"][0], results["distances"][0]
        ):
            if (
                is_feedback and cosine_distance > FEEDBACK_DISTANCE_THRESHOLD
            ):  # Feedback, das zu weit von der Anfrage entfernt ist, ignorieren
                continue
            emoji = str(metadata["emoji"])
            score = 1.0 / (1.0 + cosine_distance)
            scores[emoji] = scores.get(emoji, 0.0) + factor * score
            descriptions.setdefault(emoji, description)

    add_emojis(collection.query(query_texts=[text], n_results=n_results), 1.0)

    if feedback.count() > 0:
        add_emojis(
            feedback.query(query_texts=[text], n_results=n_results),
            weight,
            is_feedback=True,
        )

    candidates = [
        ListEntry(emoji=e, description=descriptions[e], score=sc)
        for e, sc in scores.items()
    ]
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:n_results]
