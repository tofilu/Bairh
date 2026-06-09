# Bairh
# text aufnehmen und vektorisieren
import os  # os for pathing
import uuid

import chromadb
import pandas as pd
from pydantic import BaseModel
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import Form
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.collection, app.state.feedback = init_db()
    yield

app = FastAPI(lifespan=lifespan)

# absolute Pfade
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "emoji_dataset.csv")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "emoji_db")
frontend_path = os.path.join(BASE_DIR, "..", "frontend")
frontend_path = os.path.abspath(frontend_path)

DEFAULT_FEEDBACK_WEIGHT = 1.0   # 1.0 = gleich stark wie Basis; zum Testen hochdrehen
FEEDBACK_DISTANCE_THRESHOLD = 0.7  # Feedback, das weiter von der Anfrage entfernt ist als dieser Schwellenwert, wird ignoriert

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

# frontend Ordner mounten
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
def home():
    return FileResponse(os.path.join(frontend_path, "index.html"))

@app.post("/generate")
def generate(request: Request, input: str = Form(...), weight: float = Form(DEFAULT_FEEDBACK_WEIGHT)):
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
        metadatas=[{"emoji": emoji}]
    )
    return {"status": "ok"}

#Für debug Zwecke
@app.get("/debug/feedback")
def debug_feedback(request: Request):
    fb = request.app.state.feedback
    return {"count": fb.count(), "data": fb.get()}

@app.get("/debug/collection")
def debug_collection(request: Request):
    base = request.app.state.collection
    return {"count": base.count(), "data": base.get(limit=20)}   # limit, weil die Basis riesig ist

class ListEntry(BaseModel):
    emoji: str
    description: str
    score: float


# embedding
def init_db():
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(
        name="emojis",
        embedding_function=embedding_function,  # type: ignore[arg-type]
    )
    csv = pd.read_csv(  # csv ist ein typ namens DATAFRAME, ist eigentlich immer noch eine tabelle
        CSV_PATH
    )
    ids = [
        str(index) for index in range(len(csv))
    ]  # wir brauchen die anzahl der zeilen als id, in chromadb werden nur strings angenommen deshalb durchiterieren und alles dann zu strings machen

    feedback = chroma_client.get_or_create_collection(
        name="feedback",
        embedding_function=embedding_function,  # type: ignore[arg-type]
    )

    descriptions = csv["description"].to_list()

    emojis = [{"emoji": emoji} for emoji in csv["emoji"].to_list()]

    batch_size = 100
    length = len(csv)
    for i in range(0, length, batch_size):
        end = min(i + batch_size, length)
        collection.add(
            ids=ids[i:end],
            documents=descriptions[i:end],
            metadatas=emojis[i:end],  # type: ignore[arg-type]
        )
    return collection, feedback

"""
def recommend(collection, text):
    results = collection.query(
        query_texts=[text],
        n_results=4,
    )
    candidates: list[ListEntry] = []
    for metadata, description, cosine_distance in zip(
        results["metadatas"][0],
        results["documents"][0],
        results["distances"][0],
    ):
        score = 1.0 / (1.0 + cosine_distance)
        candidates.append(
            ListEntry(
                emoji=str(metadata["emoji"]), description=description, score=score
            )
        )
    candidates.sort(key=lambda c: c.score, reverse=True)
    #top = candidates[0]
    return candidates
"""

def recommend(collection, feedback, text, weight, n_results=4):
    scores: dict[str, float] = {}
    descriptions: dict[str, str] = {}

    def add_emojis(results, factor, is_feedback=False):
        if not results["ids"][0]:          # leere Collection -> nichts tun
            return
        for metadata, description, cosine_distance in zip(
            results["metadatas"][0],
            results["documents"][0],
            results["distances"][0]
        ):
            if is_feedback and cosine_distance > FEEDBACK_DISTANCE_THRESHOLD:   # Feedback, das zu weit von der Anfrage entfernt ist, ignorieren
                continue
            emoji = str(metadata["emoji"])
            score = 1.0 / (1.0 + cosine_distance)
            scores[emoji] = scores.get(emoji, 0.0) + factor * score
            descriptions.setdefault(emoji, description)

    add_emojis(collection.query(query_texts=[text], n_results=n_results), 1.0)

    if feedback.count() > 0:
        add_emojis(feedback.query(query_texts=[text], n_results=n_results), weight, is_feedback=True)

    candidates = [
        ListEntry(emoji=e, description=descriptions[e], score=sc)
        for e, sc in scores.items()
    ]
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:n_results]