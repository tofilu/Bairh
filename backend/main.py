# Bairh
# text aufnehmen und vektorisieren
import os  # os for pathing
import re  # regex

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
    app.state.collection = init_db()
    yield

app = FastAPI(lifespan=lifespan)

# absolute Pfade
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "emoji_dataset.csv")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "emoji_db")
frontend_path = os.path.join(BASE_DIR, "..", "frontend")
frontend_path = os.path.abspath(frontend_path)


embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

# frontend Ordner mounten
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
def home():
    return FileResponse(os.path.join(frontend_path, "index.html"))

@app.post("/generate")
def generate(request: Request, input: str = Form(...)):
    return recommend(request.app.state.collection, input)


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

    descriptions = csv["description"].apply(_clean_descriptions).to_list()

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
    return collection

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


def _clean_descriptions(input: str) -> str:
    return re.sub(r"^E[\d.]+\s*", "", input).strip()
