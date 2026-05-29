# Bairh
# text aufnehmen und vektorisieren
import os  # os for pathing
import re  # regex

import chromadb
import pandas as pd
from pydantic import BaseModel
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "emoji_dataset.csv")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "emoji_db")

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)


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


def read_input():
    print("please give an input")
    user_input = input()
    return user_input


def recommend(collection, text):
    results = collection.query(
        query_texts=[text],
        n_results=3,
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
    top = candidates[0]
    return top


def _clean_descriptions(input: str) -> str:
    return re.sub(r"^E[\d.]+\s*", "", input).strip()


print(recommend(init_db(), read_input()))
