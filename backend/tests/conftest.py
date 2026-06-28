import os
import sys
import tempfile
from unittest.mock import patch
import hashlib
import pytest
import chromadb
from chromadb.api.types import EmbeddingFunction, Embeddings, Documents

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class DummyEmbedding(EmbeddingFunction):
    def __init__(self):
        pass

    @staticmethod
    def name() -> str:
        return "dummy"

    def get_config(self) -> dict:
        return {}

    @classmethod
    def build_from_config(cls, config: dict) -> "DummyEmbedding":
        return cls()

    def __call__(self, input: Documents) -> Embeddings:
        result = []
        for text in input:
            h = hashlib.md5(text.encode()).hexdigest()
            vec = [
                int(h[i : i + 2], 16) / 255.0 for i in range(0, min(384 * 2, len(h)), 2)
            ]
            vec += [0.0] * (384 - len(vec))
            result.append(vec)
        return result


_patcher = patch(
    "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
)
_mock_class = _patcher.start()
_mock_class.return_value = DummyEmbedding()


def make_collections():
    db_path = tempfile.mkdtemp()
    client = chromadb.PersistentClient(path=db_path)
    emb = DummyEmbedding()

    emojis = client.create_collection(
        name="emojis",
        embedding_function=emb,
        metadata={"hnsw:space": "cosine"},
    )
    emojis.add(
        ids=["0", "1", "2", "3"],
        documents=["lacht viel", "weint bitterlich", "kocht gerne", "tanzt im regen"],
        metadatas=[{"emoji": "😂"}, {"emoji": "😭"}, {"emoji": "🍳"}, {"emoji": "💃"}],
    )

    feedback = client.create_collection(
        name="feedback",
        embedding_function=emb,
        metadata={"hnsw:space": "cosine"},
    )
    feedback.add(
        ids=["fb-1", "fb-2"],
        documents=["lacht über witze", "tanzt auf der party"],
        metadatas=[{"emoji": "😂"}, {"emoji": "💃"}],
    )

    return emojis, feedback


@pytest.fixture(scope="function")
def collections():
    return make_collections()


@pytest.fixture(scope="function")
def empty_feedback():
    emb = DummyEmbedding()
    empty_client = chromadb.PersistentClient(path=tempfile.mkdtemp())
    return empty_client.create_collection(
        name="feedback", embedding_function=emb, metadata={"hnsw:space": "cosine"},
    )
