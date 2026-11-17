"""Configuração compartilhada entre a indexação e a busca.
O modelo de embedding PRECISA ser o mesmo nos dois lados."""
import os
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).parent
# Pasta com os .txt/.pdf. Padrão: ./livros (fora do git). Para usar outra: RAG_LIVROS_DIR=/caminho
LIVROS_DIR = Path(os.environ.get("RAG_LIVROS_DIR") or BASE_DIR / "livros")
CHROMA_DIR = BASE_DIR / "chroma_db"       # onde o Chroma persiste o índice
COLECAO = "livros"

# Modelo multilíngue (funciona bem com português). Roda local, ~470 MB.
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_SIZE = 1000      # caracteres por trecho
CHUNK_OVERLAP = 200    # sobreposição entre trechos para não cortar ideias


@lru_cache(maxsize=1)
def get_embedding_function():
    """Carrega o modelo uma única vez por processo (antes recarregava a cada busca)."""
    from chromadb.utils import embedding_functions
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_collection():
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLECAO,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )
