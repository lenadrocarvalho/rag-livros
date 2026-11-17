"""Exporta o índice do Chroma para o site estático em ./docs (GitHub Pages).

Gera:
    docs/data/acervo.json   livros + trechos (texto, livro, página)
    docs/data/vetores.bin   embeddings normalizados e quantizados em int8 (N x DIMS)

O site faz a busca híbrida (BM25 + semântica) inteiramente no navegador, com o
mesmo modelo de embedding (versão ONNX via transformers.js). Não precisa de servidor.

Uso:
    python ingest.py          # se ainda não indexou
    python exportar_web.py
"""
import json
import re

import numpy as np

from config import BASE_DIR, EMBEDDING_MODEL, get_collection

SAIDA = BASE_DIR / "docs" / "data"

# Metadados para exibição. Livros sem entrada aqui usam o nome do arquivo.
INFO = {
    "iracema": ("Iracema", "José de Alencar", 1865),
    "o_primo_basilio": ("O Primo Basílio", "Eça de Queirós", 1878),
    "memorias_postumas_de_bras_cubas": ("Memórias Póstumas de Brás Cubas", "Machado de Assis", 1881),
    "o_cortiço": ("O Cortiço", "Aluísio Azevedo", 1890),
    "Dom Casmurro - Machado de Assis": ("Dom Casmurro", "Machado de Assis", 1899),
}


def titulo_do_arquivo(nome: str) -> str:
    return re.sub(r"[_]+", " ", nome).strip().capitalize()


def main():
    col = get_collection()
    r = col.get(include=["documents", "metadatas", "embeddings"])
    if not r["ids"]:
        raise SystemExit("Índice vazio. Rode `python ingest.py` antes.")

    vetores = np.asarray(r["embeddings"], dtype=np.float32)
    vetores /= np.linalg.norm(vetores, axis=1, keepdims=True) + 1e-12
    dims = vetores.shape[1]

    # Ordena por livro (cronológico) e posição no livro: vizinhos = contexto de leitura.
    chaves = sorted({m["livro"] for m in r["metadatas"]},
                    key=lambda k: (INFO.get(k, ("", "", 9999))[2], k))
    idx_livro = {k: i for i, k in enumerate(chaves)}
    ordem = sorted(range(len(r["ids"])),
                   key=lambda i: (idx_livro[r["metadatas"][i]["livro"]], r["metadatas"][i]["trecho"]))

    livros = []
    for k in chaves:
        titulo, autor, ano = INFO.get(k, (titulo_do_arquivo(k), "", None))
        paginas = max(m["pagina"] for m in r["metadatas"] if m["livro"] == k)
        n = sum(1 for m in r["metadatas"] if m["livro"] == k)
        livros.append({"id": k, "titulo": titulo, "autor": autor, "ano": ano,
                       "paginas": paginas, "trechos": n})

    trechos = [{"l": idx_livro[r["metadatas"][i]["livro"]],
                "p": r["metadatas"][i]["pagina"],
                "t": r["documents"][i]} for i in ordem]

    SAIDA.mkdir(parents=True, exist_ok=True)
    (SAIDA / "acervo.json").write_text(
        json.dumps({"modelo": EMBEDDING_MODEL, "dims": dims, "livros": livros, "trechos": trechos},
                   ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8")
    q = np.clip(np.round(vetores[ordem] * 127), -127, 127).astype(np.int8)
    (SAIDA / "vetores.bin").write_bytes(q.tobytes())

    print(f"{len(trechos)} trechos de {len(livros)} livros -> {SAIDA}")
    for f in SAIDA.iterdir():
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
