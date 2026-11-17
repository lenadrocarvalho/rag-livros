"""Busca híbrida usada pelo servidor MCP e pelo modo local.

Combina dois rankings com Reciprocal Rank Fusion (RRF):
  - semântico (embeddings no Chroma): entende paráfrases ("como ele morreu?");
  - palavra-chave (BM25): acha frases exatas e nomes ("olhos de ressaca").

Anti-alucinação: um trecho só é devolvido se for parecido o bastante com a pergunta
(distância de cosseno) OU se contiver quase todas as palavras importantes dela.
Senão, a busca diz explicitamente que não encontrou nada.

Teste no terminal:
    python busca.py "quem é o narrador do primeiro livro?"
"""
import math
import re
import sys
import unicodedata
from collections import Counter

from config import get_collection, get_embedding_function

# Distância de cosseno: 0 = idêntico, 1 = sem relação.
# Calibrado nos 5 livros: perguntas do tema ficaram <= 0.38; fora do tema >= 0.54.
LIMIAR_DISTANCIA = 0.45
# Trecho que contém >= 75% dos termos da pergunta é aceito mesmo com distância maior,
# desde que não seja totalmente sem relação.
COBERTURA_MIN = 0.75
LIMIAR_DISTANCIA_PALAVRAS = 0.60
CANDIDATOS = 30   # quantos trechos cada ranking traz antes da fusão
RRF_K = 60

STOPWORDS = set("""
a à ao aos as às até com como da das de dela dele deles do dos e é ela ele eles em entre era
essa esse esta este eu foi for há isso isto já la lhe lo mais mas me mesmo meu minha muito na
nas não nem no nos nós num numa o os ou para pela pelas pelo pelos por qual quando que quem se
sem ser seu sua são só também te tem tinha um uma umas uns vai você estava foram sobre onde
contra porque assim depois antes ainda então
""".split())


def _sem_acento(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


_STOP = {_sem_acento(w) for w in STOPWORDS}


RADICAL = 6  # "hipocondria"/"hipocondríaco" -> "hipoco"; "dançava"/"dançar" -> "dancav"/"dancar"


def _normalizar(texto: str) -> list[str]:
    """Tokens sem acento, sem stopwords e cortados no radical (stemming por prefixo)."""
    return [t[:RADICAL] for t in re.findall(r"\w+", _sem_acento(texto))
            if len(t) > 1 and t not in _STOP]


class _BM25:
    """BM25 simples em memória (2 mil trechos cabem fácil)."""

    def __init__(self, ids, docs, metas, k1=1.5, b=0.75):
        self.ids, self.metas, self.k1, self.b = ids, metas, k1, b
        self.tfs = [Counter(_normalizar(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.media = sum(self.lens) / max(len(self.lens), 1)
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def buscar(self, termos, livro=None, n=CANDIDATOS):
        pontos = []
        for i, tf in enumerate(self.tfs):
            if livro and self.metas[i]["livro"] != livro:
                continue
            s = 0.0
            for t in termos:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.media))
            if s > 0:
                pontos.append((s, i))
        pontos.sort(reverse=True)
        return [self.ids[i] for _, i in pontos[:n]]


_indice = {"total": -1, "bm25": None}


def _bm25() -> _BM25:
    """Monta o índice de palavras a partir do Chroma; refaz se o nº de trechos mudar."""
    col = get_collection()
    total = col.count()
    if _indice["total"] != total:
        r = col.get(include=["documents", "metadatas"])
        _indice.update(total=total, bm25=_BM25(r["ids"], r["documents"], r["metadatas"]))
    return _indice["bm25"]


def _cosseno(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 1 - dot / (na * nb) if na and nb else 1.0


def buscar(pergunta: str, n: int = 5, livro: str | None = None) -> list[dict]:
    col = get_collection()
    where = {"livro": livro} if livro else None
    q_emb = [float(x) for x in get_embedding_function()([pergunta])[0]]
    termos = list(dict.fromkeys(_normalizar(pergunta)))

    # 1) ranking semântico
    sem = col.query(query_embeddings=[q_emb], n_results=CANDIDATOS, where=where,
                    include=["documents", "metadatas", "distances"])
    cand = {i: {"texto": d, "meta": m, "dist": dist}
            for i, d, m, dist in zip(sem["ids"][0], sem["documents"][0],
                                     sem["metadatas"][0], sem["distances"][0])}
    rank_sem = {i: r for r, i in enumerate(sem["ids"][0])}

    # 2) ranking por palavra-chave
    ids_kw = _bm25().buscar(termos, livro=livro) if termos else []
    rank_kw = {i: r for r, i in enumerate(ids_kw)}
    faltando = [i for i in ids_kw if i not in cand]
    if faltando:
        g = col.get(ids=faltando, include=["documents", "metadatas", "embeddings"])
        for i, d, m, e in zip(g["ids"], g["documents"], g["metadatas"], g["embeddings"]):
            cand[i] = {"texto": d, "meta": m, "dist": _cosseno(q_emb, e)}

    # 3) fusão RRF + filtro anti-alucinação
    resultados = []
    for i, c in cand.items():
        presentes = set(_normalizar(c["texto"]))
        cobertura = sum(t in presentes for t in termos) / len(termos) if termos else 0.0
        aceito = (
            c["dist"] <= LIMIAR_DISTANCIA                                          # parecido no sentido
            or (cobertura >= COBERTURA_MIN and c["dist"] <= LIMIAR_DISTANCIA_PALAVRAS)
            or (cobertura == 1 and len(termos) >= 2)                               # contém todos os termos
        )
        if not aceito:
            continue
        score = 0.0
        if i in rank_sem:
            score += 1 / (RRF_K + rank_sem[i])
        if i in rank_kw:
            score += 1 / (RRF_K + rank_kw[i])
        resultados.append({"texto": c["texto"], "distancia": round(c["dist"], 3),
                           "cobertura": round(cobertura, 2), "_score": score, **c["meta"]})

    resultados.sort(key=lambda r: r["_score"], reverse=True)
    for r in resultados:
        del r["_score"]
    return resultados[:n]


def formatar(resultados: list[dict]) -> str:
    if not resultados:
        return "NENHUM TRECHO RELEVANTE ENCONTRADO nos livros indexados."
    partes = []
    for i, r in enumerate(resultados, 1):
        local = f"p. {r['pagina']}" if r.get("pagina") else f"trecho {r['trecho']}"
        partes.append(
            f"[Fonte {i}] Livro: {r['livro']} (#{r['ordem']}) | {local} | "
            f"similaridade: {1 - r['distancia']:.2f} | termos da pergunta presentes: {r['cobertura']:.0%}\n"
            f"{r['texto']}"
        )
    return "\n\n---\n\n".join(partes)


def listar_livros() -> list[str]:
    metas = get_collection().get(include=["metadatas"])["metadatas"]
    livros = {(m["ordem"], m["livro"]) for m in metas}
    return [f"{o} - {l}" for o, l in sorted(livros)]


if __name__ == "__main__":
    pergunta = " ".join(sys.argv[1:]) or "Qual é o tema principal?"
    print(formatar(buscar(pergunta)))
