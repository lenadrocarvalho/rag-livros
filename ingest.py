"""Lê os livros da pasta ./livros, quebra em trechos e indexa no Chroma.

Convenção de nome de arquivo (opcional, mas ajuda nas citações):
    01 - Um Estudo em Vermelho.txt
    02 - O Signo dos Quatro.pdf
O número vira a ordem na série e o resto vira o título.

Uso:
    python ingest.py            # indexa (pula livros já indexados)
    python ingest.py --reset    # apaga o índice e refaz tudo
"""
import re
import shutil
from collections import Counter
import sys
from pathlib import Path

from config import CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, LIVROS_DIR, get_collection


def ler_arquivo(caminho: Path) -> list[tuple[str, int | None]]:
    """Retorna lista de (texto, pagina). Para .txt a página é None.
    PDFs usam PyMuPDF: o pypdf quebrava palavras no meio ("ro sados", "er a")."""
    if caminho.suffix.lower() == ".pdf":
        import fitz  # PyMuPDF
        with fitz.open(str(caminho)) as doc:
            paginas = [(p.get_text(), i + 1) for i, p in enumerate(doc)]
        return remover_repetidas(paginas)
    texto = caminho.read_text(encoding="utf-8", errors="ignore")
    return [(texto, None)]


def remover_repetidas(paginas: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Remove cabeçalhos/rodapés: linhas que aparecem em mais da metade das páginas
    (ex.: 'www.nead.unama.br')."""
    if len(paginas) < 4:
        return paginas
    contagem = Counter(l.strip() for t, _ in paginas for l in set(t.splitlines()) if l.strip())
    repetidas = {l for l, n in contagem.items() if n > len(paginas) / 2}
    return [("\n".join(l for l in t.splitlines() if l.strip() not in repetidas), p)
            for t, p in paginas]


# Pronomes que vêm depois de hífen ("pendurou-se-me"): nesses casos o hífen é da palavra.
_CLITICOS = r"(?:se|me|te|lhe|lhes|o|a|os|as|lo|la|los|las|no|na|nos|nas|vos|ei|ás|á|emos|eis|ão)\b"


def limpar(texto: str) -> str:
    texto = texto.replace("\r", "").replace("­", "")
    texto = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", texto)             # linha só com nº de página
    texto = re.sub(rf"(\w)-\n\s*({_CLITICOS})", r"\1-\2", texto)  # "pendurou-\nse" -> "pendurou-se"
    texto = re.sub(r"(\w)-\n\s*(\w)", r"\1\2", texto)               # "varan-\nda" -> "varanda"
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r" *\n *", "\n", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def quebrar(texto: str, tamanho=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[str]:
    """Quebra em trechos tentando respeitar fim de parágrafo/frase."""
    trechos, inicio = [], 0
    while inicio < len(texto):
        fim = min(inicio + tamanho, len(texto))
        if fim < len(texto):
            corte = max(texto.rfind("\n\n", inicio, fim), texto.rfind(". ", inicio, fim))
            if corte > inicio + tamanho // 2:
                fim = corte + 1
        trecho = texto[inicio:fim].strip()
        if len(trecho) > 50:
            trechos.append(trecho)
        if fim >= len(texto):
            break
        inicio = max(fim - overlap, inicio + 1)
    return trechos


def metadados_do_nome(caminho: Path) -> dict:
    m = re.match(r"^\s*(\d+)\s*[-_.]\s*(.+)$", caminho.stem)
    if m:
        return {"ordem": int(m.group(1)), "livro": m.group(2).strip()}
    return {"ordem": 0, "livro": caminho.stem}


def main():
    if "--reset" in sys.argv and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        print("Índice apagado.")

    arquivos = sorted(p for p in LIVROS_DIR.glob("*") if p.suffix.lower() in {".txt", ".pdf"})
    if not arquivos:
        print(f"Nenhum .txt/.pdf em {LIVROS_DIR}. Coloque os livros lá e rode de novo.")
        return

    col = get_collection()
    for arq in arquivos:
        meta = metadados_do_nome(arq)
        if col.get(where={"arquivo": arq.name}, limit=1)["ids"]:
            print(f"= {meta['livro']} já indexado, pulando.")
            continue

        ids, docs, metas = [], [], []
        n = 0
        for texto, pagina in ler_arquivo(arq):
            for trecho in quebrar(limpar(texto)):
                ids.append(f"{arq.stem}-{n}")
                docs.append(trecho)
                metas.append({**meta, "arquivo": arq.name, "trecho": n, "pagina": pagina or 0})
                n += 1

        # Envia em lotes para não estourar memória
        for i in range(0, len(ids), 200):
            col.add(ids=ids[i:i + 200], documents=docs[i:i + 200], metadatas=metas[i:i + 200])
        print(f"+ {meta['livro']}: {n} trechos indexados.")

    print(f"\nTotal no índice: {col.count()} trechos.")


if __name__ == "__main__":
    main()
