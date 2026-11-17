"""Modo 100% offline: busca no Chroma e gera a resposta com um modelo do Ollama.

Pré-requisito: Ollama rodando (https://ollama.com) e um modelo baixado, ex.:
    ollama pull llama3.1:8b

Uso:
    python perguntar_local.py "como o detetive conhece o parceiro?"
    python perguntar_local.py              # modo chat interativo
"""
import json
import sys
import urllib.request

from busca import buscar, formatar

MODELO = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/chat"

SISTEMA = """Você é um assistente que responde perguntas sobre uma série de livros.
Regras:
1. Use APENAS as informações dos trechos fornecidos em CONTEXTO.
2. Cite as fontes no formato [Fonte N] ao lado de cada afirmação.
3. Se o contexto não tiver a resposta, diga exatamente: "Não encontrei isso nos livros."
4. Não invente nomes, datas ou eventos que não estejam nos trechos."""


def perguntar(pergunta: str) -> str:
    contexto = formatar(buscar(pergunta, n=6))
    if contexto.startswith("NENHUM"):
        return "Não encontrei isso nos livros."
    corpo = {
        "model": MODELO,
        "stream": False,
        "options": {"temperature": 0.1},
        "messages": [
            {"role": "system", "content": SISTEMA},
            {"role": "user", "content": f"CONTEXTO:\n{contexto}\n\nPERGUNTA: {pergunta}"},
        ],
    }
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(corpo).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["message"]["content"]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(perguntar(" ".join(sys.argv[1:])))
    else:
        print("Pergunte sobre os livros (Ctrl+C para sair)\n")
        while True:
            q = input("> ").strip()
            if q:
                print("\n" + perguntar(q) + "\n")
