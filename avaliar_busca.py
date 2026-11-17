"""Avaliação rápida da busca: o livro certo aparece no topo? Perguntas fora do tema são recusadas?

Uso:
    python avaliar_busca.py
"""
from busca import buscar

# (pergunta, trecho do nome do livro esperado)
DENTRO = [
    ("Capitu tinha olhos de ressaca", "Dom Casmurro"),
    ("olhos de cigana oblíqua e dissimulada", "Dom Casmurro"),
    ("Bentinho ia ser padre no seminário", "Dom Casmurro"),
    ("Escobar morreu afogado no mar", "Dom Casmurro"),
    ("Iracema a virgem dos lábios de mel", "iracema"),
    ("Martim era um guerreiro branco", "iracema"),
    ("Brás Cubas é um defunto autor", "memorias_postumas"),
    ("o emplasto contra a hipocondria", "memorias_postumas"),
    ("Quincas Borba e o humanitismo", "memorias_postumas"),
    ("João Romão era dono do cortiço", "o_cortiço"),
    ("Bertoleza trabalhava para João Romão", "o_cortiço"),
    ("Rita Baiana dançava", "o_cortiço"),
    ("Luísa e o primo Basílio", "o_primo_basilio"),
    ("Juliana a criada de Luísa", "o_primo_basilio"),
]
FORA = [
    "Qual é a velocidade de uma nave espacial em Marte?",
    "receita de bolo de chocolate",
    "como configurar um roteador wifi",
    "quem ganhou a copa do mundo de 2022",
    "programação em Python com listas",
    "preço das ações da Petrobras",
]
# Obs.: perguntas "pegadinha" sobre personagens reais dos livros (ex.: "nome do cachorro de
# Capitu") devem trazer trechos de Capitu — quem recusa é o modelo, ao ver que o trecho não responde.


def main():
    top1 = top3 = 0
    for q, esperado in DENTRO:
        r = buscar(q, n=3)
        livros = [x["livro"] for x in r]
        ok1 = bool(livros) and esperado.lower() in livros[0].lower()
        ok3 = any(esperado.lower() in l.lower() for l in livros)
        top1 += ok1
        top3 += ok3
        print(f"{'OK ' if ok1 else ('~3 ' if ok3 else 'ERR')} {q:45} -> {[l[:14] for l in livros]}")
    recusas = 0
    print()
    for q in FORA:
        r = buscar(q, n=3)
        recusas += not r
        print(f"{'OK ' if not r else 'ERR'} {q:45} -> {[(x['livro'][:14], x['distancia']) for x in r]}")
    print(f"\nTop-1: {top1}/{len(DENTRO)} | Top-3: {top3}/{len(DENTRO)} | "
          f"Recusas corretas: {recusas}/{len(FORA)}")


if __name__ == "__main__":
    main()
