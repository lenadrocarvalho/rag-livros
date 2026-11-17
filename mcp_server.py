"""Servidor MCP que expõe o RAG dos livros para Claude Desktop / Claude Code
(ou qualquer cliente compatível com MCP)."""
from mcp.server.mcpserver import MCPServer  # mcp 2.x (antes: FastMCP)

from busca import buscar, formatar, listar_livros

mcp = MCPServer(
    "biblioteca",
    instructions=(
        "Use estas ferramentas para responder perguntas sobre os livros indexados. "
        "Responda SOMENTE com base nos trechos retornados e cite a fonte (livro e página/trecho). "
        "Se a busca retornar 'NENHUM TRECHO RELEVANTE', diga que a informação não está nos livros "
        "em vez de responder de memória. Faça mais de uma busca com termos diferentes se precisar."
    ),
)


@mcp.tool()
def buscar_nos_livros(pergunta: str, n_resultados: int = 5, livro: str | None = None) -> str:
    """Busca semântica nos livros da série. Retorna trechos originais com a fonte.

    Args:
        pergunta: o que procurar (frase natural funciona melhor que palavra solta).
        n_resultados: quantos trechos trazer (1 a 15).
        livro: opcional, título exato de um livro para restringir a busca (veja listar_livros).
    """
    n = max(1, min(n_resultados, 15))
    return formatar(buscar(pergunta, n=n, livro=livro))


@mcp.tool()
def listar_livros_indexados() -> str:
    """Lista os livros disponíveis no índice, na ordem da série."""
    livros = listar_livros()
    return "\n".join(livros) if livros else "Nenhum livro indexado ainda."


if __name__ == "__main__":
    mcp.run()  # transporte stdio
