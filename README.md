# txt-embed-search

Ferramenta para indexação e busca semântica de arquivos `.txt` usando **OpenAI Embeddings** e **ChromaDB**.

## 🚀 Funcionalidades
- Lê todos os arquivos `.txt` de uma pasta.
- Divide textos em **chunks** configuráveis para melhorar precisão de busca.
- Gera embeddings usando o modelo `text-embedding-3-small` da OpenAI.
- Salva os embeddings localmente no **ChromaDB** (persistente em disco).
- Usa hash SHA-256 para processar apenas arquivos novos ou modificados.
- Realiza busca textual via similaridade vetorial.

## 🛠 Tecnologias
- [Python 3.10+](https://www.python.org/)
- [OpenAI API](https://platform.openai.com/docs/)
- [ChromaDB](https://docs.trychroma.com/)
