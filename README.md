# Busca Semântica com ChromaDB e Hugging Face

Este projeto realiza indexação e busca semântica em arquivos `.txt` usando **Hugging Face Sentence Transformers** para gerar embeddings e o **ChromaDB** para armazenar e pesquisar.

## 📌 Funcionalidades
- Indexa automaticamente arquivos `.txt` da pasta configurada.
- Detecta alterações em arquivos usando hash SHA256.
- Divide textos grandes em *chunks* para melhorar a busca.
- Persistência local do banco ChromaDB.

## 🚀 Tecnologias
- [Sentence Transformers](https://www.sbert.net/)
- [ChromaDB](https://www.trychroma.com/)
- [Python](https://www.python.org/)

## ⚙️ Instalação
```bash
# Clone este repositório
git clone https://github.com/ericles-porty/txt-embed-search.git
cd txt-embed-search

# Crie o ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instale as dependências
pip install -r requirements.txt
```
# Uso
1. Coloque seus arquivos `.txt` na pasta `knowledge_database`.
2. Execute o script principal:
```bash
python main.py
```
3. Siga as instruções no terminal para realizar buscas.
