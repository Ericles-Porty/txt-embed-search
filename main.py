import os
import glob
import chromadb
import hashlib
import json
from sentence_transformers import SentenceTransformer

# === CONFIGURAÇÃO ===
PASTA_ARQUIVOS = "knowledge_database"
PASTA_CHROMA = "chroma_db"
ARQUIVO_HASHES = "hashes.json"
TAMANHO_CHUNK = 2000
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

modelo_embedding = SentenceTransformer(EMBEDDING_MODEL)

# === FUNÇÕES ===
def gerar_embedding(texto):
    """Gera embedding para um texto."""
    return modelo_embedding.encode([texto])[0].tolist()

def hash_arquivo(caminho):
    """Gera hash SHA256 do conteúdo para detectar alterações."""
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def carregar_hashes():
    if os.path.exists(ARQUIVO_HASHES):
        with open(ARQUIVO_HASHES, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def salvar_hashes(hashes):
    with open(ARQUIVO_HASHES, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2)

def dividir_em_chunks(texto, tamanho=TAMANHO_CHUNK):
    """Divide o texto em pedaços menores para não estourar o limite do modelo."""
    return [texto[i:i+tamanho] for i in range(0, len(texto), tamanho)]

# === CARREGAR CHROMADB LOCAL ===
chroma_client = chromadb.PersistentClient(path=PASTA_CHROMA)
try:
    collection = chroma_client.get_collection("documentos_sindicato")
except:
    collection = chroma_client.create_collection(name="documentos_sindicato")

# === INDEXAÇÃO COM CHUNKING ===
print("📂 Verificando arquivos...")
hashes_existentes = carregar_hashes()
hashes_atualizados = hashes_existentes.copy()
novos_ou_alterados = []

for caminho in glob.glob(os.path.join(PASTA_ARQUIVOS, "*.txt")):
    nome_arquivo = os.path.basename(caminho)
    hash_atual = hash_arquivo(caminho)

    if nome_arquivo not in hashes_existentes or hashes_existentes[nome_arquivo] != hash_atual:
        novos_ou_alterados.append((caminho, nome_arquivo, hash_atual))

if novos_ou_alterados:
    print(f"📄 {len(novos_ou_alterados)} arquivos novos/alterados detectados.")
    for caminho, nome, hash_atual in novos_ou_alterados:
        with open(caminho, "r", encoding="utf-8", errors="ignore") as f:
            conteudo = f.read()

        chunks = dividir_em_chunks(conteudo, TAMANHO_CHUNK)
        for idx_chunk, chunk in enumerate(chunks, start=1):
            embedding = gerar_embedding(chunk)
            collection.add(
                ids=[f"{nome}_chunk_{idx_chunk}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"arquivo": nome, "chunk": idx_chunk}]
            )

        hashes_atualizados[nome] = hash_atual
        print(f"✅ Indexado: {nome} ({len(chunks)} chunks)")

    salvar_hashes(hashes_atualizados)
else:
    print("✅ Nenhum arquivo novo ou alterado.")

# === FUNÇÃO DE BUSCA ===
def buscar(query, k=5):
    query_embedding = gerar_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    return results

# === LOOP DE CONSULTA ===
while True:
    consulta = input("\n🔍 Digite sua busca (ou 'sair'): ")
    if consulta.lower() == "sair":
        break

    resultado = buscar(consulta, k=5)
    print("\n📄 Resultados:")
    for doc, meta in zip(resultado['documents'][0], resultado['metadatas'][0]):
        print(f"Arquivo: {meta['arquivo']} (chunk {meta['chunk']})")
        print(f"Trecho: {doc}...\n")
