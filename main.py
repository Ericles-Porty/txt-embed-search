import os
import glob
import hashlib
import json
import chromadb
from sentence_transformers import SentenceTransformer
from ollama import chat

# === CONFIGURAÇÃO ===
PASTA_ARQUIVOS = "knowledge_database"
PASTA_CHROMA = "chroma_db"
ARQUIVO_HASHES = "hashes.json"
TAMANHO_CHUNK = 800
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARIDADE_MINIMA = 0.75  # Ajuste conforme necessário

# === EMBEDDINGS ===
print("🔄 Carregando modelo de embeddings...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

def gerar_embedding(texto):
    return embedding_model.encode(texto).tolist()

# === HASH ===
def hash_arquivo(caminho):
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

# === CHUNKING ===
def dividir_em_chunks(texto, tamanho=TAMANHO_CHUNK):
    return [texto[i:i+tamanho] for i in range(0, len(texto), tamanho)]

# === CHROMADB ===
chroma_client = chromadb.PersistentClient(path=PASTA_CHROMA)
try:
    collection = chroma_client.get_collection("documentos_sindicato")
except:
    collection = chroma_client.create_collection(name="documentos_sindicato")

# === INDEXAÇÃO ===
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

# === BUSCA POR SIMILARIDADE MÍNIMA ===
def buscar(query, distancia_maxima=1.5):  # ajuste empiricamente
    query_embedding = gerar_embedding(query)
    resultados = collection.query(
        query_embeddings=[query_embedding],
        n_results=50
    )

    filtrados = []
    for doc, meta, score in zip(resultados['documents'][0],
                                resultados['metadatas'][0],
                                resultados['distances'][0]):
        # score é L2 distance, menor é mais parecido
        if score <= distancia_maxima:
            similaridade = max(0, 1 - score / distancia_maxima)  # opcional, só pra ter uma referência 0-1
            filtrados.append((doc, meta, similaridade))
    return filtrados


# === AGENTE IA COM OLLAMA ===
def interpretar_resultados(query, resultados):
    explicacoes = []
    for doc, meta, sim in resultados:
        prompt = f"""
Você é um assistente que analisa documentos.
Pergunta original: {query}
Documento (similaridade {sim:.2f}): {doc}

Explique resumidamente o que esse documento contém e como ele se relaciona com a pergunta.
"""
        resposta = chat(model="llama3", messages=[{"role": "user", "content": prompt}])
        explicacoes.append({
            "arquivo": meta['arquivo'],
            "chunk": meta['chunk'],
            "similaridade": sim,
            "analise": resposta["message"]["content"]
        })
    return explicacoes

# === LOOP ===
while True:
    consulta = input("\n🔍 Digite sua busca (ou 'sair'): ")
    if consulta.lower() == "sair":
        break

    resultados = buscar(consulta)
    if not resultados:
        print("⚠️ Nenhum resultado encontrado com similaridade suficiente.")
        continue

    analises = interpretar_resultados(consulta, resultados)
    print("\n📄 Relatório de Análise:")
    for analise in analises:
        print(f"Arquivo: {analise['arquivo']} (chunk {analise['chunk']}) | Similaridade: {analise['similaridade']:.2f}")
        print(f"Análise: {analise['analise']}\n")
