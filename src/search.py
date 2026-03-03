import os
import sys
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_embeddings():
    load_dotenv()
    provider = os.getenv("PROVIDER", "openai").lower()
    
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Erro: A variável OPENAI_API_KEY não está definida no arquivo .env")
        return OpenAIEmbeddings(model="text-embedding-3-small")
    
    elif provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Erro: A variável GOOGLE_API_KEY não está definida no arquivo .env")
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    else:
        raise ValueError(f"Erro: Provider '{provider}' não suportado. Use 'openai' ou 'gemini'.")

def get_vectorstore():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("Erro: A variável DATABASE_URL não está definida.")
    
    embeddings = get_embeddings()
    
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name="documents",
        connection=db_url,
    )
    return vectorstore

def search_documents(query: str, k: int = 10) -> str:
    """Busca os k resultados mais relevantes e retorna o contexto concatenado."""
    vectorstore = get_vectorstore()
    
    # Fazemos a busca de similaridade e capturamos também os scores se necessário
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    if not results:
        return ""
    
    context_chunks = []
    for doc, _score in results:
        # Recuperamos o conteúdo em texto do chunk encontrado no banco
        context_chunks.append(doc.page_content)
        
    return "\n\n".join(context_chunks)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python src/search.py 'sua pergunta aqui'")
        sys.exit(1)
        
    query = " ".join(sys.argv[1:])
    try:
        context = search_documents(query)
        print("RESULTADOS DA BUSCA:\n")
        print(context)
    except Exception as e:
        print(f"Erro ao buscar no banco de dados vetorial: {e}")
