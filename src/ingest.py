import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_embeddings():
    load_dotenv()
    provider = os.getenv("PROVIDER", "openai").lower()
    
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Erro: A variável OPENAI_API_KEY não está definida no arquivo .env")
            sys.exit(1)
        # O default para text-embedding-3-small é dimensão 1536
        return OpenAIEmbeddings(model="text-embedding-3-small")
    
    elif provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("Erro: A variável GOOGLE_API_KEY não está definida no arquivo .env")
            sys.exit(1)
        # O models/gemini-embedding-001 produz embeddings usando a API do Google (versão atualizada do embedding-001)
        return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    else:
        print(f"Erro: Provider '{provider}' não suportado. Use 'openai' ou 'gemini'.")
        sys.exit(1)

def main():
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("Erro: A variável DATABASE_URL não está definida.")
        sys.exit(1)

    pdf_path = "document.pdf"
    if not os.path.exists(pdf_path):
        print(f"Erro: O arquivo {pdf_path} não foi encontrado na raiz do projeto.")
        print("Traga um PDF válido e nomeie como 'document.pdf' na mesma pasta onde rodar este comando.")
        sys.exit(1)

    try:
        print("Iniciando o carregamento do PDF...")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        print("Dividindo o texto em chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )
        chunks = text_splitter.split_documents(documents)
        print(f"PDF dividido em {len(chunks)} chunks.")

        embeddings = get_embeddings()

        print("Conectando ao banco de dados e enviando embeddings...")
        
        # Cria ou atualiza a coleção 'documents' limpando eventuais dados de uma ingestão anterior (idempotência)
        vectorstore = PGVector.from_documents(
            embedding=embeddings,
            documents=chunks,
            collection_name="documents",
            connection=db_url,
            pre_delete_collection=True
        )

        print(f"Ingestão concluída! {len(chunks)} chunks armazenados.")
    except Exception as e:
        print(f"Ocorreu um erro de conexão com o banco ou API durante a ingestão: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
