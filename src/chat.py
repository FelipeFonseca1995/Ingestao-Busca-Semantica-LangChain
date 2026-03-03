import os
import sys
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Garante que possamos importar search da pasta src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.search import search_documents

def get_llm():
    load_dotenv()
    provider = os.getenv("PROVIDER", "openai").lower()
    
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Erro: A variável OPENAI_API_KEY não está definida no arquivo .env")
            sys.exit(1)
        return ChatOpenAI(model="gpt-5-nano", temperature=0)
        
    elif provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("Erro: A variável GOOGLE_API_KEY não está definida no arquivo .env")
            sys.exit(1)
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    else:
        print(f"Erro: Provider '{provider}' não suportado.")
        sys.exit(1)

def run_chat():
    load_dotenv()
    
    try:
        llm = get_llm()
    except Exception as e:
        print(f"Erro ao inicializar conexão com LLM (verifique suas API Keys): {e}")
        return

    # Prompt exigido pelo projeto
    prompt_template = """CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta_usuario}

RESPONDA A "PERGUNTA DO USUÁRIO" """

    prompt = PromptTemplate(
        input_variables=["contexto", "pergunta_usuario"],
        template=prompt_template
    )
    
    chain = prompt | llm
    
    print("========================================")
    print(" Bem-vindo ao Chat (Busca Semântica) ")
    print(" Para sair, digite 'sair' ou 'exit' ")
    print("========================================\n")
    
    while True:
        try:
            print("Faça sua pergunta:")
            user_input = input("PERGUNTA: ")
            
            if user_input.strip().lower() in ["sair", "exit"]:
                print("Encerrando o chat. Até mais!")
                break
                
            if not user_input.strip():
                continue
                
            # Vetoriza a pergunta utilizando o search.py (k=10 resultados) e já formata via chunk de contexto
            contexto = search_documents(user_input, k=10)
            
            # Chama a LLMS pasando o conteúdo recebido do Banco Vetorial e a Pergunta
            resposta_llm = chain.invoke({
                "contexto": contexto,
                "pergunta_usuario": user_input
            })
            
            print(f"RESPOSTA: {resposta_llm.content}\n")
            print("***\n")
            
        except KeyboardInterrupt:
            print("\nEncerrando o chat...")
            break
        except Exception as e:
            print(f"Ocorreu um erro durante a consulta: {e}")

if __name__ == "__main__":
    run_chat()
