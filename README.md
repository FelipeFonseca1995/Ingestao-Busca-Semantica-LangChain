# Ingestão e Busca Semântica com LangChain e Postgres

Projeto de software que realiza a ingestão de informações de um arquivo PDF para um banco de dados vetorial PostgreSQL (usando `pgVector`) e permite realizar buscas semânticas através de um chat no terminal. O chat foi construído usando o framework open-source **LangChain** e tem suporte para os modelos da **OpenAI** e do **Google Gemini**.

## Estrutura do Projeto

```
├── docker-compose.yml
├── requirements.txt      # Dependências
├── .env.example          # Template das variáveis de ambiente
├── src/
│   ├── ingest.py         # Script de ingestão do PDF no banco vetorial
│   ├── search.py         # Módulo para a busca vetorial propriamente dita
│   ├── chat.py           # CLI interativo para chat com a IA
├── document.pdf          # PDF de exemplo para ingestão
└── README.md             # Instruções de execução do projeto
```

## Funcionalidades
1. **Ingestão:** Processa um arquivo `document.pdf`, dividindo o texto em chunks e gerando os vetores (embeddings) para persistência no banco garantindo a não-duplicação na coleção de documentos.
2. **Busca via CLI:** Interface interativa de texto que responde a perguntas **somente com base no conteúdo** do PDF ingerido anteriormente. O sistema irá tratar casos com ressalvas, se as informações não existirem, informando o usuário apropriadamente.

## Pré-requisitos
- **Python:** 3.10 ou superior
- **Docker e Docker Compose**
- Chaves de API para uso das inteligências artificiais:
  - `OPENAI_API_KEY` (se for usar o modelo text-embedding-3-small e o gpt-5-nano da OpenAI)
  - `GOOGLE_API_KEY` (se for usar os modelos embedding-001 e gemini-2.5-flash-lite do Google Gemini)

## Instruções de Instalação

1. Clone o repositório ou navegue até o diretório do projeto:

2. Crie e ative um ambiente virtual de Python (*virtualenv*):
**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Instale as bibliotecas necessárias declaradas no `requirements.txt`:
```bash
pip install -r requirements.txt
```

4. Configuração das Variáveis de Ambiente:
- Crie um arquivo com o nome `.env` com base no ` .env.example` preexistente (ou copie-o):
```bash
cp .env.example .env
```
- Insira as suas chaves e verifique a configuração do `PROVIDER` que definirá a provedora de IA a ser usada no processamento da linguagem e do embedding (`openai` ou `gemini`).
- O `DATABASE_URL` já vem configurado de fábrica para a rede local onde estará executando o seu Postgres pelo Docker.

## Ordem de execução

1. **Subir o Banco de Dados com Docker:**
```bash
docker compose up -d
```
*Garantirá o download da imagem baseada em PostgreSQL 16 do pgVector e subirá um contêiner sob a porta 5432.*

2. **Ingestão do arquivo PDF:**
*Certifique-se que o seu arquivo de origem tem o nome `document.pdf` (e está localizado na raiz do projeto).*
```bash
python src/ingest.py
```
*Ele exibirá no console a extração do PDF, geração dos fragmentos de tamanho limite 1000 e sobreposição de 150 e confirmará o número de chunks carregados.*

3. **Iniciando a interface CLI (Chat):**
```bash
python src/chat.py
```
*Digite sua pergunta ou mande um bom dia e ela será respondida baseando-se restritamente no PDF ingerido. Digite "sair" ou "exit" para fechar a conexão de chat.*

## Como alterar entre OpenAI e Gemini
No arquivo `.env` gerado anteriormente você deve configurar as flags:
- `PROVIDER=openai` (utiliza LLM OpenAI, para isto certifique-se de que a `OPENAI_API_KEY` está devidamente preenchida)
- `PROVIDER=gemini` (utiliza LLM Gemini, para isto certifique-se de que a `GOOGLE_API_KEY` está preenchida devidamente).
