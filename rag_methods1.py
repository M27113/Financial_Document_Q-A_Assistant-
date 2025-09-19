import os
import dotenv
from time import time
from pathlib import Path
import streamlit as st

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage

dotenv.load_dotenv()
DB_DOCS_LIMIT = 10
os.environ["USER_AGENT"] = "myagent"

# -----------------------------
# Stream LLM response
# -----------------------------
def stream_llm_response(llm_stream, messages):
    response_message = ""
    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk
    st.session_state.messages.append({"role": "assistant", "content": response_message})

# -----------------------------
# Load documents into DB
# -----------------------------
def load_financial_doc_to_db():
    if "rag_docs" not in st.session_state or not st.session_state.rag_docs:
        return
    docs = []
    for doc_file in st.session_state.rag_docs:
        if doc_file.name in st.session_state.rag_sources:
            continue
        if len(st.session_state.rag_sources) >= DB_DOCS_LIMIT:
            st.error(f"Maximum documents reached ({DB_DOCS_LIMIT})")
            continue
        os.makedirs("source_files", exist_ok=True)
        file_path = Path(f"./source_files/{doc_file.name}")
        with open(file_path, "wb") as f:
            f.write(doc_file.read())
        try:
            # PDF
            if doc_file.name.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            # Excel
            elif doc_file.name.lower().endswith(".xlsx"):
                loader = UnstructuredExcelLoader(file_path)
            # TXT
            elif doc_file.name.lower().endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                st.warning(f"Unsupported file type: {doc_file.name}")
                continue
            docs.extend(loader.load())
            st.session_state.rag_sources.append(doc_file.name)
        except Exception as e:
            st.error(f"Error loading {doc_file.name}: {e}")
        finally:
            os.remove(file_path)
    if docs:
        _split_and_load_docs(docs)
        st.success(f"Loaded {len(docs)} document(s) successfully!")

# -----------------------------
# Vector DB
# -----------------------------
def initialize_vector_db(docs):
    api_key = st.session_state.get("openai_api_key", os.getenv("OPENAI_API_KEY", ""))
    if not api_key:
        st.error("OpenAI API key is missing.")
        return None
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vector_db = Chroma(
        embedding_function=embeddings,
        collection_name=f"{str(time()).replace('.', '')[:14]}_" + st.session_state['session_id'],
        persist_directory=None
    )
    vector_db.add_documents(docs)
    chroma_client = vector_db._client
    collection_names = sorted([c.name for c in chroma_client.list_collections()])
    while len(collection_names) > 20:
        chroma_client.delete_collection(collection_names[0])
        collection_names.pop(0)
    return vector_db

def _split_and_load_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
    document_chunks = text_splitter.split_documents(docs)
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = initialize_vector_db(document_chunks)
    else:
        st.session_state.vector_db.add_documents(document_chunks)

# -----------------------------
# RAG Chain
# -----------------------------
def _get_context_retriever_chain(vector_db, llm):
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", "Generate a search query to find the most relevant info from documents.")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(llm):
    retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, llm)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """You are a helpful financial assistant. Answer user's queries using context of financial statements.
        Focus on Revenue, Expenses, Profit, Equity, Cash Flow, Balance Sheet highlights.\n{context}"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def stream_llm_rag_response(llm_stream, messages):
    conversation_rag_chain = get_conversational_rag_chain(llm_stream)
    response_message = ""
    for chunk in conversation_rag_chain.pick("answer").stream({
        "messages": messages,
        "input": messages[-1].content
    }):
        response_message += chunk
        yield chunk
    st.session_state.messages.append({"role": "assistant", "content": response_message})
