import streamlit as st
import os
import dotenv
import uuid
import re
import pandas as pd
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from rag_methods1 import load_financial_doc_to_db, stream_llm_response, stream_llm_rag_response

dotenv.load_dotenv()
MODELS = ["openai/gpt-4o", "openai/gpt-4o-mini"]

# --- Page Config ---
st.set_page_config(
    page_title="Financial RAG Assistant",
    page_icon="📊",
    layout="wide"
)

st.markdown("<h2 style='text-align:center; color:#1E90FF;'>📊 Financial RAG Assistant 🤖💬</h2>", unsafe_allow_html=True)

# --- Session State ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user","content":"Hello"},
        {"role": "assistant","content":"Hi! I can help you analyze financial documents."}
    ]

# --- Helper Functions ---
def highlight_financials(text):
    patterns = {
        "Revenue":"blue","Cost of Goods Sold":"red","Gross Profit":"green","Operating Expenses":"red",
        "Net Income":"green","Profit":"green","Cash":"orange","Cash Flow":"orange",
        "Assets":"purple","Liabilities":"purple","Equity":"purple","Retained Earnings":"purple",
        "Total Assets":"purple","Total Liabilities":"purple"
    }
    for kw, color in patterns.items():
        pattern = re.compile(rf"({kw}\s*(?:is|:)?\s*(?:INR|\$)?\s*[\d,]+)", re.IGNORECASE)
        text = pattern.sub(rf"<span style='color:{color}; font-weight:bold;'>\1</span>", text)
    return text

def extract_financial_summary(text):
    summary = {}
    patterns = {
        "Revenue": r"Revenue\s*(?:is|:)?\s*(?:INR|\$)?\s*([\d,]+)",
        "Net Income": r"(Net Income|Profit)\s*(?:is|:)?\s*(?:INR|\$)?\s*([\d,]+)",
        "Cash": r"Cash\s*(?:is|:)?\s*(?:INR|\$)?\s*([\d,]+)",
        "Assets": r"Assets\s*(?:is|:)?\s*(?:INR|\$)?\s*([\d,]+)",
        "Equity": r"(Equity|Total Equity).*?([\d,]+)"
    }
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        summary[key] = matches[0][1].replace(",","") if matches else "N/A"
    return summary

def display_summary_table(summary):
    df = pd.DataFrame([summary])
    def color_cells(val):
        if val=="N/A": return "color: gray"
        try: num = float(val.replace(",","")); return "color: green" if num>=0 else "color: red"
        except: return ""
    styled_df = df.style.applymap(color_cells)
    st.markdown("**Financial Summary:**")
    st.dataframe(styled_df, use_container_width=True)

# --- Sidebar ---
with st.sidebar:
    default_openai_api_key = os.getenv("OPENAI_API_KEY") or ""
    openai_api_key = st.text_input("🔐 OpenAI API Key", value=default_openai_api_key, type="password")
    st.session_state["openai_api_key"] = openai_api_key
    missing_openai = openai_api_key in ("", None) or "sk-" not in openai_api_key
    if missing_openai:
        st.warning("⬅️ Please provide a valid OpenAI API key.")
    else:
        st.selectbox("🤖 Select a Model", [m for m in MODELS if "openai" in m], key="model")
        cols0 = st.columns([1,1])
        with cols0[0]:
            is_vector_db_loaded = ("vector_db" in st.session_state and st.session_state.vector_db is not None)
            st.checkbox("Use RAG", value=is_vector_db_loaded, key="use_rag", disabled=not is_vector_db_loaded)
        with cols0[1]:
            st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")

        st.file_uploader(
            "📄 Upload PDF, Excel, TXT",
            type=["pdf", "xlsx", "txt"],
            accept_multiple_files=True,
            on_change=load_financial_doc_to_db,
            key="rag_docs"
        )
        with st.expander(f"📁 Documents in DB ({0 if not is_vector_db_loaded else len(st.session_state.rag_sources)})"):
            st.write([] if not is_vector_db_loaded else st.session_state.rag_sources)

# --- Main Chat Interface ---
model_name = st.session_state.get("model", "openai/gpt-4o")
llm_stream = ChatOpenAI(api_key=openai_api_key, model_name=model_name.split("/")[-1], temperature=0.3, streaming=True)

# Display chat history
for message in st.session_state.messages:
    role = "You" if message["role"]=="user" else "Assistant"
    content = highlight_financials(message["content"]) if role=="Assistant" else message["content"]
    st.markdown(f"**{role}:** {content}", unsafe_allow_html=True)

st.markdown("---")

# User input
if prompt := st.chat_input("Ask your question about financial statements"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"**You:** {prompt}")

    messages = [
        HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"])
        for m in st.session_state.messages
    ]

    # Stream response
    response_text = ""
    if not st.session_state.use_rag:
        for chunk in stream_llm_response(llm_stream, messages):
            response_text += chunk.content
    else:
        for chunk in stream_llm_rag_response(llm_stream, messages):
            response_text += chunk

    # Clean text
    cleaned_text = re.sub(r'\bYour\b\s*', '', response_text)
    cleaned_text = re.sub(r'\s+\n', '\n', cleaned_text)
    cleaned_text = re.sub(r'\n+', '\n', cleaned_text)
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)

    st.markdown(f"**Assistant:** {highlight_financials(cleaned_text)}", unsafe_allow_html=True)

    # Extract financial summary
    summary = extract_financial_summary(cleaned_text)

    # Only display table if at least one metric is meaningful
    if any(val != "N/A" and val != "0" for val in summary.values()):
        display_summary_table(summary)
