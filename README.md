# Financial_Document_Q-A_Assistant

### 📊 Description:
A web application to process financial documents (PDF, Excel, TXT) and provide an interactive question-answering system. Users can query revenue, expenses, profits, and other financial metrics using natural language.

### Features

-- Upload PDF, Excel, or TXT financial statements

-- Extract and analyze text and numerical data

-- Supports Income Statements, Balance Sheets, Cash Flow statements

-- Conversational Q&A interface using local LLM (Ollama)

-- Retrieval-Augmented Generation (RAG) for document-based answers

-- Clean, readable web interface

## 🚀 Setup & Run

1. Clone repo
   bash:
   git clone https://github.com/<your-username>/financial-qa-assistant.git
   cd financial-qa-assistant

2. Install dependencies
   bash :
   pip install -r requirements.txt
   
3. Install Ollama and pull models
   bash :
   # Windows example
   ollama pull llama2
   
4. Run the app:
   bash :
   streamlit run app1.py
   
-- Open browser at http://localhost:8501
-- Upload financial documents
-- Ask questions interactively

## Setup Instructions (Windows)

### Clone the repository:
git clone <your-repo-url>
cd FinancialQA

### Run the setup script (creates venv, installs dependencies, pulls Ollama model, launches app):

.\setup_and_run.ps1

-- Make sure you have Python 3.11+ installed.

### Alternative manual setup (if not using the script):

python -m venv venv
.\venv\Scripts\Activate
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull llama2-mini
streamlit run app1.py

### Usage

-- Open the app in your browser (http://localhost:8501)

-- Upload your PDF, Excel, or TXT financial statements

-- Ask questions in natural language, e.g.:

1. What is my total equity?
2. Show net profit from the income statement.
3. How much cash was generated this year?

-- If RAG is enabled, answers will use the uploaded documents as context

### Dependencies

-- Python 3.11+

-- Streamlit

-- Pandas

-- Openpyxl

-- Ollama

-- LangChain

-- LangChain Community

-- python-dotenv

All dependencies are listed in requirements.txt.

### Sample Documents

-- sample_docs/IncomeStatement.xlsx

-- sample_docs/BalanceSheet.pdf

-- sample_docs/CashFlow.txt

These files allow you to test PDF, Excel, and TXT support.

