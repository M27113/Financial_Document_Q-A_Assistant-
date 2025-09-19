# Financial_Document_Q-A_Assistant

### 🎯 Project Overview

The Financial Document Q&A Assistant is a web application that allows users to upload financial documents (PDF, Excel, and text files) and interactively query them using natural language. The application extracts key financial information such as revenue, expenses, profits, and other metrics, providing accurate answers in a conversational interface.
This project demonstrates the integration of document processing, natural language question-answering, and a user-friendly web interface.


### 📂 Features
-- Document Processing

-- Accepts PDF, Excel, and text file uploads.

-- Extracts both textual and numerical financial data.

-- Supports common financial documents like: Income Statements, Balance Sheets, Cash Flow Statements

-- Handles different layouts and formats.

### Question-Answering System

-- Users can ask natural language questions about financial metrics.

-- Provides accurate and contextual responses from uploaded documents.

-- Supports conversational follow-up questions.

### Web Application

-- Built using Streamlit for an intuitive web interface.

-- Interactive chat interface for querying financial data.

-- Displays extracted financial information in a readable format.

-- Shows processing status and provides clear feedback.

### ⚙️ Technical Implementation

-- Backend: Python with document processing libraries (PyPDFLoader, pandas, openpyxl)

-- NLP: OpenAI Chat API (used instead of Ollama for local Small Language Models)

-- Frontend: Streamlit web interface

-- Local Deployment: Runs entirely on local machine, no cloud hosting required

-- Error Handling: Graceful messages for unsupported file types or missing data

**Note** : This project uses OpenAI API instead of Ollama SLM due to accessibility and ease of development.

## 🚀 Setup & Run

1. Clone repo

   git clone https://github.com/M27113/Financial_Document_Q-A_Assistant-.git
   cd Financial_Document_Q-A_Assistant-

2. Create and activate a virtual environment

   python -m venv venv
   -- Windows
   venv\Scripts\activate
   -- Mac/Linux
   source venv/bin/activate
3. Install dependencies

   pip install -r requirements.txt

4. Set your OpenAI API key (if applicable)

   export OPENAI_API_KEY="your_api_key_here"  # Mac/Linux
   setx OPENAI_API_KEY "your_api_key_here"     # Windows
   
5. Run the app:
   bash :
   streamlit run app1.py
   
-- Open browser at http://localhost:8501
-- Upload financial documents
-- Ask questions interactively

### 📂 Supported File Types

-- PDF (.pdf)

-- Excel (.xlsx, .xls)

-- Text files (.txt)

### Usage

-- Open the app in your browser (http://localhost:8501)

-- Upload your PDF, Excel, or TXT financial statements

-- Ask questions in natural language, e.g.:

1. What is my total equity?
2. Show net profit from the income statement.
3. How much cash was generated this year?

-- If RAG is enabled, answers will use the uploaded documents as context

## 📖 Future Improvements

-- Add support for more file formats (CSV, Google Sheets).

-- Improve document layout handling for complex tables.

-- Implement advanced conversational memory for longer Q&A sessions.

-- Add automated tests for document parsing and question-answering accuracy.


## Sample Output
Once you succesfully setup this project, your web application should look like this : 

![image]()

![image]()

![image]()
