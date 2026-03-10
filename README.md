# AI Interview Preparation Assistant (RAG)

An AI-powered interview preparation assistant built using Retrieval-Augmented Generation (RAG).
The system allows users to ask questions from study materials, generate interview questions, and receive feedback on their answers.

## Features

* Chat with your study documents
* AI-generated interview questions
* Automatic answer evaluation
* Multi-document support (PDF + TXT)
* Hybrid retrieval (Vector Search + BM25)
* Chat memory for follow-up questions
* Document upload through the UI
* Streamlit interactive interface

## Tech Stack

* Python
* LangChain
* FAISS Vector Database
* Ollama (Local LLM)
* Streamlit

## System Architecture

Documents → Embeddings → FAISS Vector Database → Hybrid Retrieval → LLM → Streamlit Interface

## Installation

Clone the repository:

git clone https://github.com/sukshithalle/rag-ai-interview-assistant.git
cd rag-ai-interview-assistant

Install dependencies:

pip install -r requirements.txt

Install Ollama and download the model:

ollama pull llama3

Run the application:

streamlit run app.py

## Usage

1. Upload your study materials (PDF/TXT)
2. Ask questions in Chat Mode
3. Practice interview questions in Quiz Mode
4. Get feedback on your answers

## Example

Question:
What is BFS?

Answer:
Breadth First Search is a graph traversal algorithm that explores nodes level by level using a queue.

## Future Improvements

* Voice-based interview mode
* Streaming responses
* Online deployment
* Coding interview practice
