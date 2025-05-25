# pdfPulse

## Overview

RAG PDF Chatbot is an advanced document analysis application that enables users to upload PDF documents and interact with their content through an intelligent, context-aware question-answering system. By leveraging cutting-edge AI technologies, this application transforms static PDF documents into interactive knowledge bases.

## Key Features

- **PDF Text Extraction**: Seamlessly extract text from uploaded PDF documents
- **Semantic Search**: Utilize advanced embedding techniques to find contextually relevant information
- **AI-Powered Responses**: Generate precise, context-aware answers using Google's Gemini AI
- **Vector Database Integration**: Implement efficient information retrieval with Pinecone vector database

## Technology Stack

- Streamlit
- Google Gemini AI
- Pinecone Vector Database
- PyPDF2
- Python 3.8+

## Preview

![.](images/pdfPulse%20Preview.jpg)

## Prerequisites

- Python 3.8 or higher
- Pinecone API Key
- Google Gemini API Key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/leeh-nix/pdfPulse.git
cd pdfPulse
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Create a `.secret` file in the `.streamlit` folder (optional, for local development):
```
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

## Running the Application

```bash
streamlit run app.py
```

## Usage Instructions

1. Navigate to the Streamlit application
2. Enter your Gemini and Pinecone API keys in the sidebar
3. Upload a PDF document
4. Ask questions about the document's content
5. Receive contextually accurate responses

## How It Works

The application implements a Retrieval-Augmented Generation (RAG) workflow:
1. Extract text from uploaded PDF
2. Split text into semantic chunks
3. Generate vector embeddings for chunks
4. Store embeddings in Pinecone vector database
5. When a query is made, retrieve the most relevant chunks
6. Generate a response using retrieved context

## Security Note

- API keys are used locally and not stored
- Sensitive information remains in memory during the session

## Limitations

- Supports PDF documents only
- Requires active internet connection
- Response accuracy depends on document complexity and AI model capabilities

## Contributing

Contributions are welcome! Please submit pull requests or open issues on the GitHub repository.

## Acknowledgments

- [Streamlit](https://streamlit.io/)
- [Gemini AI](https://ai.google.dev/)
- [Pinecone](https://www.pinecone.io/)
