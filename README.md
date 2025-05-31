# ArXiv Daily

Customized ArXiv Paper Feeds with PDF Processing and RAG Database

## Overview

ArXiv Daily is a comprehensive system that:

1. **Retrieves** daily arXiv papers matching your specific keywords
2. **Downloads** and extracts content from paper PDFs
3. **Summarizes** papers using LLM (OpenAI)
4. **Stores** everything in a SQLite database for RAG applications
5. **Enables** question-answering about your research papers using RAG
6. **Notifies** you by email about relevant papers

## Project Structure

The project has been organized into a modular structure:

```
arxiv-daily/
├── arxiv_daily/               # Main package
│   ├── api/                   # API interfaces
│   │   └── arxiv_api.py       # ArXiv API client
│   ├── configs/               # Configuration
│   │   └── config.py          # Central configuration
│   ├── processing/            # Data processing
│   │   ├── llm_processor.py   # LLM summarization
│   │   ├── paper_processor.py # Paper filtering & email generation
│   │   └── pdf_manager.py     # PDF downloading & extraction
│   ├── rag/                   # RAG capabilities
│   │   └── query_engine.py    # Question answering system
│   ├── storage/               # Data persistence
│   │   ├── database_manager.py # SQLite database management
│   │   └── vector_store.py    # Vector embeddings for RAG
│   └── utils/                 # Utilities
│       └── email_sender.py    # Email notifications
├── pdfs/                      # Downloaded PDFs (created automatically)
├── main.py                    # Main entry point
├── query.py                   # RAG question-answering CLI
├── simple_query.py            # Simplified RAG using keywords
├── rag_demo.py                # Gradio demo comparing RAG vs. no RAG
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   ```
   export GMAIL_APP_PWD="your_gmail_app_password"
   export OPENAI_API_KEY="your_openai_api_key"
   ```

## Usage

### Basic Usage

Run the daily job to fetch today's papers:

```bash
python main.py
```

### Query a Specific Date

Retrieve papers from a specific date:

```bash
python main.py --date 2025-03-15
```

### Search by Keyword

Find papers in your database matching a specific keyword:

```bash
python main.py --keyword "dataset distillation"
```

### Run as a Scheduled Job

Run continuously as a scheduled daily job (runs at 8:00 AM Singapore time):

```bash
python main.py --schedule
```

### Skip PDF Processing

If you just want the email without downloading PDFs:

```bash
python main.py --no-pdf
```

### Enable LLM Summarization

Generate AI summaries of papers using OpenAI models:

```bash
python main.py --llm gpt-4o-mini
```

## RAG Capabilities

ArXiv Daily includes powerful Retrieval-Augmented Generation (RAG) capabilities that allow you to ask questions about your paper collection.

### Process Papers for RAG

First, process your papers to generate embeddings for RAG:

```bash
python main.py --process-for-rag
```

### Query Your Paper Database

Ask questions about papers in your database:

```bash
python query.py --question "What are the latest methods for dataset distillation?"
```

Or use interactive mode:

```bash
python query.py --interactive
```

### Simple Keyword-Based RAG

For quick searches without OpenAI API:

```bash
python simple_query.py --interactive
```

### Compare RAG vs. Standard LLM

See how RAG improves answers with your paper knowledge:

```bash
python rag_demo.py
```

This launches a Gradio web interface showing side-by-side comparison of answers with and without RAG.

## Customization

Edit `arxiv_daily/configs/config.py` to customize:

- Keywords to search for
- Email settings
- PDF download preferences
- LLM model and settings
- Database configuration
- RAG chunking parameters

## Future Enhancements

- Implement embedding-based keyword detection
- Add support for other LLM providers
- Create visualization tools for research trends
- Integrate with reference management systems
