# HuggingFace FastAPI Testing Platform

A comprehensive FastAPI-based platform for testing and evaluating multiple HuggingFace models through an interactive Swagger UI interface. This project allows you to easily configure, deploy, and test various NLP models including summarization, question-answering, sentiment analysis, and named entity recognition.

## Features

- **Multiple Model Support**: Test various NLP tasks in one platform
- **Interactive Swagger UI**: Easy-to-use web interface for model testing
- **Configurable Models**: Simple configuration-based model switching
- **PEFT Adapter Support**: Compatible with Parameter-Efficient Fine-Tuning adapters
- **RESTful API**: Standard REST endpoints for all models

## Supported Model Types

### 📝 Summarization
- **Default Model**: `philschmid/bart-large-cnn-samsum`
- **Task**: Generate concise summaries from longer text

### 🔍 Extractive Question Answering (RAG)
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **QA Model**: `distilbert-base-cased-distilled-squad`
- **Task**: Extract answers from provided context

### 💭 Abstractive Question Answering (RAG)
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **QA Model**: `microsoft/DialoGPT-small`
- **Task**: Generate answers based on context understanding

### 😊 Sentiment Analysis
- **Default Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Task**: Classify text sentiment (positive, negative, neutral)

### 🏷️ Named Entity Recognition
- **Default Model**: `dbmdz/bert-large-cased-finetuned-conll03-english`
- **Task**: Identify and classify named entities in text

## Configuration

The project uses a simple configuration dictionary to specify models:

```python
MODEL_CONFIG = {
    # Summarization models
    "summarization_model": "philschmid/bart-large-cnn-samsum",
    "summarization_peft_adapter": None,
    
    # RAG models for extractive QnA
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "qa_model": "distilbert-base-cased-distilled-squad",
    "qa_peft_adapter": None,
    
    # RAG models for abstractive QnA
    "abstractive_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "abstractive_qa_model": "microsoft/DialoGPT-small",
    "abstractive_qa_peft_adapter": None,
    
    # Sentiment Classification models
    "sentiment_model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "sentiment_peft_adapter": None,
    
    # Named Entity Recognition models
    "ner_model": "dbmdz/bert-large-cased-finetuned-conll03-english",
    "ner_peft_adapter": None,
}
```

## Quick Start

1. **Configure Models**: Update the `MODEL_CONFIG` dictionary with your desired HuggingFace model paths
2. **Run All Cells**: Execute all notebook cells in sequence
3. **Access Swagger UI**: Click the generated link to access the interactive API documentation
4. **Test Models**: Use the Swagger UI to test each model with your own inputs

## Alternative Model Options

### Abstractive QA Alternatives

```python
# Lightweight options
"abstractive_qa_model": "distilgpt2"  # Even smaller option
"abstractive_qa_model": "microsoft/DialoGPT-medium"  # Larger but T4-friendly
```

## Usage Examples

### Summarization
Input long articles, documents, or conversations and receive concise summaries.

### Question Answering
Provide context and questions to get either:
- **Extractive**: Direct text spans from the context
- **Abstractive**: Generated answers based on understanding

### Sentiment Analysis
Analyze text sentiment for social media posts, reviews, or customer feedback.

### Named Entity Recognition
Extract and classify entities like persons, organizations, locations, and more.

## PEFT Adapter Support

The platform supports Parameter-Efficient Fine-Tuning (PEFT) adapters. To use a PEFT adapter:

1. Set the corresponding `*_peft_adapter` field to your adapter path
2. Ensure the adapter is compatible with the base model

## Requirements

- Python 3.8+
- transformers==4.35.2
- sentence-transformers==2.3.1
- faiss-cpu==1.7.4
- fastapi==0.105.0
- uvicorn==0.24.0.post1
- python-multipart==0.0.6
- pyngrok==7.0.0
- langchain==0.0.350
- langchain-community==0.0.13
- pillow==10.0.1
- numpy==1.26.4
- psutil
- peft==0.7.1
- nest-asyncio

## API Endpoints

Once running, the following endpoints will be available:

- `POST /summarize` - Text summarization
- `POST /qa-extractive` - Extractive question answering
- `POST /qa-abstractive` - Abstractive question answering
- `POST /sentiment` - Sentiment analysis
- `POST /ner` - Named entity recognition

## Notes

- Models are optimized for T4 GPU compatibility
- First model loading may take some time depending on model size
- All models can be easily swapped by updating the configuration
