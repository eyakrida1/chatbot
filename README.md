# chatbot
# Cosmetics E-commerce RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about cosmetics products using a product database and a local Large Language Model.

The project combines LangChain, Ollama, embeddings, and FAISS to create an intelligent product assistant capable of searching a cosmetics catalog and generating accurate responses based on retrieved product information.

## Features

* Load cosmetics product data from CSV files
* Convert product descriptions into searchable vector representations
* Semantic product search using FAISS
* Retrieval-Augmented Generation (RAG) architecture
* Local LLM inference using Ollama
* Conversation history support
* Prevent hallucination by forcing answers to rely on retrieved context

## Architecture

```
User Question
      |
      v
Retriever (FAISS)
      |
      v
Relevant Product Documents
      |
      v
Prompt Construction
      |
      v
Gemma LLM (Ollama)
      |
      v
Generated Answer
```

## Technologies Used

### Programming Language

* Python

### AI / NLP

* LangChain
* Ollama
* Gemma 3 LLM
* Nomic Text Embeddings

### Vector Database

* FAISS

### Data Processing

* Pandas



## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd project
```

### 2. Install dependencies


### 3. Install Ollama

Download and install Ollama:

https://ollama.com/

Pull the required models:

```bash
ollama pull gemma3
```

```bash
ollama pull nomic-embed-text
```

## Dataset

The chatbot uses a CSV file containing cosmetics product information.

Required columns:

```
id
nom
prix
description
marque_name
category_name
```

Example:

| Column        | Description         |
| ------------- | ------------------- |
| nom           | Product name        |
| prix          | Product price       |
| description   | Product description |
| marque_name   | Brand               |
| category_name | Category            |

## How It Works

1. Product data is loaded from `products01.csv`.
2. Each product is converted into a LangChain Document.
3. Product descriptions are transformed into embeddings using Nomic Embeddings.
4. Embeddings are stored in a FAISS vector database.
5. User questions are converted into search queries.
6. Relevant products are retrieved.
7. Retrieved information is inserted into the prompt.
8. Gemma generates the final answer.

## Running the Chatbot

Start the application:

```bash
python chatbot.py
```

Example:

```
you: Do you have red lipstick?

chat:
Yes, several red lipsticks are available including...
```

## RAG Workflow

The chatbot follows the RAG pattern:

### Retrieval

Search the product database for relevant information.

### Augmentation

Add retrieved product information to the prompt.

### Generation

Use the LLM to generate a final answer.



## Author

Eya Krida
