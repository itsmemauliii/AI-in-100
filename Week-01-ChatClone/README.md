# ChatClone → AI in 100 Lines Week 1 🔥

## What is this?

A lightweight, chatbot that answers questions from your own text documents, no API keys, no cloud, just pure Python + embeddings.

## Why?

I built this because I was sick of paying for AI tokens and waiting on APIs. This is my own ChatGPT-lite for personal knowledge.

## How it works:

- Takes `.txt` files as input
- Splits them into chunks
- Creates simple embeddings using `sentence-transformers` or `sklearn`
- Answers user queries with cosine similarity search
- Runs locally in under 100 lines of code

## How to run

1. Clone repo  
2. Put your `.txt` files in `data/` folder  
3. Run `python main.py`  
4. Ask your chatbot anything related to your docs!
