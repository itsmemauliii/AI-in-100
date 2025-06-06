# ChatClone: Simple embedding chatbot under 100 lines

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_docs(folder='data'):
    docs = []
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
                docs.append(f.read())
    return docs

def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def build_embeddings(docs):
    vectorizer = TfidfVectorizer().fit(docs)
    embeddings = vectorizer.transform(docs)
    return vectorizer, embeddings

def answer_question(question, vectorizer, embeddings, docs):
    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, embeddings).flatten()
    best_idx = np.argmax(similarities)
    return docs[best_idx]

def main():
    print("Loading docs...")
    docs = load_docs()
    chunked_docs = []
    for doc in docs:
        chunked_docs.extend(chunk_text(doc))

    print(f"Loaded {len(chunked_docs)} chunks.")

    vectorizer, embeddings = build_embeddings(chunked_docs)

    print("Ask me anything about your docs! Type 'exit' to quit.")

    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break
        response = answer_question(query, vectorizer, embeddings, chunked_docs)
        print("Bot:", response[:500], "...")  # Limit output length

if __name__ == '__main__':
    main()
