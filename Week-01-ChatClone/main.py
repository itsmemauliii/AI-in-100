import os
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_docs(folder='data'):
    if not os.path.exists(folder):
        st.error(f"Folder '{folder}' not found. Please create it and add .txt files.")
        return []
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
    st.title("ChatClone - AI in 100 Lines")
    
    docs = load_docs()
    if not docs:
        st.stop()  # Stop running if no docs found
    
    chunked_docs = []
    for doc in docs:
        chunked_docs.extend(chunk_text(doc))
    
    st.write(f"Loaded {len(chunked_docs)} chunks from docs.")
    
    vectorizer, embeddings = build_embeddings(chunked_docs)
    
    query = st.text_input("Ask me anything about your docs!")
    
    if query:
        response = answer_question(query, vectorizer, embeddings, chunked_docs)
        st.write(response[:500] + " ...")

if __name__ == "__main__":
    main()
