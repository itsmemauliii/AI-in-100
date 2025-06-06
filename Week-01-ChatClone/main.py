import os
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# We no longer need the 'data' folder on the server side
# The 'load_docs' function will now take uploaded files directly.

def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def build_embeddings(docs):
    # Handle cases where docs might be empty after upload issues
    if not docs:
        return None, None # Return None for vectorizer and embeddings
    vectorizer = TfidfVectorizer().fit(docs)
    embeddings = vectorizer.transform(docs)
    return vectorizer, embeddings

def answer_question(question, vectorizer, embeddings, docs, similarity_threshold=0.2):
    if vectorizer is None or embeddings is None:
        return "Please upload documents first to enable the chat functionality."

    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, embeddings).flatten()
    best_idx = np.argmax(similarities)
    max_similarity = similarities[best_idx]

    if max_similarity < similarity_threshold:
        return "I apologize, but my current knowledge base (from your uploaded documents) does not contain information on that topic. Please rephrase your question or upload more relevant documents."
    else:
        return docs[best_idx]

def main():
    st.title("ChatClone - AI in 100 Lines")

    uploaded_files = st.file_uploader(
        "Upload your .txt documents here:",
        type="txt",
        accept_multiple_files=True
    )

    docs = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Read the content of the uploaded file
            string_data = uploaded_file.read().decode("utf-8")
            docs.append(string_data)
        st.success(f"Successfully loaded {len(uploaded_files)} document(s).")
    else:
        st.info("Please upload some .txt documents to start chatting.")
        st.stop() # Stop execution if no files are uploaded

    chunked_docs = []
    if docs: # Only proceed if documents were actually loaded
        for doc in docs:
            chunked_docs.extend(chunk_text(doc))

        st.write(f"Processed {len(chunked_docs)} chunks from your documents.")

        vectorizer, embeddings = build_embeddings(chunked_docs)

        if vectorizer is not None and embeddings is not None:
            query = st.text_input("Ask me anything about your uploaded docs!")

            if query:
                response = answer_question(query, vectorizer, embeddings, chunked_docs)
                st.write(response[:500] + " ...")
        else:
            st.warning("Could not build embeddings. Please ensure your documents are not empty.")
            st.stop()
    else:
        st.warning("No text content found in uploaded files. Please upload valid .txt files.")
        st.stop()


if __name__ == "__main__":
    main()
