import streamlit as st

def main():
    st.title("ChatClone - AI in 100 Lines")
    docs = load_docs()
    chunked_docs = []
    for doc in docs:
        chunked_docs.extend(chunk_text(doc))

    if not chunked_docs:
        st.error("No documents found in 'data' folder. Add .txt files there!")
        return

    vectorizer, embeddings = build_embeddings(chunked_docs)

    st.write(f"Loaded {len(chunked_docs)} document chunks.")

    query = st.text_input("Ask me anything about your docs!")

    if query:
        response = answer_question(query, vectorizer, embeddings, chunked_docs)
        st.write(response[:500] + " ...")  # Limit output length
