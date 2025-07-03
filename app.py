from datetime import datetime

import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import (
  GoogleGenerativeAIEmbeddings
)


# ------------------ Model Configuration ------------------ #
MODEL_OPTIONS = {
  "Groq": {
    "playground": "https://console.groq.com/",
    "models": ["llama-3.1-8b-instant", "llama3-70b-8192"]
  },
  "Gemini": {
    "playground": "https://ai.google.dev",
    "models": ["gemini-2.0-flash", "gemini-2.5-flash"]
  }
}

# ------------------ Utility Functions ------------------ #
def get_pdf_text(pdf_files):
  text = ""
  for file in pdf_files:
    reader = PdfReader(file)
    for page in reader.pages:
      text += page.extract_text() or ""
  return text

def get_text_chunks(text):
  splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
  return splitter.split_text(text)

def get_embeddings(provider, api_key=None):
  if provider.lower() == "groq":
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  elif provider.lower() == "gemini":
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
  else:
    raise ValueError("Unsupported provider")

def get_vectorstore(chunks, provider, api_key):
  embedding = get_embeddings(provider, api_key)
  store = FAISS.from_texts(chunks, embedding)
  store.save_local(f"./data/{provider.lower()}_vector_store.faiss")
  return store

def process_and_store_pdfs(pdfs, provider, api_key):
  raw_text = get_pdf_text(pdfs)
  chunks = get_text_chunks(raw_text)
  store = get_vectorstore(chunks, provider, api_key)
  st.session_state.vector_store = store
  st.session_state.pdfs_submitted = True

def render_uploaded_files():
  pdf_files = st.session_state.get("pdf_files", [])
  if pdf_files:
    with st.expander("**📎 Uploaded Files:**"):
      for f in pdf_files:
        st.markdown(f"- {f.name}")

# ------------------ Main App ------------------ #
def main():
  st.set_page_config(page_title="RAG PDFBot", layout="centered")
  st.title("👽 RAG PDFBot")
  st.caption("Chat with multiple PDFs :books:")

  for key, default in {
    "chat_history": [],
    "pdfs_submitted": False,
    "vector_store": None,
    "pdf_files": [],
    "last_provider": None,
    "unsubmitted_files": False,
  }.items():
    if key not in st.session_state:
      st.session_state[key] = default

  # Sidebar Configuration
  with st.sidebar:
    with st.expander("⚙️ Configuration", expanded=True):
      model_provider = st.selectbox("🔌 Model Provider", ["Select a model provider"] + list(MODEL_OPTIONS.keys()), index=0, key="model_provider")

      if model_provider == "Select a model provider":
        return

      api_key = st.text_input("🔑 Enter your API Key", help=f"Get API key from [here]({MODEL_OPTIONS[model_provider]['playground']})")
      if not api_key:
        return

      models = MODEL_OPTIONS[model_provider]["models"]
      model = st.selectbox("🧠 Select a model", models, key="model")

      uploaded_files = st.file_uploader("📚 Upload PDFs", type=["pdf"], accept_multiple_files=True, key="pdf_uploader")

      if uploaded_files and uploaded_files != st.session_state.pdf_files:
        st.session_state.unsubmitted_files = True

      if st.button("➡️ Submit"):
        if uploaded_files:
          with st.spinner("Processing PDFs..."):
            process_and_store_pdfs(uploaded_files, model_provider, api_key)
            st.session_state.pdf_files = uploaded_files
            st.session_state.unsubmitted_files = False
            st.toast("PDFs processed successfully!", icon="✅")
        else:
          st.warning("No files uploaded.")

      if model_provider != st.session_state.last_provider:
        st.session_state.last_provider = model_provider
        if st.session_state.pdf_files:
          with st.spinner("Reprocessing PDFs..."):
            process_and_store_pdfs(st.session_state.pdf_files, model_provider, api_key)
            st.toast("PDFs reprocessed successfully!", icon="🔁")

    with st.expander("🛠️ Tools", expanded=False):
      col1, col2, col3 = st.columns(3)

      if col1.button("🔄 Reset"):
        st.session_state.clear()
        st.session_state.model_provider = "Select a model provider"
        st.rerun()

      if col2.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.pdf_files = None
        st.session_state.vector_store = None
        st.session_state.pdfs_submitted = False
        st.toast("Chat and PDF cleared.", icon="🧼")

      if col3.button("↩️ Undo") and st.session_state.chat_history:
        st.session_state.chat_history.pop()
        st.rerun()

  if st.session_state.pdfs_submitted and st.session_state.pdf_files:
    render_uploaded_files()

  for q, a, *_ in st.session_state.chat_history:
    with st.chat_message("user"):
      st.markdown(q)
    with st.chat_message("ai"):
      st.markdown(a)

  if st.session_state.unsubmitted_files:
    st.warning("📄 New PDFs uploaded. Please submit before chatting.")
    return

  if st.session_state.pdfs_submitted:
    question = st.chat_input("💬 Ask a Question from the PDF Files")
    if question:
      with st.chat_message("user"):
        st.markdown(question)
  else:
    st.info("📄 Please upload and submit PDFs to start chatting.")

if __name__ == "__main__":
  main()
