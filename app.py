from datetime import datetime

import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader


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

# ------------------ Main App ------------------ #
def main():
  st.set_page_config(page_title="RAG PDFBot", layout="centered")
  st.title("👽 RAG PDFBot")
  st.caption("Chat with multiple PDFs :books:")

  for key, default in {
    "pdf_files": [],
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
            # process_and_store_pdfs(uploaded_files, model_provider, api_key)
            st.session_state.pdf_files = uploaded_files
            st.session_state.unsubmitted_files = False
            st.toast("PDFs processed successfully!", icon="✅")
        else:
          st.warning("No files uploaded.")

if __name__ == "__main__":
  main()
