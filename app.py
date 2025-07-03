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

  # Sidebar Configuration
  with st.sidebar:
    with st.expander("⚙️ Configuration", expanded=True):
      model_provider = st.selectbox("🔌 Model Provider", ["Select a model provider"] + list(MODEL_OPTIONS.keys()), index=0, key="model_provider")

      if model_provider == "Select a model provider":
        return

if __name__ == "__main__":
  main()
