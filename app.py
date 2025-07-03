from datetime import datetime

import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader


# ------------------ Main App ------------------ #
def main():
  st.set_page_config(page_title="RAG PDFBot", layout="centered")
  st.title("👽 RAG PDFBot")
  st.caption("Chat with multiple PDFs :books:")

if __name__ == "__main__":
  main()
