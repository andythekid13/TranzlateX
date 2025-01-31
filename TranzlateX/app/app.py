import streamlit as st
import requests
import time
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor, as_completed

# Hugging Face API Key (Get yours from https://huggingface.co/settings/tokens)
HF_API_KEY = "add the HF key"

# Updated to use a smaller model: MarianMT
API_URL = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-de"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# Define language mappings
languages = {
    "English": "en",  # Source language
    "German": "de"    # Target language
}

def split_text_into_chunks(text, max_words=100):
    """Splits text into smaller chunks based on word count instead of character length."""
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def translate_chunk(chunk, retries=3):
    """Helper function to translate a single chunk in parallel with retry logic."""
    for _ in range(retries):
        payload = {"inputs": chunk, "parameters": {"max_length": 500}}
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        
        if response.status_code == 200:
            return response.json()[0]['translation_text']
        time.sleep(2)  # Wait before retrying
    return f"Error: {response.status_code}, {response.text}"

def translate_text(text, src_lang, tgt_lang):
    """Translates text using Hugging Face's hosted MarianMT model with optimized parallel processing."""
    text_chunks = split_text_into_chunks(text)
    translated_chunks = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(translate_chunk, chunk): chunk for chunk in text_chunks}
        for future in as_completed(futures):
            translated_chunks.append(future.result())
    
    return " ".join(translated_chunks)

def translate_document(file, src_lang, tgt_lang):
    """Translates the contents of an uploaded document with optimized chunking."""
    try:
        content = file.read().decode("utf-8")
        translated_content = translate_text(content, src_lang, tgt_lang)
        return translated_content
    except Exception as e:
        return f"Error processing the document: {e}"

def translate_pdf(file, src_lang, tgt_lang):
    """Translates the text content of an uploaded PDF file with optimized parallel processing."""
    try:
        pdf_reader = PdfReader(file)
        content = "".join([page.extract_text() or "" for page in pdf_reader.pages])
        translated_content = translate_text(content, src_lang, tgt_lang)
        return translated_content
    except Exception as e:
        return f"Error processing the PDF: {e}"

# Streamlit UI
st.set_page_config(page_title="TranzlateX - AI Translator", layout="wide")
st.title("TranzlateX - AI-Powered Translation")

# Layout Styling
st.markdown("<style>body {background-color: #f7f9fc;}</style>", unsafe_allow_html=True)
st.markdown("<style>.block-container {padding: 2rem; text-align: center;}</style>", unsafe_allow_html=True)
st.markdown(
    """<style>
    .stButton > button {
        background-color: #28a745;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #218838;
    }
    </style>""",
    unsafe_allow_html=True
)

# Parallel Input and Output Windows with Centered Layout
col1, mid_col, col2 = st.columns([2, 1, 2], gap="medium")

with col1:
    src_lang = st.selectbox("Source Language", options=["Detect language"] + list(languages.keys()), index=0)
    text_input = st.text_area("Enter text:", "", height=300, placeholder="Type your text here...")
    document = st.file_uploader("Upload a document (txt or pdf):", type=["txt", "pdf"])

with mid_col:
    st.markdown("<div style='text-align: center; padding-top: 90px;'>", unsafe_allow_html=True)
    translate_button = st.button("⇄ Translate", key="translate_button")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    tgt_lang = st.selectbox("Target Language", options=list(languages.keys()), index=1)
    translated_text = ""

# Trigger translation for text input or document
if translate_button:
    if text_input.strip():
        if src_lang == "Detect language":
            src_lang_key = None
        else:
            src_lang_key = languages[src_lang]

        tgt_lang_key = languages[tgt_lang]
        translated_text = translate_text(text_input, src_lang_key, tgt_lang_key)

    elif document:
        if src_lang == "Detect language":
            src_lang_key = None
        else:
            src_lang_key = languages[src_lang]

        tgt_lang_key = languages[tgt_lang]

        if document.type == "text/plain":
            translated_text = translate_document(document, src_lang_key, tgt_lang_key)
        elif document.type == "application/pdf":
            translated_text = translate_pdf(document, src_lang_key, tgt_lang_key)

if translated_text:
    col2.text_area("Translation:", translated_text, height=300)

# Footer Section
st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>Built with ❤️ by TranzlateX Team</div>",
    unsafe_allow_html=True
)

