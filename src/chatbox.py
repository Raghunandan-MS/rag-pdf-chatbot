import streamlit as st
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# --- Page Config ---
st.set_page_config(page_title="CoreRAG Local AI", layout="centered")
st.title("📄 RAG Application on PDF")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# --- Sidebar: PDF Upload ---
with st.sidebar:
    st.header("Setup")
    uploaded_file = st.file_uploader("Upload a PDF to begin", type="pdf")
    
    if uploaded_file:
        with st.spinner("Processing PDF with PyMuPDF..."):
            # Save temp file
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # PDF Loader
            loader = PyMuPDFLoader("temp.pdf")
            data = loader.load()
            
            # Chunking logic
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = text_splitter.split_documents(data)
            
            # Local Embeddings (Ollama)
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            
            # Create/Update Vector Store
            st.session_state.vector_db = Chroma.from_documents(
                documents=chunks, 
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
            st.success("PDF Knowledge Base Ready!")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about the document...", disabled=not st.session_state.vector_db):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                llm = ChatOllama(model="phi", temperature=0, num_ctx=2048)
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.vector_db.as_retriever(search_kwargs={"k": 1})
                )
                
                result = qa_chain.invoke(prompt)
                response = result["result"]
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"Error during retrieval: {e}")

if not uploaded_file:
    st.info("Please upload a PDF in the sidebar to start chatting.")