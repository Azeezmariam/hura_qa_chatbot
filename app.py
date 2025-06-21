from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_community.llms import CTransformers
from langchain.schema.output_parser import StrOutputParser
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
import os
import zipfile
import shutil
import sqlite3
import logging
import json
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

def load_data(path):
    """Load JSON data files"""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {path}: {e}")
        return []

def create_vector_store(embeddings):
    """Create a new vector store from source data"""
    # Load source data
    tripadvisor_data = load_data("data/tripadvisor_forum.json")
    faq_data = load_data("data/tourism_faq_gov.json")
    blog_data = load_data("data/local_blog_etiquette.json")
    combined_data = tripadvisor_data + faq_data + blog_data

    # Create documents
    documents = []
    for idx, item in enumerate(combined_data):
        doc_text = f"QUESTION: {item['question']}\nANSWER: {item['answer']}"
        source = "tripadvisor" if idx < len(tripadvisor_data) else "gov_faq" if idx < len(tripadvisor_data)+len(faq_data) else "blog"
        metadata = {"source": source, "original_question": item['question']}
        documents.append({"page_content": doc_text, "metadata": metadata})
    
    # Create new vector store
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="vector_db",
        collection_name="kigali_tourism"
    )
    vector_store.persist()
    return vector_store

def fix_chromadb_schema(db_path):
    """Fix ChromaDB schema compatibility issues"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if topic column exists
        cursor.execute("PRAGMA table_info(collections)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'topic' not in columns:
            logger.info("Fixing ChromaDB schema...")
            # Add missing columns
            cursor.execute("ALTER TABLE collections ADD COLUMN topic TEXT")
            cursor.execute("ALTER TABLE collections ADD COLUMN dimensionality INTEGER")
            conn.commit()
        
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Schema fix failed: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    # Optimize memory usage
    gc.collect()
    gc.freeze()

    # 1. Create/update embedding model
    embedding_model_path = "models/embedding_model"
    if os.path.exists(embedding_model_path):
        logger.info("Removing old embedding model...")
        shutil.rmtree(embedding_model_path, ignore_errors=True)
    
    logger.info("Creating new embedding model...")
    os.makedirs(embedding_model_path, exist_ok=True)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model.save(embedding_model_path)
    
    # 2. Load models
    app.state.vector_store, app.state.rag_chain = load_models()

def load_models():
    """Load all required models and components"""
    # 1. Load embedding model
    embeddings = HuggingFaceEmbeddings(model_name="models/embedding_model")
    
    # 2. Handle vector store
    vector_db_path = "vector_db"
    vector_db_zip = "vector_db.zip"
    
    # Remove existing vector_db if it exists
    if os.path.exists(vector_db_path):
        logger.info("Removing old vector database...")
        shutil.rmtree(vector_db_path, ignore_errors=True)
    
    # Extract from zip if available
    if os.path.exists(vector_db_zip):
        logger.info("Extracting vector database...")
        with zipfile.ZipFile(vector_db_zip, 'r') as zip_ref:
            zip_ref.extractall(vector_db_path)
        
        # Fix ChromaDB schema if needed
        db_file = os.path.join(vector_db_path, "chroma.sqlite3")
        if os.path.exists(db_file):
            fix_chromadb_schema(db_file)
    
    # Create new vector store if still doesn't exist
    if not os.path.exists(vector_db_path) or not os.listdir(vector_db_path):
        logger.info("Creating new vector database...")
        vector_store = create_vector_store(embeddings)
    else:
        logger.info("Loading existing vector database...")
        vector_store = Chroma(
            persist_directory=vector_db_path,
            embedding_function=embeddings,
            collection_name="kigali_tourism"
        )
    
    # 3. Download and load LLM directly from Hugging Face Hub
    logger.info("Downloading Mistral-7B model...")
    model_path = hf_hub_download(
        repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        revision="main",
        resume_download=True,  # Enable resuming interrupted downloads
        local_files_only=False
    )
    
    logger.info("Initializing Mistral-7B model...")
    llm = CTransformers(
        model=model_path,
        model_type="mistral",
        config={'max_new_tokens': 150, 'temperature': 0.3, 'gpu_layers': 0}
    )
    
    # 4. Create prompt template directly (avoid pickling issues)
    template = """<|im_start|>system
You are a tourism assistant for Kigali, Rwanda. Answer based ONLY on this context.
If answer isn't in context, say: "I couldn't find official info, contact tourism@rdb.rw".
Respond concisely in 1-3 sentences.<|im_end|>

Context:
{context}

<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
    prompt = ChatPromptTemplate.from_template(template)
    
    # 5. Build RAG chain
    def expand_query(input: str):
        location_keywords = ["how to get", "transport", "where is"]
        if any(kw in input.lower() for kw in location_keywords):
            return input + " in Kigali, Rwanda"
        return input

    rag_chain = (
        {"context": vector_store.as_retriever(search_kwargs={"k": 3}), 
         "question": RunnablePassthrough() | RunnableLambda(expand_query)}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return vector_store, rag_chain

class Query(BaseModel):
    text: str

@app.post("/ask")
async def answer_query(query: Query):
    response = app.state.rag_chain.invoke(query.text)
    return {"response": response}

@app.get("/")
def read_root():
    return {"message": "Welcome to Hura Q&A API!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)