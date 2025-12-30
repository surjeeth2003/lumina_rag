import os
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Local Embeddings (Free & Fast)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from google.api_core import exceptions

load_dotenv()

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # Using local CPU model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def answer_query(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load DB safely
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""
    You are Lumina, a medical assistant. Answer the question based ONLY on the context below.
    
    Context:
    {context_text}
    
    Question: 
    {user_question}
    
    Detailed Answer (include a medical disclaimer):
    """
    
    model = ChatGoogleGenerativeAI(
        model="gemini-flash-latest", 
        temperature=0.3,
        api_key=os.getenv("GOOGLE_API_KEY") 
    )
    
    try:
        response = model.invoke(prompt)
        
        # --- NEW: CLEANER LOGIC ---
        # If the model gives us a list (JSON style), we extract just the text.
        content = response.content
        if isinstance(content, list):
            final_text = ""
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    final_text += block["text"]
            return final_text
        # If it's already a string, just return it
        return content
        
    except exceptions.ResourceExhausted:
        return "Error: System is busy. Please wait 30 seconds and try again."
    except Exception as e:
        return f"An error occurred: {str(e)}"