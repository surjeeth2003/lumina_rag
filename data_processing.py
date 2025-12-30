import pypdf
import os
import io  # <--- NEW IMPORT
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import base64
from google.api_core import exceptions

load_dotenv()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        # FIX: Explicitly read the stream into a BytesIO buffer
        # This prevents the "Stream Truncated" error on Streamlit Cloud
        bytes_stream = io.BytesIO(pdf.read())
        pdf_reader = pypdf.PdfReader(bytes_stream)
        
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def analyze_medical_image(image_file):
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest", 
        temperature=0.5,
        api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Convert uploaded file to bytes for processing
    image_bytes = image_file.getvalue()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{image_b64}"

    try:
        response = llm.invoke([
            HumanMessage(
                content=[
                    {"type": "text", "text": "Analyze this medical image. Describe the visual findings in detail. Transcribe any text found. Do not give a diagnosis."},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            )
        ])
        return response.content
    except Exception as e:
        return f"Error analyzing image: {str(e)}"