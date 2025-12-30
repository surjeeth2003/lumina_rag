import streamlit as st
from dotenv import load_dotenv
from data_processing import get_pdf_text, analyze_medical_image
from rag_engine import get_text_chunks, get_vector_store, answer_query

load_dotenv()

st.set_page_config(page_title="Lumina: Medical RAG", layout="wide")

def main():
    st.header("Lumina 🩺 Multimodal Medical Assistant")

    with st.sidebar:
        st.title("Medical Data Ingestion")
        image_file = st.file_uploader("Upload Medical Image", type=["jpg", "png", "jpeg"])
        pdf_docs = st.file_uploader("Upload Medical PDFs", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            if not pdf_docs and not image_file:
                st.warning("Please upload at least one file.")
            else:
                with st.spinner("Processing Medical Data..."):
                    raw_text = ""
                    
                    if pdf_docs:
                        raw_text += get_pdf_text(pdf_docs)
                        st.success("PDF Text Extracted")
                        
                    if image_file:
                        st.info("Analyzing Medical Image...")
                        image_desc = analyze_medical_image(image_file)
                        st.write(f"**Image Analysis:** {image_desc}")
                        raw_text += f"\n\n[IMAGE ANALYSIS CONTEXT]: {image_desc}"
                        
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        # This will now use local CPU embeddings (No Rate Limit)
                        get_vector_store(text_chunks)
                        st.success("Indexing Complete! You can now chat.")
                    else:
                        st.error("No extractable data found.")

    user_question = st.text_input("Ask a question about the uploaded medical records:")

    if user_question:
        try:
            response = answer_query(user_question)
            st.write("Reply: ", response)
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()