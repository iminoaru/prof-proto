import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader, UnstructuredPPTXLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

# Initialize Hugging Face Hub
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# Initialize embeddings
embeddings = HuggingFaceHubEmbeddings()

# Initialize HuggingFace model
llm = HuggingFaceHub(repo_id="impira/layoutlm-document-qa", model_kwargs={"temperature": 0.7, "max_length": 512})

def process_document(file, file_type):
    if file_type == "pdf":
        loader = PyPDFLoader(file)
    elif file_type == "pptx":
        loader = UnstructuredPPTXLoader(file)
    else:
        raise ValueError("Unsupported file type")
    
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts

def create_vectorstore(texts):
    return FAISS.from_documents(texts, embeddings)

def get_retrieval_qa(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Be short and precise

    {context}

    Question: {question}
    Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)

def main():
    st.title("Cloud-based GenAI RAG App")

    # File upload
    pdf_file = st.file_uploader("Upload a PDF file", type="pdf")
    ppt_file = st.file_uploader("Upload a PowerPoint file", type="pptx")

    if pdf_file and ppt_file:
        # Process documents
        pdf_texts = process_document(pdf_file.name, "pdf")
        ppt_texts = process_document(ppt_file.name, "pptx")

        # Combine texts
        all_texts = pdf_texts + ppt_texts

        # Create vectorstore
        vectorstore = create_vectorstore(all_texts)

        # Create retrieval QA chain
        qa_chain = get_retrieval_qa(vectorstore)

        # User input
        user_question = st.text_input("Ask a question about the uploaded documents:")

        if user_question:
            # Get answer
            answer = qa_chain.run(user_question)
            st.write("Answer:", answer)

if __name__ == "__main__":
    main()
