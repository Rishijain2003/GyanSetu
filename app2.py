import os
import streamlit as st
import time 
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Updated paths
DB_FAISS_PATH = "faiss_hindi_index_e5"
MODEL_NAME = "intfloat/multilingual-e5-base"

st.set_page_config(page_title="Gyan Setu v2", page_icon="🕉️", layout="centered")

@st.cache_resource
def load_embeddings_model():
    st.write(f"🔤 Loading multilingual embedding model: {MODEL_NAME}")
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def load_faiss_store(embeddings):
    if not os.path.exists(DB_FAISS_PATH):
        st.error(f"FAISS database not found: {DB_FAISS_PATH}")
        st.stop()
    st.write("📂 Loading FAISS vector database...")
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

    template = """आप एक मददगार AI सहायक हैं जो केवल निम्नलिखित संदर्भ के आधार पर हिंदी में उत्तर देता है।
    यदि उत्तर संदर्भ में नहीं है, तो विनम्रता से कहें कि आपको जानकारी नहीं मिल पाई।

    Context: {context}
    Question: {question}

    Hindi Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def main():
    st.title("🕉️ ज्ञान सेतु v2 (Multilingual Gita RAG)")
    st.markdown("##### Powered by `intfloat/multilingual-e5-base` embeddings")

    if "OPENAI_API_KEY" not in os.environ:
        st.warning("Missing OpenAI API key. Please set it before running.")
        st.stop()

    embeddings = load_embeddings_model()
    vectorstore = load_faiss_store(embeddings)
    rag_chain = create_rag_chain(vectorstore)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("अपना प्रश्न हिंदी में पूछें..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = rag_chain.invoke(prompt)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
