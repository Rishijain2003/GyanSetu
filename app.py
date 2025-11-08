import os
import streamlit as st
import time 
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser



DB_FAISS_PATH = 'faiss_hindi_index'
MODEL_NAME = 'hiiamsid/sentence_similarity_hindi' 

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Gyan Setu", 
    page_icon="🕉️", 
    layout="centered"
)



@st.cache_resource
def load_embeddings_model():
    """Loads and caches the Sentence Transformer model for performance."""
    st.write("Loading Hindi Embedding Model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'},  
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings

@st.cache_resource
def load_faiss_store(_embeddings_model):
    """Loads and caches the FAISS vector store."""
    if not os.path.exists(DB_FAISS_PATH):
        st.error(f"Error: Vector database folder '{DB_FAISS_PATH}' not found.")
        st.info("Please run the 'build_db.py' script first to create the index.")
        st.stop()
    
    st.write(f"Loading Hindi vector database from {DB_FAISS_PATH}...")
    # Use the renamed argument in the function body
    vectorstore = FAISS.load_local(
        DB_FAISS_PATH, 
        _embeddings_model, 
        allow_dangerous_deserialization=True 
    )
    return vectorstore

def format_docs(docs):
    """Formats the retrieved document chunks for the LLM prompt."""
    return "\n\n".join(doc.page_content for doc in docs)



def create_rag_chain(vectorstore):
    """Creates and returns the LangChain RAG pipeline."""
    # Initialize the OpenAI LLM (Ensure API key is set)
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.1
    )

    # Retriever instance (k=20 as specified)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

    # Prompt Template (CRITICAL for guiding the LLM in Hindi)
    template = """आप एक मददगार AI सहायक हैं जो केवल निम्नलिखित संदर्भ के आधार पर हिंदी में उत्तर देता है।
    यदि उत्तर संदर्भ में नहीं है, तो विनम्रता से कहें कि आपको जानकारी नहीं मिल पाई।

    Context: {context}
    Question: {question}
    
    Hindi Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # RAG Chain using LCEL
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain



def main():
    """Main function to run the Streamlit app."""
    st.title("🕉️ ज्ञान सेतु  (Gyan Setu) ")
    st.markdown("##### Bhagavad Gita RAG")
    
    # Sidebar for debugging
    st.sidebar.header("Debugging Tools")
    debug_mode = st.sidebar.checkbox("Show Retrieved Chunks (Debug Mode)", value=False)
    
    # Check for API Key
    if "OPENAI_API_KEY" not in os.environ and not st.secrets.get("OPENAI_API_KEY"):
        st.warning("OpenAI API Key not found!")
        st.info("Please set the OPENAI_API_KEY environment variable or add it to Streamlit Secrets.")
        st.stop()
    
    # Load resources
    try:
        embeddings = load_embeddings_model()
        # Note: We pass the 'embeddings' object to the cached function
        vectorstore = load_faiss_store(embeddings) 
        rag_chain = create_rag_chain(vectorstore)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    except Exception as e:
        st.error(f"Failed to initialize resources: {e}")
        st.stop()

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("अपना प्रश्न हिंदी में पूछें (Ask your question in Hindi)..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process and display AI response
        with st.chat_message("assistant"):
            with st.spinner("Searching for answer..."):
                try:
                    # 1. Retrieve documents (only needed if debug is on)
                    retrieved_docs = retriever.invoke(prompt)

                    if debug_mode:
                        st.subheader("Retrieved Chunks:")
                        for i, doc in enumerate(retrieved_docs):
                            st.text_area(f"Chunk {i+1} (Source: Page {doc.metadata.get('page')})", doc.page_content, height=150)
                        
                        # Use the retrieved documents to form the context
                        context = format_docs(retrieved_docs)
                        
                        # Generate the response using the chain
                        response = rag_chain.invoke(prompt)

                    else:
                        # Standard RAG chain execution
                        response = rag_chain.invoke(prompt)
                    
                    # Simulate streaming for a nice effect
                    message_placeholder = st.empty()
                    full_response = ""
                    for chunk in response.split():
                        full_response += chunk + " "
                        time.sleep(0.02)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                
                except Exception as e:
                    response = f"Sorry, an error occurred while contacting the LLM: {e}"
                    st.error(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()