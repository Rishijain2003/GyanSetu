import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()


# Ensure your API key is set in your environment variables:
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# The name of the FAISS index folder
DB_FAISS_PATH = 'faiss_hindi_index'
# The embedding model used to create the index (MUST be the same)
MODEL_NAME = 'hiiamsid/sentence_similarity_hindi' 

def format_docs(docs):
    """Formats the retrieved document chunks for the LLM prompt."""
    return "\n\n".join(doc.page_content for doc in docs)

def run_hindi_chatbot(hindi_query: str):
    """
    Loads the database, creates the RAG chain, and invokes it with a Hindi query.
    """
    try:
        
        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={'device': 'cpu'},  
            encode_kwargs={'normalize_embeddings': True}
        )
        
       
        print(f"Loading Hindi vector database from {DB_FAISS_PATH}...")
        vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)


        retriever = vectorstore.as_retriever(search_kwargs={"k": 8}) # Retrieve top 8 chunks

        # 4. Initialize the OpenAI LLM
        # Using a chat model (like gpt-4o or gpt-3.5-turbo) is best practice
        llm = ChatOpenAI(
            model="gpt-4o",  # Or "gpt-4o" for better Hindi fluency
            temperature=0.1        # Lower temperature for factual, less creative answers
        )

        # 5. Define the Prompt Template
        # This is CRITICAL for instructing the LLM to use the context and respond in HINDI.
        template = """आप एक मददगार AI सहायक हैं जो केवल निम्नलिखित संदर्भ के आधार पर हिंदी में उत्तर देता है।
        यदि उत्तर संदर्भ में नहीं है, तो विनम्रता से कहें कि आपको जानकारी नहीं मिल पाई।

        Context: {context}
        Question: {question}
        
        Hindi Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        print(f"\nProcessing Hindi Query: '{hindi_query}'...")
        response = rag_chain.invoke(hindi_query)
        
        print("-" * 50)
        print("Chatbot Response:")
        print(response)
        print("-" * 50)

    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nPlease ensure your OPENAI_API_KEY environment variable is set.")

if __name__ == '__main__':
    # You must have run the build_db.py script previously to create 'faiss_index'
    # Example Hindi Query
    query = "अर्जुन का रथ कौन चला रहा था?" 
    
    run_hindi_chatbot(query)
    
    # Another example
    # query_2 = "युद्ध में विजय प्राप्त करने के लिए क्या आवश्यक है?"
    # run_hindi_chatbot(query_2)