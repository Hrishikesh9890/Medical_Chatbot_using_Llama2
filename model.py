import streamlit as st
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import sentence_transformers


DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,input_variables=['context', 'question'])
    return prompt


def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,chain_type='stuff',retriever=db.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain


def load_llm():
    
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.2
    )
    return llm

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

# Final Result Function
def final_result(user_query):
    qa_result = qa_bot()

    truncated_query = user_query[:512]
    
    response = qa_result({'query': truncated_query})
    return response


# Streamlit UI
def main():
    st.title("Welcome to Llama2 Medical Q&A App! 🤖")

    user_query = st.text_input("Input: ")
    button_pressed = st.button("Ask Question")

    if button_pressed and user_query:
        response = final_result(user_query)
        st.write(f"med-Bot: {response['result']}")
        
        # Display sources if available
        sources = response.get('source_documents', [])
        if sources:

            st.write("Sources:")
            for source in sources:
                st.write(f"Source: {source}")

if __name__ == '__main__':
    main()