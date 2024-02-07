import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdftext(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_textchunks(text):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000,
        length_function=len
    )
    chunks=text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore=FAISS.from_texts(texts=text_chunks, embedding=embedding)
    vectorstore.save_local("fiass_index")
    # return vectorstore


def get_conversationchain():
      prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
      model=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)
      prompt=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
      chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)   
      return chain

def user_input(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore=FAISS.load_local("fiass_index",embeddings)
    docs=vectorstore.similarity_search(user_question)
    chain=get_conversationchain()
    answer=chain({"input_documents":docs,"question":user_question},
    return_only_outputs=True)
    print(answer)
    st.write("reply",answer["output_text"])
    
   
          
# create a streamlit app for above functions
def main():
    st.set_page_config("chat pdf")
    st.header("chat pdf using gemini")
    user_question=st.text_input("enter your question")
    if user_question:
        user_input(user_question)

    with st.sidebar:

         st.title("menue")
         pdf_docs=st.file_uploader("upload your pdf",type="pdf",accept_multiple_files=True)
         if st.button("submit & process"):
            with st.spinner("processing..."):
                raw_text = get_pdftext(pdf_docs)
                questions = get_textchunks(raw_text)
                get_vectorstore(questions)
                st.success("processed")
                st.balloons()



if __name__ == "__main__":
    main()
