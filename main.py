from dotenv import load_dotenv
import pickle
import streamlit as st
from streamlit_extras.add_vertical_space import  add_vertical_space
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pinecone
from langchain.vectorstores import Pinecone



load_dotenv()
pinecone.init(	api_key= os.getenv("PINECONE_API_KEY") ,
	environment=os.getenv("PINECONE_ENV"))

with st.sidebar:
    st.title('New user corpus chat appliction')
    st.write('Made by Aman Singh')

    add_vertical_space(3)
    st.markdown('''
    ## About this project:
    this application is a LLM-powered made with:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models)  LLM Model,embeder and text generator
    
    ''')



def get_text(pdf_corpus):
  """Extracts the text from a PDF corpus."""
  text = ""
  for page in pdf_corpus.pages:
    text += page.extract_text()
  return text


def main():
    st.header("💬 Chat with your own 💻  Corpus")
    corpus = st.file_uploader("Upload your PDF here", type = 'pdf')
    #st.write(corpus.name)
    if corpus is not None:
        pdf_corpus = PdfReader(corpus)
        #st.write(pdf_corpus)
        corpus_text = get_text(pdf_corpus)
        #st.write(get_text(pdf_corpus))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,length_function=len )
        corpus_chunks = text_splitter.split_text(text = corpus_text)
        #st.write(corpus_chunks)
        corpus_embeddings = OpenAIEmbeddings(model_name="ada")
        copy = corpus_embeddings.copy()
        index_name = 'langchain-project'
        vectorstore = Pinecone.from_texts(corpus_chunks,embedding = copy, index_name=index_name)
        vectorstore.embedding = copy
        name_extr = corpus.name[:-4]
        with open(f"{name_extr}.pkl", "wb") as f:
            pickle.dump(vectorstore, f)



if __name__ == '__main__':
    main()
