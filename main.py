from dotenv import load_dotenv
import pickle
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

import os

load_dotenv()

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
    corpus = st.file_uploader("Upload your PDF here", type='pdf')
    # st.write(corpus.name)
    if corpus is not None:
        pdf_corpus = PdfReader(corpus)
        # st.write(pdf_corpus)
        corpus_text = get_text(pdf_corpus)
        # st.write(get_text(pdf_corpus))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        corpus_chunks = text_splitter.split_text(text=corpus_text)
        # st.write(corpus_chunks)
        name_extr = corpus.name[:-4]

        if os.path.exists(f"{name_extr}.pkl"):
            with open(f"{name_extr}.pkl", "rb") as f:
                vectorstore = pickle.load(f)
            # st.write("Embeddings loaded from the disk")
        else:
            corpus_embeddings = OpenAIEmbeddings(model_name="ada")
            vectorstore = FAISS.from_texts(corpus_chunks, embedding=corpus_embeddings)
            with open(f"{name_extr}.pkl", "wb") as f:
                pickle.dump(vectorstore, f)
        query = st.text_input("Ask anything related to your submission!")
        # st.write(query)
        if query:
            result = vectorstore.similarity_search(query=query, k=3)
            # st.write(result)
            llm = OpenAI(temperature=0, model_name="text-davinci-003")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=result, question=query)
            print(cb)
            st.write(response)


if __name__ == '__main__':
    main()


