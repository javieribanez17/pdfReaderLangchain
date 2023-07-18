#Libraries
#Enviroment variables
from dotenv import load_dotenv
import os
#GUI with streamlit
import streamlit as st
#Pdf file
from PyPDF2 import PdfReader
#Text analysis
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

#Principal function
def main():
    load_dotenv()
    #print(os.getenv("OPEN_AI_KEY"))
    #GUI configuration
    st.set_page_config(page_title="PDF GPT extractor")
    st.header("¡Bienvenido! haz clic en el botón para comenzar 👇")
    #Work with pdf file
    #Upload file
    pdf = st.file_uploader("Carga tu PDF", type="pdf")
    #Read file
    if pdf is not None:
        pdfReader = PdfReader(pdf)
        text = ""
        for page in pdfReader.pages:
            text += page.extract_text()
        #Split text into chunks
        textSplitter = CharacterTextSplitter(
            separator="\n",
            chunk_size = 1000,
            chunk_overlap = 200, #Propiedad para tomar n caracteres antes del chunck anterior que es cortado en medio de una frase
            length_function = len
        )
        chunks = textSplitter.split_text(text)
        #st.write(chunks)
        #Create embeddings
        embeddings = OpenAIEmbeddings()
        #Create knowledge base for search
        knowledgeBase = FAISS.from_texts(chunks, embeddings)
        #User question for ask to model
        userQuestion = st.text_input("Haz una pregunta de tu PDF:")
        if userQuestion:
            docs = knowledgeBase.similarity_search(userQuestion)
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            #Model response and monitorind pricing
            with get_openai_callback() as cb:
                response = chain.run(input_documents = docs, question = userQuestion)
                print(cb)
            st.write(response)

#Call to principal function
if __name__ == '__main__':
    main()