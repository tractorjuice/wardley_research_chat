import os
import re
import openai
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.vectorstores import Pinecone
import pinecone

# Set OpenAI Model and API key
openai.api_key = st.secrets["OPENAI_API_KEY"]
#MODEL = "gpt-3"
#MODEL = "gpt-3.5-turbo"
MODEL = "gpt-3.5-turbo-0613"
#MODEL = "gpt-3.5-turbo-16k"
#MODEL = "gpt-3.5-turbo-16k-0613"
#MODEL = "gpt-4"
#MODEL = "gpt-4-0613"
#MODEL = "gpt-4-32k-0613"

st.set_page_config(page_title="Chat with Simon's Research Maps")
st.title("Chat with Simon's Research Maps")
st.sidebar.markdown("# Query all the maps using AI")
st.sidebar.divider()
st.sidebar.markdown("Developed by Mark Craddock](https://twitter.com/mcraddock)", unsafe_allow_html=True)
st.sidebar.markdown("Current Version: 0.0.1")
st.sidebar.divider()
st.sidebar.markdown("Using GPT-3 API")
st.sidebar.markdown("Not optimised")
st.sidebar.markdown("May run out of OpenAI credits")
st.sidebar.divider()
st.sidebar.markdown("Wardley Mapping is provided courtesy of Simon Wardley and licensed Creative Commons Attribution Share-Alike.")

# initialize pinecone
pinecone.init(
    api_key=st.secrets["PINECONE_API_KEY"],  # find at app.pinecone.io
    environment=st.secrets["PINECONE_ENV"]  # next to api key in console
    )

index_name = st.secrets["PINECONE_INDEX_NAME"] # Put your Pincecone index name here
name_space = st.secrets["PINECONE_NAME_SPACE"] # Put your Pincecone namespace here

embeddings = OpenAIEmbeddings()
vector_store = Pinecone.from_existing_index(index_name, embeddings, namespace=name_space)

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

system_template="""
    You are SimonGPT a strategy researcher based in the UK.
    “Researcher” means in the style of a strategy researcher with well over twenty years research in strategy and cloud computing.
    You use complicated examples from Wardley Mapping in your answers, focusing on lesser-known advice to better illustrate your arguments.
    Your language should be for an 12 year old to understand.
    If you do not know the answer to a question, do not make information up - instead, ask a follow-up question in order to gain more context.
    Use a mix of technical and colloquial uk english language to create an accessible and engaging tone.
    Only use the following context to answer the users question.
    ----------
    {summaries}
    """
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)

chain_type_kwargs = {"prompt": prompt}
llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k-0613", temperature=0, max_tokens=256)  # Modify model_name if you have access to GPT-4
chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] in ["user", "assistant"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if query := st.chat_input("What question do you have for the book?"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
      
    with st.spinner():
        with st.chat_message("assistant"):
            response = chain(query)
            st.markdown(response['answer'])
            for document in response['source_documents']:
                if 'source' in document.metadata:
                    st.markdown("\nSource: ", document.metadata['source'],"\n")
                    #st.markdown(document.page_content)
        st.session_state.messages.append({"role": "assistant", "content": response['answer']})
