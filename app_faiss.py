import os
import re
import openai
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Set API keys
openai.api_key = st.secrets["OPENAI_API_KEY"]
#MODEL = "gpt-3"
#MODEL = "gpt-3.5-turbo"
#MODEL = "gpt-3.5-turbo-0613"
#MODEL = "gpt-3.5-turbo-16k"
MODEL = "gpt-3.5-turbo-16k-0613"
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
st.sidebar.markdown("Using gpt-3.5-turbo API")
st.sidebar.markdown("Not optimised")
st.sidebar.markdown("May run out of OpenAI credits")
st.sidebar.divider()
st.sidebar.markdown("Wardley Mapping is provided courtesy of Simon Wardley and licensed Creative Commons Attribution Share-Alike.")

# initialize FAISS
MAPS_DATASTORE = "datastore"

if os.path.exists(MAPS_DATASTORE):
    vector_store = FAISS.load_local(".", OpenAIEmbeddings())
else:
    st.write(f"Missing files. Upload index.faiss and index.pkl files to {DATA_STORE_DIR} directory first")

system_template="""
    You are SimonGPT with the style of a strategy researcher with well over twenty years research in strategy and cloud computing.
    You use complicated examples from Wardley Mapping in your answers.
    Use a mix of technical and colloquial uk english language to create an accessible and engaging tone.
    Your language should be for an 12 year old to understand.
    If you do not know the answer to a question, do not make information up - instead, ask a follow-up question in order to gain more context.
    Only use the following context to answer the question at the end.

    ----------
    {summaries}
    """
prompt_messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(prompt_messages)

chain_type_kwargs = {"prompt": prompt}
llm = ChatOpenAI(
    model_name=MODEL,
    temperature=0,
    max_tokens=1000
)  # Modify model_name if you have access to GPT-4

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

if query := st.chat_input("How is AI used in these maps?"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
      
    with st.spinner():
        with st.chat_message("assistant"):
            response = chain(query)
            st.markdown(response['answer'])
            source_documents = response['source_documents']
            for index, document in enumerate(source_documents):
                if 'source' in document.metadata:
                    source_details = document.metadata['source']
                    st.write(f"Source {index + 1}:", source_details[source_details.find('/maps'):],"\n")
        st.session_state.messages.append({"role": "assistant", "content": response['answer']})
