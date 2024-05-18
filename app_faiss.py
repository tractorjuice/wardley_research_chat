import os, re, uuid
from langchain_openai import OpenAI
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

#MODEL = "gpt-3.5-turbo" # 4K, Sept 2021. Legacy. Currently points to gpt-3.5-turbo-0613.
#MODEL = "gpt-3.5-turbo-1106" # 16K, Sept 2021. New Updated GPT 3.5 Turbo. The latest GPT-3.5 Turbo model with improved instruction following, JSON mode, reproducible outputs, parallel function calling, and more. Returns a maximum of 4,096 output tokens.
MODEL = "gpt-4o"

# Set API keys
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Get datastore
DATA_STORE_DIR = "."

st.set_page_config(page_title="Chat with Simon's Research Maps")
st.title("Chat with Simon's Research Maps")
st.sidebar.markdown("# Query all the maps using AI")
st.sidebar.divider()
st.sidebar.markdown("Developed by Mark Craddock](https://twitter.com/mcraddock)", unsafe_allow_html=True)
st.sidebar.markdown("Current Version: 1.1.0")
st.sidebar.divider()
st.sidebar.markdown(st.session_state.session_id)
st.sidebar.markdown("Wardley Mapping is provided courtesy of Simon Wardley and licensed Creative Commons Attribution Share-Alike.")
st.sidebar.divider()

# Set styling for buttons. Full column width, primary colour border.
primaryColor = st.get_option("theme.primaryColor")
custom_css_styling = f"""
<style>
    /* Style for buttons */
    div.stButton > button:first-child, div.stDownloadButton > button:first-child {{
        border: 5px solid {primaryColor};
        border-radius: 20px;
        width: 100%;
    }}
    /* Center align button container */
    div.stButton, div.stDownloadButton {{
        text-align: center;
    }}
    .stButton, .stDownloadButton {{
        width: 100%;
        padding: 0;
    }}
</style>
"""
st.html(custom_css_styling)


# Check if the user has provided an API key, otherwise default to the secret
user_openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", placeholder="sk-...", type="password")

if user_openai_api_key:
    os.environ["OPENAI_API_KEY"] = user_openai_api_key

    if "vector_store" not in st.session_state:
        # If the user has provided an API key, use it
        # Swap out openai for promptlayer

        if os.path.exists(DATA_STORE_DIR):
            st.session_state.vector_store = FAISS.load_local(
                DATA_STORE_DIR,
                OpenAIEmbeddings(),
                allow_dangerous_deserialization='True',
            )
        else:
            st.write(f"Missing files. Upload index.faiss and index.pkl files to {DATA_STORE_DIR} directory first")

        custom_system_template="""
            You are a strategy researcher with over twenty years of experience in strategy, Wardley Mapping, and cloud computing. You provide answers that incorporate examples from Wardley Mapping.
            Use a mix of technical and colloquial UK English to create an accessible and engaging tone suitable for a 12-year-old to understand.
            If you do not know the answer to a question, do not make up information. Instead, ask a follow-up question to gain more context.
            Your primary objective is to help the user formulate excellent answers by utilizing context about Wardley Maps and relevant details from your knowledge, along with insights from previous conversations.
            ----------------
            Reference Context and Knowledge: {context}
            Previous Conversations: {chat_history}
        """

        custom_user_template = "Question:'''{question}'''"

        prompt_messages = [
            SystemMessagePromptTemplate.from_template(custom_system_template),
            HumanMessagePromptTemplate.from_template(custom_user_template)
            ]
        prompt = ChatPromptTemplate.from_messages(prompt_messages)

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, output_key='answer')

    if "llm" not in st.session_state:
        st.session_state.llm = ChatOpenAI(
            model_name=MODEL,
            temperature=0,
            max_tokens=256,
            tags=["chatbot_research2023", st.session_state.session_id],
        )  # Modify model_name if you have access to GPT-4

    if "chain" not in st.session_state:
        st.session_state.chain = ConversationalRetrievalChain.from_llm(
            llm=st.session_state.llm,
            retriever=st.session_state.vector_store.as_retriever(
                search_kwargs={
                    "k": 5,
                    #"score_threshold": .95,
                    }
                ),
            chain_type="stuff",
            rephrase_question = True,
            return_source_documents=True,
            memory=st.session_state.memory,
            combine_docs_chain_kwargs={'prompt': prompt}
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
                response = st.session_state.chain(query)
                st.markdown(response['answer'])
                with st.expander("Source"):
                    source_documents = response['source_documents']
                    for index, document in enumerate(source_documents):
                        if 'source' in document.metadata:
                            source_details = document.metadata['source']
                            st.write(f"Source {index + 1}:", source_details[source_details.find('/maps'):],"\n")
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})
else:
    st.warning("Please enter your OpenAI API key", icon="⚠️")
