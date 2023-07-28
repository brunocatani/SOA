from langchain.llms.base import LLM
from langchain import PromptTemplate,  LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from aideploy.model import modeltemplate
from aideploy.nexusclass import NexusClass, prompthelper
import streamlit as st
from streamlit_chat import message
from PIL import Image
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from llama_index import SimpleDirectoryReader, LLMPredictor, ServiceContext, ListIndex, query_engine
from aideploy.model import personalpipe as pipeline
from aideploy.nexusllm import NexusLLM




def create_index():
    llm = LLMPredictor(NexusLLM(pipeline=pipeline, verbose=True, callbacks=[StreamingStdOutCallbackHandler()]))
    service_context=ServiceContext.from_defaults(
        llm_predictor=llm,
        prompt_helper=prompthelper
    )
    docs = SimpleDirectoryReader("/home/ubuntu/llamaindex/data").load_data()
    index = ListIndex.from_documents(docs, service_context=service_context)
    return index


st.set_page_config(page_title="N.E.X.U.S.", layout="wide")



if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

print("Hola")
index = create_index()

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    query_engine=index.as_query_engine()
    response = query_engine.query(modeltemplate(prompt), response_mode="no_text")

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})