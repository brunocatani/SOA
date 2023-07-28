from langchain.llms.base import LLM
from langchain import PromptTemplate,  LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import HuggingFacePipeline
import os 
from aideploy.model import personalpipe as pipeline
from aideploy.nexusllm import NexusLLM
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit.components.v1 import html
from PIL import Image
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth


template = PromptTemplate(
        #É enviado para a IA nesse template onde o "topic" é o prompt que a gente escreveu la em cima. A geração de texto começa depois de "<|assistant|>"
        template="<|prompter|>{topic}<|endoftext|><|assistant|>", 
        input_variables=["topic"])



# Pass hugging face pipeline to langchain class
llm = NexusLLM(pipeline=pipeline, verbose=True, callbacks=[StreamingStdOutCallbackHandler()])
# Build stacked LLM chain i.e. prompt-formatting + LLM
chain = LLMChain(llm=llm, prompt=template, verbose=True)


st.set_page_config(page_title="N.E.X.U.S.", layout="wide")



with open('./config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

#Autenticação de usuario armazenada no config.yaml

name, authentication_status, username = authenticator.login('Login', 'main')


# Response output
## Function for taking user prompt as input followed by producing AI generated responses
if authentication_status:

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        response = chain.run(topic=prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    with st.sidebar:
        
        authenticator.logout('Logout', 'main', key='unique_key')
    
elif authentication_status is False:
    st.error('Username/password is incorrect')
elif authentication_status is None:
    st.warning('Please enter your username and password')

with open('../config.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)