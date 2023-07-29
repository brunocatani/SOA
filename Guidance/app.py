
import guidance
from server.tools import load_tools
from server.agent import CustomAgentGuidance
from server.model import personal_model, personal_tokenizer
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit.components.v1 import html
import streamlit_authenticator as stauth
from PIL import Image
import yaml
from yaml.loader import SafeLoader
from aideploy.localapi import TGWebUI
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, pipeline, TextIteratorStreamer


import os
os.environ["SERPER_API_KEY"] = ''



examples = [
    ["How much is the salary of number 8 of Manchester United?"],
    ["What is the population of Congo?"],
    ["Where was the first president of South Korean born?"],
    ["What is the population of the country that won World Cup 2022?"]    
]

def greet(name):
    final_answer = custom_agent(name)
    return final_answer['fn']

streamer = TextStreamer(tokenizer=personal_tokenizer)

nexus = guidance.llms.transformers(model=personal_model, tokenizer=personal_tokenizer, stream=True)
guidance.llm = nexus
dict_tools = load_tools()

custom_agent = CustomAgentGuidance(guidance, dict_tools)

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
        response = greet(prompt)
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
