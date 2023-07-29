from langchain.llms.base import LLM
from langchain import PromptTemplate,  LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import PromptTemplate, ConversationChain
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
import streamlit as st
from streamlit_chat import message
from aideploy.model import personalpipe as pipeline
from aideploy.nexusllm import NexusLLM



template = PromptTemplate(template="<|prompter|>{input}<|endoftext|><|assistant|>{history}", input_variables=["input","history"])

# Pass hugging face pipeline to langchain class
llm = NexusLLM(pipeline=pipeline, verbose=True, callbacks=[StreamingStdOutCallbackHandler()]) 
# Build stacked LLM chain i.e. prompt-formatting + LLM
nexuschatchain = ConversationChain(
    llm = llm,
    prompt=template,
    verbose=True,
    memory=ConversationBufferWindowMemory(k=2),
)

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
    response = nexuschatchain.predict(input=prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

