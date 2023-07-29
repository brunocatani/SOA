from langchain.llms.base import LLM
from langchain import PromptTemplate,  LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os 
from aideploy.model import personalpipe as pipeline
from aideploy.nexusllm import NexusLLM
import streamlit as st
from streamlit_chat import message


template = PromptTemplate(
        #É enviado para a IA nesse template onde o "topic" é o prompt que a gente escreveu la em cima. A geração de texto começa depois de "<|assistant|>"
        template="<|prompter|>{topic}<|endoftext|><|assistant|>", 
        input_variables=["topic"])



# Pass hugging face pipeline to langchain class
llm = NexusLLM(pipeline=pipeline, verbose=True, callbacks=[StreamingStdOutCallbackHandler()])
# Build stacked LLM chain i.e. prompt-formatting + LLM
chain = LLMChain(llm=llm, prompt=template, verbose=True)



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
