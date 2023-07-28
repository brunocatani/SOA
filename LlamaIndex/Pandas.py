from langchain.llms.base import LLM
from langchain import PromptTemplate,  LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from aideploy.model import modeltemplate
from aideploy.nexusclass import prompthelper
from aideploy.model import personalpipe as pipeline
from aideploy.nexusllm import NexusLLM
import streamlit as st
from streamlit_chat import message
from PIL import Image
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from llama_index import SimpleDirectoryReader, LLMPredictor, ServiceContext, ListIndex, query_engine, GPTVectorStoreIndex
import pandas as pd
from llama_index.query_engine import PandasQueryEngine
import logging
import sys



upfile = st.file_uploader("Load your files", type=["csv"])

llm = LLMPredictor(NexusLLM(pipeline=pipeline, verbose=True, callbacks=[StreamingStdOutCallbackHandler()]))
service_context=ServiceContext.from_defaults(
    llm_predictor=llm,
    prompt_helper=prompthelper
    )

index = GPTVectorStoreIndex.from_documents(upfile, service_context=service_context)

df = pd.read_csv(upfile)

query_engine=index.as_query_engine()

query_engine = PandasQueryEngine(df=df, verbose=True)

response = query_engine.query()

if upfile is not None:
    query = st.text_area("Ask Away")
    button = st.button('Manda chefe')
    if button:
        st.markdown(query_engine.query(query))