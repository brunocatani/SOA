import transformers
import torch
import speech_recognition as sr
import pyttsx3
import gradio as gr
from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain, ConversationChain
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from shared.aideploy.model import personalpipe as pipeline


# Pass hugging face pipeline to langchain class
llms = HuggingFacePipeline(pipeline=pipeline) 
# Build stacked LLM chain i.e. prompt-formatting + LLM

template = PromptTemplate(template="<|prompter|>{input}<|endoftext|><|assistant|>{history}", input_variables=["input","history"])


nexuschatchain = ConversationChain(
    llm = llms,
    prompt=template,
    verbose=True,
    memory=ConversationBufferWindowMemory(k=2),
)

engine = pyttsx3.init()

r = sr.Recognizer()

with sr.Microphone() as source:
        print("Calibrating...")
        r.adjust_for_ambient_noise(source, duration=5)

def listen():
    with sr.Microphone() as source:
        # optional parameters to adjust microphone sensitivity
        # r.energy_threshold = 200
        # r.pause_threshold=0.5

        print("Okay, go!")
        while 1:
            text = ""
            print("listening now...")
            try:
                audio = r.listen(source, timeout=5, phrase_time_limit=30)
                print("Recognizing...")
                # https://github.com/openai/whisper#available-models-and-languages
                # tiny.en model = 74mb GPU Memory
                text = r.recognize_whisper(
                    audio,
                    model="tiny.en",
                    show_dict=True,
                )["text"]
            except Exception as e:
                unrecognized_speech_text = (
                    f"Sorry, I didn't catch that. Exception was: {e}s"
                )
                text = unrecognized_speech_text
            #print(text)

            response_text = nexuschatchain.predict(input=text)
            print(response_text)
            #engine.say(response_text)
            engine.runAndWait()
            return response_text



with gr.Blocks() as demo:
    with gr.Row():
         gr.Markdown("N.E.X.U.S.")

    output = gr.Textbox(label="Resposta", lines=6)
    btnn = gr.Button(label="Enviar")

    btnn.click(fn=listen, outputs=[output])
    #print(listen(""))

demo.launch()
