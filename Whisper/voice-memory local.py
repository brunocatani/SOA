from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.llms import TextGen
import speech_recognition as sr
import pyttsx3
import gradio as gr
from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import os 
import torch
import gradio as gr

# Define Model ID
model_id = "OpenAssistant/stablelm-7b-sft-v7-epoch-3"
# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Load Model 
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir='/home/ubuntu/Projects/shared/models', 
    torch_dtype=torch.bfloat16, trust_remote_code=True, load_in_8bit=True, device_map="auto", offload_folder="offload")
# Set PT model to inference mode
model.eval()
# Build HF Transformers pipeline 
pipeline = transformers.pipeline(
    "text-generation", 
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    max_length=400,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)


# Pass hugging face pipeline to langchain class
llms = HuggingFacePipeline(pipeline=pipeline) 
# Build stacked LLM chain i.e. prompt-formatting + LLM

prompt_t = PromptTemplate(template="<|prompter|>{input}<|endoftext|><|assistant|>{history}", input_variables=["input","history"])



chatgpt_chain = ConversationChain(
    llm = llms,
    prompt=prompt_t,
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
                # whisper model options are found here: https://github.com/openai/whisper#available-models-and-languages
                # other speech recognition models are also available.
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

            response_text = chatgpt_chain.predict(input=text)
            print(response_text)
            #engine.say(response_text)
            engine.runAndWait()
            return response_text


def my_chatbot(input, history):
    history = history or []
    my_history = list(sum(history, ()))
    my_history.append(input)
    my_input = ' '.join(my_history)
    output = listen(my_input)
    history.append((input, output))
    return history, history 


with gr.Blocks() as demo:
    gr.Markdown("""<h1><center>My Chatbot</center></h1>""")
    chatbot = gr.Chatbot()
    state = gr.State()
    btnn = gr.Button(label="Enviar")
    btnn.click(fn=listen , outputs=[chatbot, state])

demo.launch()
