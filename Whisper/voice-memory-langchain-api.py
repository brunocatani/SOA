from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.llms import TextGen
import speech_recognition as sr
import pyttsx3
import gradio as gr

model_url = "http://localhost:5000"

llms = TextGen(model_url=model_url)

prompt_t = PromptTemplate(template="<|prompter|>{input}<|endoftext|><|assistant|>{history}", input_variables=["input","history"])



chatgpt_chain = ConversationChain(
    llm = llms,
    prompt=prompt_t,
    verbose=True,
    memory=ConversationBufferWindowMemory(k=2),
)

engine = pyttsx3.init()


def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Calibrating...")
        r.adjust_for_ambient_noise(source, duration=5)
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
            print(text)

            response_text = chatgpt_chain.predict(input=text)
            print(response_text)
            #engine.say(response_text)
            #engine.runAndWait()



with gr.Blocks() as demo:
    gr.Markdown("N.E.X.U.S.")
    name = gr.Textbox()

    output = gr.Textbox(Label="Resposta", lines=6)
    btnn = gr.Button(label="Enviar")

    btnn.click(fn=listen, inputs=[name], outputs=[output])

demo.launch()