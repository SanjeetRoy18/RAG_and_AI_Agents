import torch
import os
import tempfile
import gradio as gr
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from scipy.io import wavfile
import numpy as np
from transformers import pipeline   # For Speech-to-Text

#######------------- LLM Initialization-------------#######

os.environ["GROQ_API_KEY"] = "Your_API_Key_Here" 

parameters = {
    "temperature": 0.5,
    "max_tokens": 2048,
    "top_p": 1.0,
}

model_id = 'openai/gpt-oss-120b'

groq_llm = ChatGroq(
    model=model_id,
    **parameters # unpacks temperature, max_tokens, etc. directly into the client
)


#######------------- Prompt Template and Chain-------------#######

# Define the prompt template
template = """
Generate concise meeting minutes and a simple list of tasks based on the provided context. Follow the structure below using regular bullet points and sentences.

CRITICAL RULE: Do not make up, invent, or hallucinate any names, assignees, dates, or deadlines. Only include an assignee or deadline if it is explicitly stated in the context below. If no name or date is provided, omit them entirely or write "[Assignee: Not Specified]".

Meeting Minutes:
- [Insert key point or decision here as a clean bullet point]

Task List:
1. [Assignee: Name, Deadline: Date] - [Actionable task description]

Context:
{context}
"""

prompt = ChatPromptTemplate.from_template(template)

# Define the chain
chain = (
    {"context": RunnablePassthrough()}
    | prompt
    | groq_llm
    | StrOutputParser()
)


#######------------- Speech2text and Pipeline-------------#######

# Initialize the speech recognition pipeline
pipe = pipeline(
	"automatic-speech-recognition",
    model="openai/whisper-tiny.en",
    chunk_length_s=30,
	)

# Speech-to-text pipeline
def transcript_audio(audio_data):
    if audio_data is None:
        return "No audio file provided.", None

    # Gradio type="numpy" returns a tuple: (sampling_rate, numpy_array)
    samplerate, data = audio_data

    # Convert data to float32 and normalize it to prevent the PyTorch tensor error
    if data.dtype != np.float32:
        # Normalize based on integer type max boundaries
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483647.0
        else:
            data = data.astype(np.float32)

    # Convert stereo to mono if necessary
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Transcribe using the raw data dictionary (bypassing FFmpeg file reading)
    try:
        result = pipe({"sampling_rate": samplerate, "raw": data}, batch_size=8)["text"]
    except Exception as e:
        return f"Transcription Error: {str(e)}", None

    if not result.strip():
        return "Audio could not be transcribed into text.", None

    # Pass transcript to LangChain/Groq LLM
    try:
        meeting_analysis = chain.invoke(result)
    except Exception as e:
        return f"LLM Generation Error: {str(e)}", None

    # Generate a temporary file for the gr.File download output
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, "meeting_minutes.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(meeting_analysis)

    return meeting_analysis, file_path


#######------------- Gradio Interface-------------#######

audio_input = gr.Audio(sources="upload", type="numpy", label="Upload your audio file")
output_text = gr.Textbox(label="Meeting Minutes and Tasks")
download_file = gr.File(label="Download the Generated Meeting Minutes and Tasks")

iface = gr.Interface(
    fn=transcript_audio,
    inputs=audio_input,
    outputs=[output_text, download_file],
    title="AI Meeting Assistant",
    description="Upload an audio file of a meeting. This tool will transcribe the audio, fix product-related terminology, and generate meeting minutes along with a list of tasks."
)

iface.launch(server_name="127.0.0.1", server_port=5000)