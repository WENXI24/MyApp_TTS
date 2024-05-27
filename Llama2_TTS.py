import streamlit as st
import replicate
import os
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# App title
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot with TTS")

# Replicate Credentials
with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api) == 40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    os.environ['REPLICATE_API_TOKEN'] = replicate_api

    st.subheader('Models and parameters')
    selected_model = st.selectbox('Choose a Llama2 model', ['Llama2-7B', 'Llama2-13B'], key='selected_model')
    temperature = st.slider('Temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    top_p = st.slider('Top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.slider('Max_length', min_value=32, max_value=128, value=120, step=8)
    st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Initialize TTS model
@st.cache(allow_output_mutation=True)
def load_tts_model():
    config = XttsConfig()
    config.load_json("/path/to/xtts/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir="/path/to/xtts/", eval=True)
    model.cuda()
    return model, config

tts_model, tts_config = load_tts_model()

# Function for generating LLaMA2 response
def generate_llama2_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    
    model_version = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea' if selected_model == 'Llama2-7B' else 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
    
    output = replicate.run(
        model_version,
        input={
            "prompt": f"{string_dialogue} {prompt_input} Assistant: ",
            "temperature": temperature,
            "top_p": top_p,
            "max_length": max_length,
            "repetition_penalty": 1.0
        }
    )
    return output

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''.join(response)
            placeholder.markdown(full_response)
            
            # Generate TTS audio
            try:
                outputs = tts_model.synthesize(
                    full_response,
                    tts_config,
                    speaker_wav="/path/to/speaker.wav",  # Update this path as needed
                    gpt_cond_len=3,
                    language="en",
                )
                audio_path = outputs[0]
                audio_bytes = open(audio_path, "rb").read()
                st.audio(audio_bytes, format="audio/wav")
            except Exception as e:
                st.error(f"Error generating TTS audio: {e}")
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

