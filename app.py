import truststore

truststore.inject_into_ssl()

import streamlit as st
from transformers import pipeline
from deep_translator import GoogleTranslator
from langdetect import detect
from gtts import gTTS
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
import tempfile
import os

# ----------------------------------
# Page Config
# ----------------------------------

st.set_page_config(
    page_title="Multilingual AI Voice Assistant",
    page_icon="🌍",
    layout="wide"
)

# ----------------------------------
# Hugging Face Model
# ----------------------------------

@st.cache_resource
def load_model():

    model = pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )

    return model

chatbot = load_model()

# ----------------------------------
# Speech To Text
# ----------------------------------

def speech_to_text(audio_file):

    recognizer = sr.Recognizer()

    try:

        with sr.AudioFile(audio_file) as source:

            audio_data = recognizer.record(source)

            text = recognizer.recognize_google(
                audio_data
            )

            return text

    except Exception as e:

        return f"Error: {e}"

# ----------------------------------
# AI Response
# ----------------------------------

def generate_response(text):

    result = chatbot(
        text,
        max_new_tokens=100
    )

    return result[0]["generated_text"]

# ----------------------------------
# Text To Speech
# ----------------------------------

def text_to_speech(text, lang):

    tts = gTTS(
        text=text,
        lang=lang
    )

    file_path = "reply.mp3"

    tts.save(file_path)

    return file_path

# ----------------------------------
# UI
# ----------------------------------

st.title("🌍 Multilingual AI Voice Assistant")

st.markdown("""
Speak in:

✅ English  
✅ Tamil  
✅ Hindi  

The assistant will understand,
translate,
generate AI response,
and speak back.
""")

# ----------------------------------
# Session State
# ----------------------------------

if "history" not in st.session_state:

    st.session_state.history = []

# ----------------------------------
# Voice Input
# ----------------------------------

st.subheader("🎤 Record Voice")

audio_bytes = audio_recorder(
    pause_threshold=2.0
)

if audio_bytes:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as temp_audio:

        temp_audio.write(audio_bytes)

        temp_path = temp_audio.name

    st.audio(audio_bytes)

    user_text = speech_to_text(temp_path)

    os.remove(temp_path)

    st.success(
        f"Recognized Text: {user_text}"
    )

    try:

        # Detect language

        detected_lang = detect(user_text)

        st.info(
            f"Detected Language: {detected_lang}"
        )

        # Translate To English

        english_text = GoogleTranslator(
            source='auto',
            target='en'
        ).translate(user_text)

        st.write(
            f"Translated: {english_text}"
        )

        # AI Response

        ai_response = generate_response(
            english_text
        )

        # Translate Back

        translated_response = GoogleTranslator(
            source='en',
            target=detected_lang
        ).translate(ai_response)

        st.subheader(
            "🤖 Assistant Response"
        )

        st.write(
            translated_response
        )

        # Store Chat

        st.session_state.history.append(
            (
                user_text,
                translated_response
            )
        )

        # Voice Output

        mp3_file = text_to_speech(
            translated_response,
            detected_lang
        )

        audio_file = open(
            mp3_file,
            "rb"
        )

        audio_bytes = audio_file.read()

        st.audio(
            audio_bytes,
            format="audio/mp3"
        )

        audio_file.close()

        os.remove(mp3_file)

    except Exception as e:

        st.error(e)

# ----------------------------------
# Text Chat Option
# ----------------------------------

st.subheader("💬 Text Chat")

user_input = st.text_input(
    "Ask anything"
)

if st.button("Send"):

    if user_input:

        detected_lang = detect(
            user_input
        )

        english_text = GoogleTranslator(
            source="auto",
            target="en"
        ).translate(
            user_input
        )

        ai_response = generate_response(
            english_text
        )

        translated_response = GoogleTranslator(
            source="en",
            target=detected_lang
        ).translate(
            ai_response
        )

        st.write(
            "🤖",
            translated_response
        )

        mp3_file = text_to_speech(
            translated_response,
            detected_lang
        )

        st.audio(
            mp3_file
        )

        st.session_state.history.append(
            (
                user_input,
                translated_response
            )
        )

# ----------------------------------
# Chat History
# ----------------------------------

st.subheader("📜 Conversation History")

for q, a in st.session_state.history[::-1]:

    st.markdown(
        f"""
        **👤 User:** {q}

        **🤖 Assistant:** {a}
        """
    )