import streamlit as st
import requests

st.title("🎤 Stress Detection using AI")

uploaded_file = st.file_uploader("Upload an Audio File", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    with st.spinner("🔍 Analyzing..."):
        audio_bytes = uploaded_file.read()
        response = requests.post("http://localhost:8000/predict/", files={"file": audio_bytes})

        try:
            if response.status_code == 200:
                emotion = response.json().get("emotion", "Unknown")
                st.success(f"Predicted Emotion: **{emotion}**")
            else:
                st.error(f"❌ Error: {response.text}")
        except Exception as e:
            st.error(f"❌ Error processing response: {e}")
