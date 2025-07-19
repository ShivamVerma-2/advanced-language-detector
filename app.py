
import streamlit as st
import pickle
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import base64
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import queue
import speech_recognition as sr
import io

model = pickle.load(open('language_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def get_text_stats(text):
    words = text.split()
    return len(words), len(text), np.mean([len(word) for word in words]) if words else 0, Counter(words).most_common(5)
from textblob import TextBlob


def get_sentiment(text):
    try:
        from textblob import TextBlob
        polarity = TextBlob(text).sentiment.polarity
        return ('Positive' if polarity > 0 else 'Negative' if polarity < 0 else 'Neutral'), polarity
    except:
        return 'Sentiment Unavailable', 0

st.set_page_config(page_title="Language Detector ", page_icon="🪐", layout="centered")


if "clicked" not in st.session_state:
    st.session_state.clicked = False


button_color = "#ff0080ff" if not st.session_state.clicked else "#000000"
button_hover_color = "#f9f196" if not st.session_state.clicked else "#333333"
button_text_color = "white" if not st.session_state.clicked else "white"

# st.title(" Language Detector and Analyzer")
st.markdown(f"""
<style>
.stApp {{
background: linear-gradient(135deg, #ff0080, #7928ca, #2afadf, #00ff99);
background-size: 600% 600%;
animation: gradient 20s ease infinite;
}}
.h1{{
text-align: center;
color: black;
font-weight: bold;
font-size: 36px;
                        
}}
            .p{{
            color: black;
            font-size: 20px;

        }}
@keyframes gradient {{
0%{{background-position:0% 50%}}
50%{{background-position:100% 50%}}
100%{{background-position:0% 50%}}
}}
.stButton button {{
background-color: {button_color};
color: {button_text_color};
 border: 3px solid  #0A2BB2;
border-radius: 8px;
padding: 10px 20px;
font-weight: bold;
font-size: 18px;
transition: background-color 0.1s;
}}
.stButton button:hover {{
background: {button_hover_color};
color: black;
}}
textarea {{
background: rgba(255,255,255,0.95);
color: black;
border: 2px solid #00ff99;
border-radius: 10px;
font-size: 18px;
}}
.output-text {{
color: black !important;
font-size: 18px;
}}
.sidebar-img {{
display: block;
margin-left: auto;
margin-right: auto;
border-radius: 50%;
width: 150px;
height: 150px;
object-fit: cover;
border: 3px grey solid;
}}
.social-logo {{
display: block;
margin-left: auto;
margin-right: auto;
width: 40px;
height: 40px;
margin-top: 15px;
transition: transform 0.3s;
}}
.social-logo:hover {{
transform: scale(1.2);
}}
</style>
""", unsafe_allow_html=True)
st.set_page_config(page_title="Advanced Language Detector | Shivam Verma", page_icon="🪐", layout="centered")


st.markdown("""
<h1 class="output-text" style="text-align: center; margin-top: 0;">
🙏 Echoes of Earth 🙏
</h1>
""", unsafe_allow_html=True)

# Circular autoplaying video
video_file = "rev.mp4"
with open(video_file, "rb") as file:
    video_bytes = file.read()
    encoded = base64.b64encode(video_bytes).decode()

st.markdown(f"""
<div style="display: flex; justify-content: center; align-items: center; padding: 20px;">
    <video autoplay loop muted playsinline
        style="border-radius: 50%; object-fit: cover; border: 3px solid  #0A2BB2;"
        width="200" height="200">
        <source src="data:video/mp4;base64,{encoded}" type="video/mp4">
    </video>
</div>
""", unsafe_allow_html=True)


st.markdown('<p class="output-text"style="color:black;font-size:19px;font-type:bold">Paste, type, or speak your text below to detect language, analyze statistics, view prediction graphs, and get sentiment insights.</p>', unsafe_allow_html=True)

st.markdown('<p style="font-size:19px; color:black; font-weight:500; text-align: center"> Enter your text: ⬇️</p>', unsafe_allow_html=True)
user_input = st.text_area("", height=200)

#  Speech-to-Text Section
st.markdown('<p class="output-text">🎤 Or record your voice to transcribe and analyze:</p>', unsafe_allow_html=True)

audio_queue = queue.Queue()

RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

def audio_frame_callback(frame):
    audio = frame.to_ndarray()
    audio_queue.put(audio)
    return frame

webrtc_ctx = webrtc_streamer(
    key="speech_to_text",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration=RTC_CONFIGURATION,
    audio_receiver_size=1024,
    media_stream_constraints={"audio": True, "video": False},
    audio_frame_callback=audio_frame_callback,
)

if webrtc_ctx.state.playing:
    st.markdown("<p class='output-text'>🎙️ Listening... Click the mic a" \
    "gain to stop recording.</p>", unsafe_allow_html=True)

    try:
        import whisper
        import tempfile
        import numpy as np
        import soundfile as sf

        audio_frames = []
        while not audio_queue.empty():
            audio_frames.append(audio_queue.get())

        if audio_frames:
            audio_np = np.concatenate(audio_frames, axis=0)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f, audio_np, 48000)
                model_whisper = whisper.load_model("base")
                result = model_whisper.transcribe(f.name)
                st.session_state.transcribed_text = result["text"]
                st.experimental_rerun()

    except Exception as e:
        st.error(f"Whisper transcription error: {e}")
st.markdown("""
<div style='text-align: center;'>
""", unsafe_allow_html=True)

if st.button("🔍 Detect & Analyze"):
    # your logic



    st.session_state.clicked = True 
    if user_input.strip() == "":
        st.error("⚠️ Please enter or record some text.")
    else:
        X = vectorizer.transform([user_input])
        pred = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        class_labels = model.classes_

        st.markdown(f'<p class="output-text">🌐 <b>Detected Language:</b> {pred}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="output-text"><b>Confidence:</b> {np.max(probs)*100:.2f}%</p>', unsafe_allow_html=True)

        st.markdown('<h3 class="output-text">📈 Top Predictions</h3>', unsafe_allow_html=True)
        top_indices = np.argsort(probs)[::-1][:5]
        for idx in top_indices:
            st.markdown(f'<p class="output-text">{class_labels[idx]}: {probs[idx]*100:.2f}%</p>', unsafe_allow_html=True)

        st.markdown('<h3 class="output-text">📊 Prediction Graph</h3>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(
            [class_labels[i] for i in top_indices][::-1],
            [probs[i]*100 for i in top_indices][::-1],
            color="#024711"
        )
        ax.set_xlabel('Probability (%)')
        ax.set_title('Top 5 Predicted Languages')
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        for i, v in enumerate([probs[i]*100 for i in top_indices][::-1]):
            ax.text(v + 1, i, f"{v:.2f}%", color='black', va='center', fontweight='bold')
        st.pyplot(fig)

        st.markdown('<h3 class="output-text">📊 Text Stats</h3>', unsafe_allow_html=True)
        w, c, awl, mc = get_text_stats(user_input)
        st.markdown(f'<p class="output-text">Words: {w}, Characters: {c}, Avg Word Length: {awl:.2f}</p>', unsafe_allow_html=True)
        st.markdown('<p class="output-text">Most Common Words:</p>', unsafe_allow_html=True)
        for word, freq in mc:
            st.markdown(f'<p class="output-text">{word}: {freq}</p>', unsafe_allow_html=True)

        st.markdown('<h3 class="output-text">💡 Sentiment</h3>', unsafe_allow_html=True)
        sentiment, score = get_sentiment(user_input)
        st.markdown(f'<p class="output-text">Sentiment: {sentiment} ({score:.2f})</p>', unsafe_allow_html=True)

with st.sidebar:
    st.title("👨‍💻 About")
    st.info("""
**Project:** Advanced Language Detector  
**Student:** Shivam Verma  
**Stack:** Python, Streamlit, ML, Matplotlib, SpeechRecognition  
**Goal:** ML App Deployment with Data Visualization
    """)

    st.markdown('<img src="https://tse4.mm.bing.net/th/id/OIP.Y7SK42gnMD0GUovX0Zyl2AHaHa?r=0&w=626&h=626&rs=1&pid=ImgDetMain&o=7&rm=3" class="sidebar-img">', unsafe_allow_html=True)

    st.markdown("""
    <a href="https://github.com/ShivamVerma-2" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" class="social-logo">
    </a>
    """, unsafe_allow_html=True)

    st.markdown("""
    <a href="https://linkedin.com/in/shivam-verma-6a8b0631a" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" class="social-logo">
    </a>
    """, unsafe_allow_html=True)

# st.markdown("<h3 style='text-align: right; color: black; font-size: 20px; margin-top: 30%'></h3>", unsafe_allow_html=True)

st.markdown("""
<hr>
<div style='text-align: center; color: solid dark black ; font-weight:500 ;font-size: 16px;'>
© 2025 Shivam Verma | All rights reserved.<br>
This project is for educational and personal portfolio purposes only.
</div>
""", unsafe_allow_html=True)