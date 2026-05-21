import os
import pickle
import random
import numpy as np
import streamlit as st
import pretty_midi
from scipy.io import wavfile
from music21 import note, chord, stream, instrument
from tensorflow.keras.models import load_model

MODEL_PATH = "models/music_model.keras"
NOTES_PATH = "models/notes.pkl"
OUTPUT_PATH = "generated/generated_music.mid"
WAV_OUTPUT_PATH = "generated/generated_music.wav"
SEQUENCE_LENGTH = 100

st.set_page_config(page_title="AI Music Generator", page_icon="🎵", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#050816,#13072e,#2b064d);
    color: white;
}

.block-container {
    max-width: 1250px;
    padding-top: 1rem;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.title {
    text-align: center;
    font-size: 62px;
    font-weight: 900;
    background: linear-gradient(90deg,#38bdf8,#8b5cf6,#ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    font-size: 22px;
    color: #eee;
    margin-bottom: 25px;
}

[data-testid="stImage"] img {
    border-radius: 22px;
    height: 220px;
    object-fit: cover;
    border: 1px solid rgba(255,255,255,0.18);
    box-shadow: 0 16px 45px rgba(0,0,0,0.35);
}

[data-testid="stImageCaption"] {
    color: #d8d5ff;
    font-weight: 700;
    text-align: center;
}

.section-box {
    background: rgba(20, 25, 65, 0.78);
    border: 1px solid rgba(130,100,255,0.35);
    border-radius: 22px;
    padding: 24px;
    box-shadow: 0 18px 45px rgba(0,0,0,0.25);
}

.stat-card {
    background: rgba(255,255,255,0.08);
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.10);
}

.stat-card h2 {
    font-size: 34px;
    margin: 0;
}

.stat-card p {
    color: #d8d5ff;
    margin-top: 8px;
}

.info-box {
    background: rgba(14,101,170,0.25);
    border: 1px solid rgba(56,189,248,0.35);
    padding: 15px;
    border-radius: 14px;
    color: #8bddff;
    margin: 18px 0;
}

.note-box {
    background: rgba(0,0,0,0.28);
    border: 1px solid rgba(255,255,255,0.1);
    padding: 18px;
    border-radius: 16px;
    color: #ff77d9;
    font-size: 15px;
    line-height: 1.8;
    max-height: 100px;
    overflow-y: auto;
}

.stButton button {
    width: 100%;
    border: none;
    border-radius: 14px;
    padding: 15px;
    font-size: 18px;
    font-weight: 900;
    color: white;
    background: linear-gradient(90deg,#22d3ee,#a855f7,#ec4899);
}

.stDownloadButton button {
    width: 100%;
    border: none;
    border-radius: 14px;
    padding: 14px;
    font-size: 16px;
    font-weight: 800;
    color: white;
    background: linear-gradient(90deg,#34d399,#3b82f6);
}

.footer-text {
    text-align: center;
    margin-top: 25px;
    color: #aaa4d9;
}
</style>
""", unsafe_allow_html=True)


def generate_notes(model, notes, length=300, temperature=1.0):
    pitchnames = sorted(set(notes))
    note_to_int = {n: i for i, n in enumerate(pitchnames)}
    int_to_note = {i: n for i, n in enumerate(pitchnames)}

    start = random.randint(0, len(notes) - SEQUENCE_LENGTH - 1)
    pattern = [note_to_int[n] for n in notes[start:start + SEQUENCE_LENGTH]]

    output = []
    progress_bar = st.progress(0)

    for i in range(length):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(len(pitchnames))

        prediction = model.predict(prediction_input, verbose=0)[0]
        prediction = np.log(prediction + 1e-9) / temperature
        prediction = np.exp(prediction) / np.sum(np.exp(prediction))

        index = np.random.choice(len(prediction), p=prediction)
        result = int_to_note[index]

        output.append(result)
        pattern.append(index)
        pattern = pattern[1:]

        progress_bar.progress((i + 1) / length)

    return output


def create_midi(prediction_output):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if "." in pattern or pattern.isdigit():
            chord_notes = []
            for current_note in pattern.split("."):
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                chord_notes.append(new_note)

            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)

        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp=OUTPUT_PATH)


def convert_midi_to_wav():
    midi_data = pretty_midi.PrettyMIDI(OUTPUT_PATH)
    audio_data = midi_data.synthesize(fs=44100)

    max_value = np.max(np.abs(audio_data))

    if max_value > 0:
        audio_data = np.int16(audio_data / max_value * 32767)
    else:
        audio_data = np.int16(audio_data)

    wavfile.write(WAV_OUTPUT_PATH, 44100, audio_data)


training_notes = 0
if os.path.exists(NOTES_PATH):
    with open(NOTES_PATH, "rb") as f:
        training_notes = len(pickle.load(f))


st.markdown('<div class="title">AI Music Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Generate original music using LSTM Deep Learning ✨</div>', unsafe_allow_html=True)

img1, img2 = st.columns(2)

with img1:
    st.image(
        "https://images.unsplash.com/photo-1511379938547-c1f69419868d?auto=format&fit=crop&w=1200&q=80",
        caption="AI learns musical patterns from MIDI files",
        use_container_width=True
    )

with img2:
    st.image(
        "https://images.unsplash.com/photo-1520523839897-bd0b52f945a0?auto=format&fit=crop&w=1200&q=80",
        caption="Generate new melodies with LSTM",
        use_container_width=True
    )

st.write("")

left, right = st.columns([0.9, 1.1], gap="large")

with left:
    with st.container():
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.subheader("🎚️ Music Settings")

        length = st.slider("Generated Notes Length", 100, 1000, 500, 50)
        temperature = st.slider("Creativity Level", 0.3, 2.0, 1.2, 0.1)

        st.markdown(
            '<div class="info-box">Low creativity = smoother music<br>High creativity = more experimental</div>',
            unsafe_allow_html=True
        )

        generate = st.button("✨ Generate Music")
        st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("📊 AI Music Statistics")

    s1, s2, s3 = st.columns(3)

    with s1:
        st.markdown(f'<div class="stat-card"><h2>🎵 {length}</h2><p>Generated Notes</p></div>', unsafe_allow_html=True)

    with s2:
        st.markdown(f'<div class="stat-card"><h2>✨ {temperature}</h2><p>Creativity Level</p></div>', unsafe_allow_html=True)

    with s3:
        st.markdown(f'<div class="stat-card"><h2>💿 {training_notes}</h2><p>Training Notes Learned</p></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("🎧 Generated Music")

    if os.path.exists(WAV_OUTPUT_PATH):
        st.audio(WAV_OUTPUT_PATH)
    else:
        st.info("Generate music first to listen here.")

    d1, d2 = st.columns(2)

    with d1:
        if os.path.exists(OUTPUT_PATH):
            with open(OUTPUT_PATH, "rb") as midi_file:
                st.download_button("⬇ Download MIDI File", midi_file, "generated_music.mid", "audio/midi")

    with d2:
        if os.path.exists(WAV_OUTPUT_PATH):
            with open(WAV_OUTPUT_PATH, "rb") as wav_file:
                st.download_button("⬇ Download WAV Audio", wav_file, "generated_music.wav", "audio/wav")

    st.markdown('</div>', unsafe_allow_html=True)


if generate:
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Run train_model.py first.")
    elif not os.path.exists(NOTES_PATH):
        st.error("Notes file not found. Run train_model.py first.")
    else:
        with st.spinner("Generating AI music... 🎼"):
            model = load_model(MODEL_PATH)

            with open(NOTES_PATH, "rb") as f:
                notes = pickle.load(f)

            prediction_output = generate_notes(model, notes, length, temperature)

            create_midi(prediction_output)
            convert_midi_to_wav()

        st.success("Music generated successfully. اضغط Generate مرة تانية أو اعمل Refresh لو الصوت ما ظهرش فورًا.")

        st.subheader("🎼 Generated Notes Preview")
        preview_notes = "  |  ".join(prediction_output[:70])
        st.markdown(f'<div class="note-box">{preview_notes}</div>', unsafe_allow_html=True)


st.markdown('<div class="footer-text">❤️ Built with TensorFlow, Streamlit, Music21 & PrettyMIDI</div>', unsafe_allow_html=True)