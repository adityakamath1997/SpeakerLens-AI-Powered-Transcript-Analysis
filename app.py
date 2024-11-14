# app.py
import os
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords
from src.assemblyai_processing import get_audio_intelligence

# Using NLTK's stop words in this case. This could  be replaced with acustom list of stop words if desired.
nltk.download("stopwords")
STOP_WORDS = set(stopwords.words("english"))

# Defining a set of colors to differentiate speakers visually, this could be randomized. Custom colors have been chsoen for now
SPEAKER_COLORS = [
    "#FF6F61", "#6B5B95", "#88B04B", "#F7CAC9", "#92A8D1", "#955251", "#B565A7", "#009B77"
]

# The default duration is in milliseconds, hence format durations from milliseconds to a "minutes:seconds" or "hours:minutes:seconds" style
def format_duration(milliseconds):
    seconds = milliseconds / 1000
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    hours = minutes // 60
    minutes = minutes % 60
    
    if hours > 0:
        return f"{hours}h {minutes}m {remaining_seconds}s"
    else:
        return f"{minutes}m {remaining_seconds}s"

# Extracts keywords from text, filtering out common stop words
def extract_keywords(text):
    words = text.lower().split()
    keywords = [word for word in words if word.isalpha() and word not in STOP_WORDS]
    return Counter(keywords)

# Applying a dark themed background
st.markdown(
    """
    <style>
    .stApp { background-color: #2E2E2E; color: #ffffff; }
    .box {
        background-color: #3E3E3E;
        border: 1px solid #555555;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 8px;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
        color: #ffffff;
    }
    h1, h2, h3, h4, .st-upload-area {
        color: #ffffff;
    }
    .legend-item {
        display: inline-block;
        margin-right: 10px;
        padding: 3px 8px;
        border-radius: 5px;
        color: white;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set up the main user interface
st.title("🎙️ Enhanced Audio Analysis with Full Feature Set")
st.write("Upload an audio file (.mp3) for transcription, speaker identification, and analysis with color-coded sections for each speaker.")

# File upload section:
uploaded_file = st.file_uploader("Choose an audio file...", type=["mp3"])

if uploaded_file is not None:
    raw_audio_path = os.path.join("data", "raw", uploaded_file.name)
    with open(raw_audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File '{uploaded_file.name}' uploaded successfully!")

    st.header("📝 Full Transcription and Speaker-Specific Highlights")
    try:
        # Retrieve transcription and audio data
        transcription, speakers, summary, entities, sentiment_analysis, topics, content_safety = get_audio_intelligence(raw_audio_path)

        # Display the raw transcription in a single color block
        st.subheader("Transcription (Original Order, No Color)")
        st.markdown(f"<div class='box'>{transcription}</div>", unsafe_allow_html=True)

        # Assign each speaker a color for the visualizations
        speaker_colors = {}
        color_index = 0
        for speaker in speakers.keys():
            speaker_colors[speaker] = SPEAKER_COLORS[color_index % len(SPEAKER_COLORS)]
            color_index += 1

        # Create a legend to help differentiate between speakers
        st.subheader("Speaker Legend")
        legend_html = ""
        for speaker, color in speaker_colors.items():
            legend_html += f"<div class='legend-item' style='background-color: {color};'>{speaker}</div>"
        st.markdown(legend_html, unsafe_allow_html=True)

        # Show each speakers text segment with assigned color
        st.header("🔊 Speaker-Specific Transcription")
        for speaker, data in speakers.items():
            color = speaker_colors[speaker]
            duration_formatted = format_duration(data["duration"])
            st.subheader(f"{speaker} - {duration_formatted}")
            st.markdown(
                f"<div class='box' style='background-color: {color}; color: #ffffff;'>{data['text']}</div>",
                unsafe_allow_html=True
            )

        # Speaker duration chart for visual comparison of speaking times
        st.subheader("🔊 Speaking Duration per Speaker")
        speaker_names = list(speakers.keys())
        durations = [data["duration"] / 1000 for data in speakers.values()]  # Convert ms to seconds
        fig, ax = plt.subplots()
        ax.barh(speaker_names, durations, color=[speaker_colors[name] for name in speaker_names])
        ax.set_xlabel("Duration (seconds)")
        ax.set_title("Speaking Duration per Speaker")
        st.pyplot(fig)

        # Keyword Analysis for overall transcript and speaker specific keywords
        st.header("🗣️ Keyword Analysis")
        
        # Keywords across all speakers
        all_text = " ".join(data["text"] for data in speakers.values())
        overall_keywords = extract_keywords(all_text)
        most_common_keywords = overall_keywords.most_common(10)

        # Show bar chart of common keywords
        st.subheader("Most Common Keywords (Overall)")
        keywords, counts = zip(*most_common_keywords)
        fig, ax = plt.subplots()
        ax.barh(keywords, counts, color="#88B04B")
        ax.set_xlabel("Frequency")
        st.pyplot(fig)

        # Individual keyword breakdown per speaker
        st.subheader("Speaker-Specific Keywords")
        for speaker, data in speakers.items():
            st.write(f"**{speaker}**")
            speaker_keywords = extract_keywords(data["text"])
            common_keywords = speaker_keywords.most_common(5)

            if common_keywords:
                keywords, counts = zip(*common_keywords)
                fig, ax = plt.subplots()
                ax.barh(keywords, counts, color=speaker_colors[speaker])
                ax.set_xlabel("Frequency")
                ax.set_title(f"Top Keywords for {speaker}")
                st.pyplot(fig)
            else:
                st.write("No significant keywords found.")

        # Summary of the audio content
        st.header("📝 Summary")
        st.markdown(f"<div class='box'>{summary}</div>", unsafe_allow_html=True)

        # Display entities detected in the audio(WIP, I intend to make this more concise)
        st.header("🔍 Unique Entities Detected")
        unique_entities = {entity["text"]: entity["entity_type"] for entity in entities}
        if unique_entities:
            for entity, entity_type in unique_entities.items():
                st.markdown(f"<div class='box'>Entity: <b>{entity_type}</b>, Value: <b>{entity}</b></div>", unsafe_allow_html=True)
        else:
            st.write("No unique entities detected.")

        # Summary of sentiment analysis(WIP)
        st.header("💬 Sentiment Analysis Overview")
        sentiments_summary = {}
        for sentiment in sentiment_analysis:
            sentiment_type = sentiment["sentiment"]
            sentiments_summary[sentiment_type] = sentiments_summary.get(sentiment_type, 0) + 1
        st.markdown(
            "<div class='box'>" +
            ", ".join(f"<b>{key}:</b> {value} occurrences" for key, value in sentiments_summary.items()) +
            "</div>",
            unsafe_allow_html=True
        )

        # Filtered topic detection display
        st.header("📚 Relevant Topic Detection")
        CONFIDENCE_THRESHOLD = 0.5  #This threshold can be changed as per requirements
        if topics:
            filtered_topics = {topic: confidence for topic, confidence in topics.items() if confidence > CONFIDENCE_THRESHOLD}
            if filtered_topics:
                for topic, confidence in filtered_topics.items():
                    st.markdown(f"<div class='box'>Topic: <b>{topic}</b>, Confidence: {confidence:.2f}</div>", unsafe_allow_html=True)
            else:
                st.write("No highly relevant topics detected.")
        else:
            st.write("No topics detected.")

        # Content safety information
        st.header("⚠️ Content Safety")
        if content_safety:
            for label, confidence in content_safety.items():
                st.markdown(f"<div class='box'>Label: <b>{label}</b>, Confidence: {confidence:.2f}</div>", unsafe_allow_html=True)
        else:
            st.write("No safety concerns detected.")

    except Exception as e:
        st.error(f"Error during audio analysis: {e}")
