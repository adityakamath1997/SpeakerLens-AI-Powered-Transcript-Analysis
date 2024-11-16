# app.py
import os
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter,defaultdict
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from src.assemblyai_processing import get_audio_intelligence

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Audio Insights Hub",  # Title that appears on the browser tab
    page_icon="🎙️",  # Favicon
    layout="centered",  # Use wide layout
    initial_sidebar_state="expanded"
)

# Display App Title Bar Information
st.markdown(
    """
    <style>
        .title-bar {
            font-size: 16px;
            color: #d3d3d3;
            text-align: center;
            margin-top: -20px;
            padding-bottom: 10px;
        }
        .title-bar a {
            color: #92A8D1;
            text-decoration: none;
        }
        .title-bar a:hover {
            text-decoration: underline;
        }
    </style>
    <div class="title-bar">
        Made with ❤️ <a href="https://github.com/adityakamath1997" target="_blank">Aditya Kamath</a> | 
        <a href="https://github.com/adityakamath1997/Speech-Diarization-Project" target="_blank">GitHub Repository</a>
    </div>
    """,
    unsafe_allow_html=True
)


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

        st.header("📊 Visualizations")
        visualization_tabs = st.tabs([
            "Bar Chart: Total Words",
            "Bar Chart: Words per Speaker",
            "Word Cloud: Total Words",
            "Word Cloud: Words per Speaker"
        ])

        # Tab 1: Bar Chart of Total Words
        with visualization_tabs[0]:
            st.subheader("Bar Chart of Most Common Words (Total)")
            all_text = " ".join(data["text"] for data in speakers.values())
            overall_keywords = extract_keywords(all_text)
            most_common_keywords = overall_keywords.most_common(10)
            if most_common_keywords:
                keywords, counts = zip(*most_common_keywords)
                fig, ax = plt.subplots()
                ax.barh(keywords, counts, color="#88B04B")
                ax.set_xlabel("Frequency")
                ax.set_title("Most Common Words (Total)")
                st.pyplot(fig)
            else:
                st.write("No significant keywords found.")

        # Tab 2: Bar Chart of Words per Speaker
        with visualization_tabs[1]:
            st.subheader("Bar Chart of Most Common Words (Per Speaker)")
            for speaker, data in speakers.items():
                st.write(f"**{speaker}**")
                speaker_keywords = extract_keywords(data["text"])
                common_keywords = speaker_keywords.most_common(5)
                if common_keywords:
                    keywords, counts = zip(*common_keywords)
                    fig, ax = plt.subplots()
                    ax.barh(keywords, counts, color=SPEAKER_COLORS[list(speakers.keys()).index(speaker) % len(SPEAKER_COLORS)])
                    ax.set_xlabel("Frequency")
                    ax.set_title(f"Top Keywords for {speaker}")
                    st.pyplot(fig)
                else:
                    st.write("No significant keywords found for this speaker.")

        # Tab 3: Word Cloud of Total Words
        with visualization_tabs[2]:
            st.subheader("Word Cloud of Most Frequent Non-Common Words (Total)")
            wordcloud_text = " ".join([word for word in transcription.split() if word.lower() not in STOP_WORDS])
            wordcloud = WordCloud(width=800, height=400, background_color="black", colormap="Pastel1").generate(wordcloud_text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(plt)

        # Tab 4: Word Cloud of Words per Speaker
        with visualization_tabs[3]:
            st.subheader("Word Cloud of Most Frequent Non-Common Words (Per Speaker)")
            for speaker, data in speakers.items():
                st.write(f"**{speaker}**")
                speaker_wordcloud_text = " ".join([word for word in data["text"].split() if word.lower() not in STOP_WORDS])
                if speaker_wordcloud_text:
                    speaker_wordcloud = WordCloud(width=800, height=400, background_color=SPEAKER_COLORS[list(speakers.keys()).index(speaker) % len(SPEAKER_COLORS)], colormap="Pastel1").generate(speaker_wordcloud_text)
                    plt.figure(figsize=(10, 5))
                    plt.imshow(speaker_wordcloud, interpolation="bilinear")
                    plt.axis("off")
                    st.pyplot(plt)
                else:
                    st.write("No significant keywords for this speaker.")

        # Summary of the audio content
        st.header("📝 Summary")
        st.markdown(f"<div class='box'>{summary}</div>", unsafe_allow_html=True)

         # Entities Grouped by Category with Dropdowns
        st.header("🔍 Unique Entities Detected (Grouped by Category)")
        grouped_entities = defaultdict(list)
        for entity in entities:
            grouped_entities[entity["entity_type"]].append(entity["text"])
        
        for category, items in grouped_entities.items():
            with st.expander(f"{category.capitalize()} (Click to expand)"):
                unique_items = list(set(items))  # Remove duplicates
                st.write(", ".join(unique_items))

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
