import os
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter,defaultdict
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from src.assemblyai_processing import get_audio_intelligence

# Basic configuration for the Streamlit application interface
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

# Standard English stop words used for text analysis
nltk.download("stopwords")
STOP_WORDS = set(stopwords.words("english"))

# Predefined color scheme for speaker differentiation
SPEAKER_COLORS = [
    "#FF6F61", "#6B5B95", "#88B04B", "#F7CAC9", "#92A8D1", "#955251", "#B565A7", "#009B77"
]

# Converts millisecond duration into readable time format
def format_duration(milliseconds):
    """
    Converts duration from milliseconds to a readable time format.
    Returns formatted string in hours, minutes, and seconds.
    """
    seconds = milliseconds / 1000
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    hours = minutes // 60
    minutes = minutes % 60
    
    if hours > 0:
        return f"{hours}h {minutes}m {remaining_seconds}s"
    else:
        return f"{minutes}m {remaining_seconds}s"

def extract_keywords(text):
    """
    Analyzes text to find significant keywords.
    Removes common words and returns frequency count of remaining terms.
    """
    words = text.lower().split()
    keywords = [word for word in words if word.isalpha() and word not in STOP_WORDS]
    return Counter(keywords)

# Custom dark theme styling for better readability
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
        # Retrieves analysis results from audio processing
        transcription, speakers, summary, entities, sentiment_analysis, topics, content_safety, transcript_data = get_audio_intelligence(raw_audio_path)

        # Confidence threshold for topic relevance filtering, this can be changed as per requirements
        CONFIDENCE_THRESHOLD = 0.5

        # Create tabs for transcription display
        transcript_tabs = st.tabs([
            "Line by Line Transcript",
            "Full Transcript"
        ])

        # Tab 1: Line by Line Transcript
        with transcript_tabs[0]:
            st.subheader("Line by Line Transcript")
            for utterance in transcript_data["utterances"]:
                speaker_index = list(speakers.keys()).index(utterance["speaker"]) % len(SPEAKER_COLORS)
                color = SPEAKER_COLORS[speaker_index]
                st.markdown(
                    f"""<div class='box' style='background-color: {color}; padding: 8px; margin-bottom: 5px;'>
                    <strong>{utterance["speaker"]}</strong>: {utterance["text"]}
                    </div>""",
                    unsafe_allow_html=True
                )

        # Tab 2: Full Transcript
        with transcript_tabs[1]:
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

        # Advanced Business Insights Visualizations
        st.header("📈 Advanced Analytics & Business Insights")
        insights_tabs = st.tabs([
            "Speaker Engagement Metrics",
            "Conversation Flow Analysis",
            "Topic Distribution",
            "Sentiment Timeline"
        ])

        # Tab 1: Speaker Engagement Metrics - Shows who talks the most and how much each person contributes
        with insights_tabs[0]:
            st.subheader("Speaker Engagement Analysis")
            
            # Calculate speaking time distribution - Shows what percentage of the conversation each person takes up
            total_duration = sum(data["duration"] for data in speakers.values())
            speaking_times = {speaker: (data["duration"] / total_duration) * 100 
                            for speaker, data in speakers.items()}
            
            # Create pie chart for speaking time distribution - Visual breakdown of talking time
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.pie(speaking_times.values(), 
                   labels=[f"{speaker}\n({percentage:.1f}%)" 
                          for speaker, percentage in speaking_times.items()],
                   colors=[speaker_colors[speaker] for speaker in speaking_times.keys()],
                   autopct='%1.1f%%')
            plt.title("Speaking Time Distribution")
            st.pyplot(fig)
            
            # Calculate average words per turn - Shows how much each person says when they speak
            avg_utterance_lengths = {}
            for speaker in speakers:
                speaker_utterances = [u for u in transcript_data["utterances"] 
                                    if u["speaker"] == speaker]
                avg_length = sum(len(u["text"].split()) for u in speaker_utterances) / len(speaker_utterances)
                avg_utterance_lengths[speaker] = avg_length
            
            # Bar chart for average utterance length
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(avg_utterance_lengths.keys(), 
                         avg_utterance_lengths.values(),
                         color=[speaker_colors[speaker] for speaker in avg_utterance_lengths.keys()])
            ax.set_title("Average Words per Turn")
            ax.set_ylabel("Number of Words")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # Tab 2: Conversation Flow Analysis - Shows how the conversation moves between speakers over time
        with insights_tabs[1]:
            st.subheader("Conversation Flow Patterns")
            
            # Timeline visualization - Shows when each person speaks throughout the meeting
            fig, ax = plt.subplots(figsize=(12, 6))
            speakers_list = list(speakers.keys())
            for i, utterance in enumerate(transcript_data["utterances"]):
                speaker_idx = speakers_list.index(utterance["speaker"])
                start_time = utterance["start"] / 1000  # Convert to seconds
                duration = (utterance["end"] - utterance["start"]) / 1000
                ax.barh(y=speaker_idx, 
                       width=duration, 
                       left=start_time, 
                       color=speaker_colors[utterance["speaker"]],
                       alpha=0.7)
            
            ax.set_yticks(range(len(speakers_list)))
            ax.set_yticklabels(speakers_list)
            ax.set_xlabel("Time (seconds)")
            ax.set_title("Conversation Flow Timeline")
            st.pyplot(fig)

        # Tab 3: Topic Distribution - Shows what topics were discussed and how confident we are about each topic
        with insights_tabs[2]:
            st.subheader("Topic Distribution Analysis")
            
            if topics:
                # Filter out less relevant topics to focus on the important ones
                significant_topics = {k: v for k, v in topics.items() 
                                   if v > CONFIDENCE_THRESHOLD}
                
                if significant_topics:
                    # Create treemap of topics
                    fig, ax = plt.subplots(figsize=(12, 8))
                    plt.rcParams['font.size'] = 12
                    
                    # Sort topics by confidence
                    sorted_topics = dict(sorted(significant_topics.items(), 
                                              key=lambda x: x[1], 
                                              reverse=True))
                    
                    # Create horizontal bar chart
                    y_pos = range(len(sorted_topics))
                    plt.barh(y_pos, sorted_topics.values())
                    plt.yticks(y_pos, sorted_topics.keys())
                    plt.xlabel('Confidence Score')
                    plt.title('Topic Distribution by Confidence')
                    
                    st.pyplot(fig)
                else:
                    st.write("No significant topics detected above the confidence threshold.")
            else:
                st.write("No topic data available.")

        # Tab 4: Sentiment Timeline - Shows how the emotional tone changes throughout the conversation
        with insights_tabs[3]:
            st.subheader("Sentiment Analysis Timeline")
            
            if sentiment_analysis:
                # Create timeline showing emotional changes - Green for positive, Gray for neutral, Red for negative
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Map sentiments to numerical values
                sentiment_map = {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}
                times = [(s["start"] + s["end"]) / 2000 for s in sentiment_analysis]  # Convert to seconds
                sentiments = [sentiment_map[s["sentiment"]] for s in sentiment_analysis]
                
                # Create scatter plot with connecting lines
                plt.plot(times, sentiments, 'b-', alpha=0.3)
                plt.scatter(times, sentiments, c=[{
                    "POSITIVE": "green",
                    "NEUTRAL": "gray",
                    "NEGATIVE": "red"
                }[s["sentiment"]] for s in sentiment_analysis])
                
                plt.ylim([-1.5, 1.5])
                plt.yticks([-1, 0, 1], ['Negative', 'Neutral', 'Positive'])
                plt.xlabel('Time (seconds)')
                plt.title('Sentiment Timeline')
                
                st.pyplot(fig)
                
                # Calculate and display sentiment distribution
                sentiment_counts = Counter(s["sentiment"] for s in sentiment_analysis)
                total_segments = len(sentiment_analysis)
                
                st.markdown("### Sentiment Distribution")
                cols = st.columns(3)
                
                with cols[0]:
                    positive_percentage = (sentiment_counts.get("POSITIVE", 0) / total_segments) * 100
                    st.metric("Positive", f"{positive_percentage:.1f}%")
                
                with cols[1]:
                    neutral_percentage = (sentiment_counts.get("NEUTRAL", 0) / total_segments) * 100
                    st.metric("Neutral", f"{neutral_percentage:.1f}%")
                
                with cols[2]:
                    negative_percentage = (sentiment_counts.get("NEGATIVE", 0) / total_segments) * 100
                    st.metric("Negative", f"{negative_percentage:.1f}%")
            else:
                st.write("No sentiment analysis data available.")

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
        if topics:
            filtered_topics = {topic: confidence for topic, confidence in topics.items() if confidence > CONFIDENCE_THRESHOLD}
            if filtered_topics:
                for topic, confidence in filtered_topics.items():
                    st.markdown(f"<div class='box'>Topic: <b>{topic}</b>, Confidence: {confidence:.2f}</div>", unsafe_allow_html=True)
            else:
                st.write("No highly relevant topics detected.")
        else:
            st.write("No topics detected.")

        # Content Safety Analysis - Checks for potentially problematic content like hate speech or threats
        st.header("⚠️ Content Safety Analysis")
        if content_safety:
            safety_categories = {
                "hate_speech": "Hate Speech",
                "insult": "Insults",
                "profanity": "Profanity",
                "threat": "Threats",
                "self_harm": "Self-Harm References",
                "sexual": "Sexual Content",
                "violence": "Violence"
            }
            
            for category, label in safety_categories.items():
                if category in content_safety:
                    confidence = content_safety[category]
                    if confidence > CONFIDENCE_THRESHOLD:
                        st.markdown(
                            f"<div class='box' style='background-color: #FF6B6B;'>{label} detected with {confidence:.1%} confidence</div>",
                            unsafe_allow_html=True
                        )
        else:
            st.write("No content safety concerns detected.")

    except Exception as e:
        st.error(f"Error during audio analysis: {e}")
