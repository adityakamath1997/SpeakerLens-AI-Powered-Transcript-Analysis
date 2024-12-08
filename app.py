import os
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter,defaultdict
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from src.assemblyai_processing import get_audio_intelligence
from src.rag_system import initialize_rag_system

# Basic configuration for the Streamlit application interface
st.set_page_config(
    page_title="Audio Insights Hub",  # Title that appears on the browser tab
    page_icon="🎙️",  # Icon
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

# Arbitrary color scheme for speaker differentiation, you can changet his as per requirements
SPEAKER_COLORS = [
    "#FF6F61",  # Coral
    "#6B5B95",  # Purple
    "#88B04B",  # Green
    "#F7CAC9",  # Pink
    "#92A8D1",  # Blue
    "#955251",  # Brown
    "#B565A7",  # Magenta
    "#009B77"   # Teal
]

# Create speaker_colors dictionary dynamically when processing transcript
def assign_speaker_colors(speakers):
    """
    Assigns colors to speakers from the predefined color scheme
    """
    return {
        speaker: SPEAKER_COLORS[i % len(SPEAKER_COLORS)]
        for i, speaker in enumerate(speakers.keys())
    }

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

def extract_keywords(text, entities):
    """
    Analyzes text to count occurrences of detected entities.
    Returns frequency count of entities found in the text.
    """
    # Create a dictionary of all entities and their variations
    entity_dict = {}
    for entity in entities:
        # Use the entity text as key and store its type
        entity_dict[entity["text"].lower()] = entity["entity_type"]
    
    # Count occurrences of entities in text
    words = text.lower().split()
    entity_counts = Counter()
    
    # Look for entities in the text
    for word in words:
        if word in entity_dict:
            entity_counts[word] += 1
            
    return entity_counts

# Custom dark theme
st.markdown(
    """
    <style>
    /* Gradient background */
    .stApp {
        background: linear-gradient(to bottom right, #1a1a2e, #16213e, #1a1a2e);
        color: #ffffff;
    }
    
    /* Enhanced box styling */
    .box {
        background: rgba(62, 62, 62, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Enhanced headings */
    h1, h2, h3, h4 {
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        padding: 10px 0;
        border-bottom: 2px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    
    /* Styled file uploader */
    .st-upload-area {
        color: #ffffff;
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 20px;
        transition: all 0.3s ease;
    }
    
    .st-upload-area:hover {
        border-color: rgba(255, 255, 255, 0.4);
        background: rgba(255, 255, 255, 0.1);
    }
    
    /* Enhanced legend items */
    .legend-item {
        display: inline-block;
        margin-right: 12px;
        margin-bottom: 8px;
        padding: 5px 12px;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease;
    }
    
    .legend-item:hover {
        transform: scale(1.05);
    }
    
    /* Styled tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: #ffffff;
        padding: 8px 16px;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Styled metrics */
    [data-testid="stMetricValue"] {
        background: rgba(255, 255, 255, 0.05);
        padding: 10px;
        border-radius: 8px;
        font-size: 1.2em !important;
        color: #4CAF50;
    }
    
    /* Styled expanders */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 10px !important;
        transition: background-color 0.2s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: rgba(255, 255, 255, 0.1);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set up the main user interface
st.title("🎙️ Enhanced Audio Analysis")
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
        # Retrieves analysis results from audio processing including all features
        transcription, speakers, summary, entities, sentiment_analysis, topics, content_safety, transcript_data = get_audio_intelligence(raw_audio_path)

        # Assign colors to speakers
        speaker_colors = assign_speaker_colors(speakers)

        # Confidence threshold for topic relevance filtering, this can be changed as per requirements
        CONFIDENCE_THRESHOLD = 0.5

        # Create main dashboard tabs
        st.header("📊 Analytics Dashboard")
        dashboard_tabs = st.tabs([
            "📝 Transcript",
            "👥 Speaker Analysis",
            "📊 Entity Insights",
            "💡 Advanced Analytics",
            "📋 Summary & Analysis",
            "❓ Q&A"
        ])

        # Tab 1: Transcript View
        with dashboard_tabs[0]:
            transcript_view = st.tabs([
                "Line by Line",
                "Full Transcript"
            ])
            
            with transcript_view[0]:
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
            
            with transcript_view[1]:
                st.subheader("Full Transcript")
                st.markdown(f"<div class='box'>{transcription}</div>", unsafe_allow_html=True)

        # Tab 2: Speaker Analysis
        with dashboard_tabs[1]:
            st.subheader("Speaker Analysis")
            
            # Speaker Legend
            st.markdown("### Speaker Legend")
            legend_html = ""
            for speaker, color in speaker_colors.items():
                legend_html += f"<div class='legend-item' style='background-color: {color};'>{speaker}</div>"
            st.markdown(legend_html, unsafe_allow_html=True)
            
            # Speaker Metrics
            with st.expander("📊 Speaking Time Distribution", expanded=True):
                total_duration = sum(data["duration"] for data in speakers.values())
                speaking_times = {speaker: (data["duration"] / total_duration) * 100 
                                for speaker, data in speakers.items()}
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.pie(speaking_times.values(), 
                       labels=[f"{speaker}\n({percentage:.1f}%)" 
                              for speaker, percentage in speaking_times.items()],
                       colors=[speaker_colors[speaker] for speaker in speaking_times.keys()],
                       autopct='%1.1f%%')
                plt.title("Speaking Time Distribution")
                st.pyplot(fig)

            # Individual Speaker Transcripts
            with st.expander("📝 Speaker-Specific Transcripts"):
                for speaker, data in speakers.items():
                    color = speaker_colors[speaker]
                    duration_formatted = format_duration(data["duration"])
                    st.markdown(f"#### {speaker} - {duration_formatted}")
                    st.markdown(
                        f"<div class='box' style='background-color: {color}; color: #ffffff;'>{data['text']}</div>",
                        unsafe_allow_html=True
                    )

        # Tab 3: Entity Insights
        with dashboard_tabs[2]:
            st.subheader("Entity Analysis")
            
            entity_tabs = st.tabs([
                "Overall Entity Distribution",
                "Speaker-Specific Entities",
                "Entity Word Clouds"
            ])
            
            with entity_tabs[0]:
                st.markdown("### Most Mentioned Entities")
                all_text = " ".join(data["text"] for data in speakers.values())
                overall_entities = extract_keywords(all_text, entities)
                most_common_entities = overall_entities.most_common(10)
                if most_common_entities:
                    entities_words, counts = zip(*most_common_entities)
                    fig, ax = plt.subplots()
                    ax.barh(entities_words, counts, color="#88B04B")
                    ax.set_xlabel("Frequency")
                    st.pyplot(fig)
            
            with entity_tabs[1]:
                for speaker, data in speakers.items():
                    with st.expander(f"🎤 {speaker}'s Entities"):
                        speaker_entities = extract_keywords(data["text"], entities)
                        common_entities = speaker_entities.most_common(5)
                        if common_entities:
                            entities_words, counts = zip(*common_entities)
                            fig, ax = plt.subplots()
                            ax.barh(entities_words, counts, color=speaker_colors[speaker])
                            ax.set_xlabel("Frequency")
                            st.pyplot(fig)
            
            with entity_tabs[2]:
                # Your existing word clouds
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Overall Entity Cloud")
                    wordcloud_dict = dict(overall_entities)
                    if wordcloud_dict:
                        wordcloud = WordCloud(width=400, height=200, 
                                            background_color="black", 
                                            colormap="Pastel1").generate_from_frequencies(wordcloud_dict)
                        plt.figure(figsize=(10, 5))
                        plt.imshow(wordcloud, interpolation="bilinear")
                        plt.axis("off")
                        st.pyplot(plt)

        # Tab 4: Advanced Analytics
        with dashboard_tabs[3]:
            st.subheader("Advanced Insights")
            
            with st.expander("🔄 Conversation Flow", expanded=True):
                fig, ax = plt.subplots(figsize=(12, 6))
                speakers_list = list(speakers.keys())
                for i, utterance in enumerate(transcript_data["utterances"]):
                    speaker_idx = speakers_list.index(utterance["speaker"])
                    start_time = utterance["start"] / 1000
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
            
            with st.expander("📊 Topic Distribution"):
                if topics:
                    significant_topics = {k: v for k, v in topics.items() 
                                       if v > CONFIDENCE_THRESHOLD}
                    if significant_topics:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        plt.barh(list(significant_topics.keys()), 
                                list(significant_topics.values()))
                        plt.xlabel('Confidence Score')
                        plt.title('Topic Distribution')
                        st.pyplot(fig)

        # Tab 5: Summary & Analysis
        with dashboard_tabs[4]:
            st.subheader("Summary & Analysis")
            
            analysis_tabs = st.tabs([
                "📝 Summary",
                "🔍 Entities",
                "💭 Sentiment",
                "📚 Topics",
                "⚠️ Content Safety"
            ])
            
            # Summary Tab
            with analysis_tabs[0]:
                st.markdown("### 📝 Transcript Summary")
                st.markdown(f"<div class='box'>{summary}</div>", unsafe_allow_html=True)
            
            # Entities Tab
            with analysis_tabs[1]:
                st.markdown("### 🔍 Entities Detected")
                grouped_entities = defaultdict(list)
                for entity in entities:
                    grouped_entities[entity["entity_type"]].append(entity["text"])
                
                for category, items in grouped_entities.items():
                    with st.expander(f"{category.capitalize()}"):
                        unique_items = list(set(items))  # Remove duplicates
                        st.write(", ".join(unique_items))
            
            # Sentiment Tab
            with analysis_tabs[2]:
                st.markdown("### 💭 Sentiment Analysis")
                sentiments_summary = {}
                for sentiment in sentiment_analysis:
                    sentiment_type = sentiment["sentiment"]
                    sentiments_summary[sentiment_type] = sentiments_summary.get(sentiment_type, 0) + 1
                
                # Create a pie chart for sentiment distribution
                if sentiments_summary:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    colors = {
                        'POSITIVE': '#90EE90',
                        'NEUTRAL': '#F0E68C',
                        'NEGATIVE': '#FFB6C1'
                    }
                    plt.pie(sentiments_summary.values(),
                           labels=[f"{k}\n({v} occurrences)" for k, v in sentiments_summary.items()],
                           colors=[colors.get(k, '#808080') for k in sentiments_summary.keys()],
                           autopct='%1.1f%%')
                    plt.title("Sentiment Distribution")
                    st.pyplot(fig)
                
                st.markdown(
                    "<div class='box'>" +
                    ", ".join(f"<b>{key}:</b> {value} occurrences" for key, value in sentiments_summary.items()) +
                    "</div>",
                    unsafe_allow_html=True
                )
            
            # Topics Tab
            with analysis_tabs[3]:
                st.markdown("### 📚 Relevant Topics")
                if topics:
                    filtered_topics = {topic: confidence 
                                     for topic, confidence in topics.items() 
                                     if confidence > CONFIDENCE_THRESHOLD}
                    if filtered_topics:
                        # Create bar chart for topic confidence
                        fig, ax = plt.subplots(figsize=(10, 6))
                        topics_sorted = dict(sorted(filtered_topics.items(), 
                                                  key=lambda x: x[1], 
                                                  reverse=True))
                        plt.barh(list(topics_sorted.keys()), 
                                list(topics_sorted.values()))
                        plt.xlabel('Confidence Score')
                        plt.title('Topic Detection Confidence')
                        st.pyplot(fig)
                        
                        for topic, confidence in topics_sorted.items():
                            st.markdown(
                                f"<div class='box'>Topic: <b>{topic}</b>, "
                                f"Confidence: {confidence:.2f}</div>",
                                unsafe_allow_html=True
                            )
                    else:
                        st.write("No highly relevant topics detected.")
                else:
                    st.write("No topics detected.")
            
            # Content Safety Tab
            with analysis_tabs[4]:
                st.markdown("### ⚠️ Content Safety Analysis")
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
                    
                    # Create bar chart for safety metrics
                    fig, ax = plt.subplots(figsize=(10, 6))
                    safety_data = {safety_categories[k]: v 
                                 for k, v in content_safety.items() 
                                 if k in safety_categories}
                    if safety_data:
                        plt.barh(list(safety_data.keys()), 
                                list(safety_data.values()))
                        plt.xlabel('Confidence Score')
                        plt.title('Content Safety Analysis')
                        st.pyplot(fig)
                    
                    for category, label in safety_categories.items():
                        if category in content_safety:
                            confidence = content_safety[category]
                            color = "#FF6B6B" if confidence > CONFIDENCE_THRESHOLD else "#4CAF50"
                            st.markdown(
                                f"<div class='box' style='background-color: {color};'>"
                                f"{label}: {confidence:.1%} confidence</div>",
                                unsafe_allow_html=True
                            )
                else:
                    st.write("No content safety concerns detected.")

        # Separator between dashboard and Q&A section
        st.markdown("---")

         # RAG-based Chat Interface
        st.header("💬 Chat with Your Transcript")
        
        # Initialize session state
        if "rag_system" not in st.session_state:
            rag, qa_chain = initialize_rag_system(transcription, speakers)
            st.session_state.rag_system = rag
            st.session_state.qa_chain = qa_chain
            st.session_state.chat_history = []

        # Chat interface
        user_question = st.text_input(
            "Ask a question about the conversation:",
            key="user_question"
        )
        
        if user_question:
            with st.spinner("Processing your question..."):
                result = st.session_state.rag_system.query(
                    st.session_state.qa_chain,
                    user_question
                )
                
                # Display the answer
                st.markdown("**Answer:**")
                st.markdown(result["answer"])
                
                # Display sources in an expander
                with st.expander("View Sources"):
                    for idx, source in enumerate(result["sources"], 1):
                        st.markdown(f"**Source {idx}:**")
                        if source["metadata"]["type"] == "speaker_specific":
                            st.markdown(f"*Speaker: {source['metadata']['speaker']}*")
                        st.markdown(f"```\n{source['text']}\n```")
                        st.markdown("---")

                # Add to chat history
                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": result["answer"]
                })

        # Display chat history
        if st.session_state.chat_history:
            st.markdown("### Chat History")
            for entry in st.session_state.chat_history:
                st.markdown(f"**Q:** {entry['question']}")
                st.markdown(f"**A:** {entry['answer']}")
                st.markdown("---")

    except Exception as e:
        st.error(f"Error during audio analysis: {e}")