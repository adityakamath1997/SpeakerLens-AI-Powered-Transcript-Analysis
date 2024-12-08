import os
import requests
from dotenv import load_dotenv
import time

# API key loaded from environment variables
load_dotenv()
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

# Base URL and headers setup for AssemblyAI API requests
ASSEMBLYAI_URL = "https://api.assemblyai.com/v2"
HEADERS = {"authorization": ASSEMBLYAI_API_KEY}

def upload_audio(file_path):
    """
    Handles the upload of audio filesto AssemblyAI servers and returns a URL for the stored file.
    Includes error handling for failed uploads and logs the process for debuging purposes.
    """
    try:
        with open(file_path, "rb") as f:
            response = requests.post(f"{ASSEMBLYAI_URL}/upload", headers=HEADERS, files={"file": f})
        response.raise_for_status()
        audio_url = response.json().get("upload_url")
        print("Audio file uploaded successfully. URL:", audio_url)  # Log the audio file URL for reference and for deebugging
        return audio_url
    except requests.exceptions.HTTPError as e:
        print(f"Upload failed with HTTP Error: {e}")
        print("Server Response:", e.response.json())
        raise

def transcribe_basic_audio(audio_url):
    """
    Performs basic audio transcrpition with speaker identification. Returns a unique ID
    that can be used to track the transcription progress.
    """
    endpoint = f"{ASSEMBLYAI_URL}/transcript"
    json_data = {
        "audio_url": audio_url,
        "speaker_labels": True  # Identifying different speakers, the crux of this project.
    }
    print("Sending basic transcription request with data:", json_data)  # Log the request payload for debugging
    try:
        response = requests.post(endpoint, headers=HEADERS, json=json_data)
        response.raise_for_status()
        return response.json()["id"]  # Return the unique ID for tracking transcription status
    except requests.exceptions.HTTPError as e:
        print(f"Error during transcription request: {e}")
        print("Server Response:", e.response.json())
        raise

def transcribe_audio_with_features(audio_url):
    """
    Enhanced transcription that includes speaker labels, entity detection, sentiment analysis,
    and content summarization.Provides comprehensive analysis of the audio content using
    all of AssemblyAI's avialable features
    """
    endpoint = f"{ASSEMBLYAI_URL}/transcript"
    json_data = {
        "audio_url": audio_url,
        "speaker_labels": True,
        "entity_detection": True,
        "sentiment_analysis": True,
        "summarization": True,
        "summary_type": "bullets",  # Provide the summary as bullet points
        "summary_model": "conversational",  # Use a conversational model for summarization
        "iab_categories": True,  # Categorize the content using IAB categories
        "content_safety": True  # Flag content safety issues
    }
    print("Sending transcription request with extended features:", json_data)  # Log the request payload for debugging
    try:
        response = requests.post(endpoint, headers=HEADERS, json=json_data)
        response.raise_for_status()
        return response.json()["id"]
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error during transcription request: {e}")
        print("Server Response:", e.response.json())
        raise

def poll_transcription_status(transcript_id):
    """
    Monitors the transcription progress by checking status every 5 seconds until completion.
    Returns the final transcription data or raises an error if the process fails.
    """
    endpoint = f"{ASSEMBLYAI_URL}/transcript/{transcript_id}"
    while True:
        response = requests.get(endpoint, headers=HEADERS)
        response_data = response.json()
        status = response_data["status"]
        if status == "completed":
            print("Transcription completed successfully.")
            return response_data
        elif status == "failed":
            raise RuntimeError("Transcription failed due to an error.")
        print("Transcription in progress... checking again in 5 seconds.")
        time.sleep(5)

def process_transcription_data(transcript_data):
    """
    Processes and organizes the transcription results into structured data. Extracts key information
    including speaker segments, entities,sentiment analysis, and topic categorization.
    """
    # Get the main transcription text
    transcription = transcript_data["text"]
    # Check for available summary or provide a default
    summary = transcript_data.get("summary", "No summary available.")
    
    # Organize speaker specific information
    speakers = {}
    for utterance in transcript_data["utterances"]:
        speaker = utterance["speaker"]
        duration = utterance["end"] - utterance["start"]
        if speaker not in speakers:
            speakers[speaker] = {"duration": 0, "text": ""}
        speakers[speaker]["duration"] += duration
        speakers[speaker]["text"] += f" {utterance['text']} "
    
    # Gather other analysis data
    entities = transcript_data.get("entities", [])
    sentiment_analysis = transcript_data.get("sentiment_analysis_results", [])
    topics = transcript_data.get("iab_categories_result", {}).get("summary", {})
    content_safety = transcript_data.get("content_safety_labels", {}).get("summary", {})
    
    return transcription, speakers, summary, entities, sentiment_analysis, topics, content_safety, transcript_data

def get_audio_intelligence(file_path, basic=False):
    """
    Main processing function that handles the complete workflow from upload to transcription.
    Supports both basic and advanced transcription modes based on the requirements.
    """
    #Upload the audio file and retrieve the URL
    audio_url = upload_audio(file_path)
    
    #Choose between basic and full feature transcription
    transcript_id = transcribe_basic_audio(audio_url) if basic else transcribe_audio_with_features(audio_url)
    
    #Poll until the transcription process is complete,then retrieve data
    transcript_data = poll_transcription_status(transcript_id)
    
    #Process and return transcription data
    return process_transcription_data(transcript_data)
