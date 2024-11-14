import os
import requests
from dotenv import load_dotenv
import time

# Load the AssemblyAI API key from environment variables
load_dotenv()
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

# Set up base URL and headers for making requests to the AssemblyAI API
ASSEMBLYAI_URL = "https://api.assemblyai.com/v2"
HEADERS = {"authorization": ASSEMBLYAI_API_KEY}

def upload_audio(file_path):
    """
    Upload an audio file to AssemblyAI and return the URL where it's stored.
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
    Start a basic transcription request with just speaker labels.
    """
    endpoint = f"{ASSEMBLYAI_URL}/transcript"
    json_data = {
        "audio_url": audio_url,
        "speaker_labels": True  # Identifying different speakers
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
    Start a transcription request with additional features like entity detection, sentiment analysis, and summarization.
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
    Continuously check the status of the transcription until it's finished.
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
        print("Transcription in progress... will check again in 5 seconds.")
        time.sleep(5)

def process_transcription_data(transcript_data):
    """
    Extracts useful information from the transcription, including speakers, entities, topics, and sentiment.
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
    
    # Gather other anaylsis data
    entities = transcript_data.get("entities", [])
    sentiment_analysis = transcript_data.get("sentiment_analysis_results", [])
    topics = transcript_data.get("iab_categories_result", {}).get("summary", {})
    content_safety = transcript_data.get("content_safety_labels", {}).get("summary", {})
    
    return transcription, speakers, summary, entities, sentiment_analysis, topics, content_safety

def get_audio_intelligence(file_path, basic=False):
    """
    Upload an audio file, transcribe it, and extract relevant audio intelligence data.
    If 'basic' is set to True, only basic transcription is performed.
    """
    #Upload the audio file and retrieve the URL
    audio_url = upload_audio(file_path)
    
    #Choose between basic and full feature transcription
    transcript_id = transcribe_basic_audio(audio_url) if basic else transcribe_audio_with_features(audio_url)
    
    #Poll until the transcription process is complete,then retrieve data
    transcript_data = poll_transcription_status(transcript_id)
    
    #Process and return transcription data
    return process_transcription_data(transcript_data)
