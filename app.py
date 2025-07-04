import os
import streamlit as st 
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import AzureChatOpenAI
from pytube import YouTube
from youtube_transcript_api import (
    YouTubeTranscriptApi, TranscriptsDisabled,
    NoTranscriptFound, VideoUnavailable, CouldNotRetrieveTranscript
)
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

# Load .env variables
load_dotenv()
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
os.environ["AZURE_OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION")

# Clean YouTube URL
def clean_youtube_url(url):
    if "youtu.be" in url:
        return url.split("?")[0]
    elif "youtube.com" in url:
        parsed_url = urlparse(url)
        video_id = parse_qs(parsed_url.query).get("v")
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id[0]}"
    return url

# Get transcript from YouTube
def get_youtube_transcript(video_url):
    try:
        video_id = YouTube(video_url).video_id
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_transcript(["en"])
        except NoTranscriptFound:
            transcript = transcript_list.find_generated_transcript(["en"])

        transcript_data = transcript.fetch()
        text = " ".join([item.text for item in transcript_data])
        return text
    except TranscriptsDisabled:
        st.error("Transcript disabled for this video.")
    except NoTranscriptFound:
        st.error("No transcript found for this video.")
    except VideoUnavailable:
        st.error("Video unavailable.")
    except CouldNotRetrieveTranscript:
        st.error("Could not retrieve transcript. Please try again later.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    return ""

# Streamlit UI
st.title("AI Powered Tutor (No Embeddings)")
st.write("Ask questions from a YouTube video transcript")

video_url = st.text_input("Enter YouTube video URL")

# Get and process transcript
if st.button("Get Transcript"):
    if video_url:
        video_url = clean_youtube_url(video_url)
        transcript_text = get_youtube_transcript(video_url)
        if transcript_text:
            st.info(f"Transcript length: {len(transcript_text)} characters")
            st.session_state.transcript = transcript_text

            # Setup LLMChain
            llm = AzureChatOpenAI(
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY")
            )

            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""
You are a helpful assistant. Use the following video transcript to answer the question.

Transcript:
{context}

Question:
{question}
"""
            )

            qa_chain = LLMChain(llm=llm, prompt=prompt_template)
            st.session_state.qa_chain = qa_chain
            st.success("Transcript loaded. You can now ask questions.")

# Handle question input
if "qa_chain" in st.session_state and "transcript" in st.session_state:
    user_question = st.text_input("Ask a question:")
    if user_question:
        answer = st.session_state.qa_chain.run({
            "context": st.session_state.transcript,
            "question": user_question
        })
        st.write("Answer:", answer)
