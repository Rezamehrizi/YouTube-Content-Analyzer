import streamlit as st
from pytube import YouTube
import os
import pandas as pd
import assemblyai as aai
from functools import lru_cache

# Constants
API_KEY = "fdadb4bb044b40f58685660be655bcd9"

# Initialize AssemblyAI
aai.settings.api_key = API_KEY

st.markdown("""
<style>

	.stTabs [data-baseweb="tab-list"] {gap: 50px; }
    
# 	.stTabs [data-baseweb="tab"] {height: 50px;
#         white-space: pre-wrap;
# 		background-color: #F0F2F6;
# 		border-radius: 4px 4px 0px 0px;
# 		gap: 1px;
# 		padding-top: 10px;
# 		padding-bottom: 10px;}

# 	.stTabs [aria-selected="true"] {background-color: grey;}

</style>""", unsafe_allow_html=True)


st.write("""
<style>
    button[data-baseweb="tab"] {font-size: 17px; }
    button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p { font-size: 24px;}
</style>
""", unsafe_allow_html=True)

# Function to download audio
@st.experimental_memo
def save_audio(url):
    yt = YouTube(url)
    video = yt.streams.filter(only_audio=True).first()
    out_file = video.download()
    base, ext = os.path.splitext(out_file)
    file_name = base.replace(" ", "-") + '.mp3'
    os.rename(out_file, file_name)
    return yt.title, file_name, yt.thumbnail_url

# Function to transcribe audio
# @st.experimental_memo
@lru_cache(maxsize=None)
def transcribe_audio(audio_path):
    config = aai.TranscriptionConfig(
        summarization=True,
        summary_model=aai.SummarizationModel.informative,
        summary_type=aai.SummarizationType.bullets,
        content_safety=True,
        sentiment_analysis=True,
        entity_detection=True,
        iab_categories=True,
        auto_highlights=True
    )
    transcriber = aai.Transcriber(config=config)
    return transcriber.transcribe(audio_path)

# Main App
def main():
    st.title("YouTube Content Analyzer")
    st.markdown("With this app, you can receive information about a YouTube video.")

    # Check if session_state is initialized
    if 'video_info' not in st.session_state:
        st.session_state.video_info = None

    if 'transcript_data' not in st.session_state:
        st.session_state.transcript_data = None

    # Input
    # 'https://www.youtube.com/watch?v=TGSXzFP1WkE'
    # 'https://www.youtube.com/watch?v=E9hog8Maq1I'
    
    link = st.text_input('Enter your YouTube video link', 'https://www.youtube.com/watch?v=TGSXzFP1WkE')
    st.video(link)

    if st.button("Analyze Video"):
        # Check if the video URL has changed
        if st.session_state.video_info is None or st.session_state.video_info[0] != link:
            video_title, save_location, video_thumbnail = save_audio(link)
            st.session_state.video_info = (link, video_title, save_location, video_thumbnail)
        else:
            link, video_title, save_location, video_thumbnail = st.session_state.video_info
            
        st.header(video_title)
        st.audio(save_location)

        if st.session_state.transcript_data is None or st.session_state.transcript_data[0] != link:
            with st.spinner("Transcribing audio..."):
                transcript = transcribe_audio(save_location)
            st.session_state.transcript_data = (link, transcript)
        else:
            link, transcript = st.session_state.transcript_data


        # Display tabs with emojis and spacing
        tabs = st.tabs(["🎙️ Transcription  |  Summary", "📊 Topic | Sensitive Content", "🤔 Sentiment | Entity"])

        with tabs[0]:
            st.subheader("Transcription")
            st.write(transcript.text)

            st.subheader("Summary")
            st.write(transcript.summary)

        with tabs[1]:
            st.header("Topic")
            topics_df = pd.DataFrame(transcript.iab_categories.summary.items())
            topics_df.columns = ['topic','confidence']
            topics_df["topic"] = topics_df["topic"].str.split(">")
            expanded_topics = topics_df.topic.apply(pd.Series).add_prefix('topic_level_')
            topics_df = topics_df.join(expanded_topics).drop('topic', axis=1).sort_values(['confidence'], ascending=False).fillna('')
            st.dataframe(topics_df)

            st.subheader("Sensitive content")
            if transcript.content_safety.summary:
                sensitive_data = [{
                    'Label': key,
                    'Confidence': value}
                    for key, value in transcript.content_safety.summary.items()]
    
                sensitive_df = pd.DataFrame(sensitive_data)
                st.write(sensitive_df)
            else:
                st.write('There is no sensitive content in this video')

        with tabs[2]:
            st.subheader("Sentiment Analysis")
            sentiment_data = [{
                'Text': result.text,
                'Sentiment Type': result.sentiment,
                'Confidence': result.confidence}
                for result in transcript.sentiment_analysis]

            sentiment_df = pd.DataFrame(sentiment_data)
            st.dataframe(sentiment_df)

            st.header("Entity")
            entity_data = [{
                'Text': entity.text,
                'Entity Type': entity.entity_type}
                for entity in transcript.entities]

            entity_df = pd.DataFrame(entity_data)
            st.dataframe(entity_df)

if __name__ == "__main__":
    main()
