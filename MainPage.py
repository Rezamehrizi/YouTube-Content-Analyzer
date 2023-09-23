import streamlit as st
from streamlit_option_menu import option_menu
from pytube import YouTube
import os
import pandas as pd
import assemblyai as aai
from functools import lru_cache

# Constants
API_KEY = "fdadb4bb044b40f58685660be655bcd9"

# Initialize AssemblyAI
aai.settings.api_key = API_KEY

# Markdown for Centered Text
st.markdown("""<style>p {text-align: justify;}</style>""", unsafe_allow_html=True)
# Use one-line Markdown to set the page layout to wide
st.markdown('<style>.reportview-container{width:100%;}</style>', unsafe_allow_html=True)
# set the body font size
st.markdown("""<style>body {font-size: 22px;}</style>""", unsafe_allow_html=True)


# Markdown for Background Color
# st.markdown("""<style>[data-testid="stAppViewContainer"] {background-color: lightblue;}</style>""", unsafe_allow_html=True)


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
    # App title and description
    st.title("YouTube Video Analyzer")
    st.write("Using this application, you can comprehensively analyze the content of a YouTube video, gaining access to the following information:")
    
    # List of features
    features = [
        "1. **Video Transcription:** Obtain a complete written transcript of the video's spoken content.",
        "2. **Video Summary:** Receive a concise overview summarizing the key points and themes within the video.",
        "3. **Topics Discussed:** Explore the main subjects and themes addressed within the video.",
        "4. **Content Sensitivity:** Evaluate the video's content for sensitivity, including considerations related to hate speech and sexuality.",
        "5. **Sentiment Analysis:** Gauge the overall sentiment or emotional tone conveyed by the video's content.",
        "6. **Entity Recognition:** Identify and categorize objects or entities discussed within the video."
    ]
    # Display features
    st.write("\n".join(features))

    # Check if session_state is initialized
    if 'video_info' not in st.session_state:
        st.session_state.video_info = None

    if 'transcript_data' not in st.session_state:
        st.session_state.transcript_data = None

    # Use Markdown to create a horizontal line
    st.markdown("---")

    # Input
    # 'https://www.youtube.com/watch?v=TGSXzFP1WkE'
    # 'https://www.youtube.com/watch?v=E9hog8Maq1I'
    
    st.write( "**Note**: When submitting a video, kindly ensure that it is short in length. The analysis of the video will take approximately 20 percent of the video's total length.")
    default_video_bool = st.checkbox('Use the default video')
    if default_video_bool:
        url = 'https://www.youtube.com/watch?v=UNP03fDSj1U'
    else: 
        url = ""

    link = st.text_input('Enter your YouTube video link', url)
    if url:
        st.video(link)

    # if st.button("Analyze Video"):
    # Check if the video URL has changed
    if st.session_state.video_info is None or st.session_state.video_info[0] != link:
        if st.button("Analyze Video"):
            video_title, save_location, video_thumbnail = save_audio(link)
            st.session_state.video_info = (link, video_title, save_location, video_thumbnail)
            with st.spinner("Transcribing audio..."):
                transcript = transcribe_audio(save_location)
            st.session_state.transcript_data = (link, transcript)
            
    else:
        link, video_title, save_location, video_thumbnail = st.session_state.video_info
        link, transcript = st.session_state.transcript_data
        

    st.markdown('---')
    # Hide the navigation bar until after clicking "Analyze Video"
    if st.session_state.transcript_data != None and st.session_state.transcript_data[0] == link:
        # Navigation bar
        selected = option_menu(
            menu_title=None,
            options=["Transcription  |  Summary", "Topic | Sensitivity", "Sentiment | Entity"],
            icons=["mic", "search", "emoji-smile"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "15px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#E0EEEE",
                },
                "nav-link-selected": {"background-color": "#458B74"},
            },
        )
        
        # Update the selected section in session state
        st.session_state.selected_section = selected
        

        if st.session_state.selected_section == "Transcription  |  Summary":
            st.subheader("Transcription")
            st.write(transcript.text)

            st.markdown('---')
            st.subheader("Summary")
            st.write(transcript.summary)

        elif st.session_state.selected_section == "Topic | Sensitivity":
            st.subheader("Topic")
            st.write("This section offers insights into the various topics discussed within the video along with their confidence level. It provides a comprehensive list of the main subjects, themes, or subjects covered during the video's content. Exploring the topic analysis can help you gain a clear understanding of the key areas of focus within the video, making it easier to navigate and comprehend its content.")
            topics_df = pd.DataFrame(transcript.iab_categories.summary.items())
            topics_df.columns = ['topic','confidence']
            topics_df["topic"] = topics_df["topic"].str.split(">")
            expanded_topics = topics_df.topic.apply(pd.Series).add_prefix('topic_level_')
            topics_df = topics_df.join(expanded_topics).drop('topic', axis=1).sort_values(['confidence'], ascending=False).fillna('')
            confidence_column = topics_df.pop('confidence')
            topics_df['confidence'] = confidence_column
            topics_df = topics_df[topics_df['confidence'] >= .5]
            st.table(topics_df)

            st.markdown('---')
            st.subheader("Sensitive content")
            st.write("This section provides an assessment of the sensitivity of the content within the video. It evaluates whether the video contains any potentially sensitive or controversial topics, themes, or discussions. The analysis assigns confidence scores to indicate the level of sensitivity associated with different aspects of the video's content.")
            if transcript.content_safety.summary:
                sensitive_data = [{
                    'Label': key,
                    'Confidence': value}
                    for key, value in transcript.content_safety.summary.items()]
    
                sensitive_df = pd.DataFrame(sensitive_data)
                st.write(sensitive_df)
            else:
                # st.write('There is no sensitive content in this video')
                st.markdown('<p style="color: green;">There is no sensitive content in this video</p>', unsafe_allow_html=True)


        elif st.session_state.selected_section == "Sentiment | Entity":
            st.subheader("Sentiment Analysis")
            st.write("This section offers an evaluation of the overall sentiment expressed within the video  along with their confidence level. It assesses the emotional tone and sentiment type conveyed throughout the video's content, whether it be positive, negative, or neutral. The analysis provides insights into the prevailing sentiments and emotions that characterize the video's narrative or discussion.")
            sentiment_data = [{
                'Text': result.text,
                'Sentiment Type': result.sentiment,
                'Confidence': result.confidence}
                for result in transcript.sentiment_analysis]

            sentiment_df = pd.DataFrame(sentiment_data)

            st.table(sentiment_df)
            

            st.markdown('---')
            st.subheader("Entity")
            st.write("This section presents a comprehensive examination of the entities discussed in the video. Entities refer to specific individuals, organizations, locations, or other notable subjects that are mentioned throughout the video's content. The analysis provides insights into the types of entities featured and their relevance within the context of the video. ")
            entity_data = [{
                'Text': entity.text,
                'Entity Type': entity.entity_type}
                for entity in transcript.entities]

            entity_df = pd.DataFrame(entity_data)
            st.table(entity_df)

if __name__ == "__main__":
    cols = st.columns([1, 10, 1])
    with cols[1]:
        main()
