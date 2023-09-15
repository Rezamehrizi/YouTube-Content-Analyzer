import streamlit as st
import assemblyai as aai
import pandas as pd
from pytube import YouTube
# import requests
# from time import sleep


#%%
if 'status' not in st.session_state:
    st.session_state['status'] = 'submitted'

#%%
API_KEY = "fdadb4bb044b40f58685660be655bcd9"
upload_endpoint = "https://api.assemblyai.com/v2/upload"
transcript_endpoint = "https://api.assemblyai.com/v2/transcript"

headers = {
    # "authorization": st.secrets["auth_key"],
    "authorization": API_KEY,
    "content-type": "application/json"
}

# Your API token is already set here
aai.settings.api_key = API_KEY

#%%
@st.experimental_memo
def save_audio(url):
    yt = YouTube(url)
    video = yt.streams.filter(only_audio=True).first()
    out_file = video.download()
    base, ext = os.path.splitext(out_file)
    file_name = base.replace(" ", "-") + '.mp3'
    os.rename(out_file, file_name)
    print(yt.title + " has been successfully downloaded.")
    print(file_name)
    return yt.title, file_name, yt.thumbnail_url


#%%
st.title("YouTube Content Analyzer")
st.markdown("With this app you can recieve the following information of Youtube video:")
st.markdown('1. The full transcrption of the video')
st.markdown("2. A summary of the video,") 
st.markdown("2. the topics that are discussed in the video,") 
st.markdown("3. whether there are any sensitive topics discussed in the video.")
st.markdown("Make sure your video is not long and link is in the format: https://www.youtube.com/watch?v=HfNnuQOHAaw and not https://youtu.be/HfNnuQOHAaw")

# 'https://www.youtube.com/watch?v=E9hog8Maq1I'
# 'https://www.youtube.com/watch?v=E9hog8Maq1I'
link = st.text_input('Enter your YouTube video link', 'https://www.youtube.com/watch?v=E9hog8Maq1I')
st.video(link)
# st.text("The transcription is " + st.session_state['status'])

video_title, save_location, video_thumbnail = save_audio(link)

st.header(video_title)
st.audio(save_location)

#%%
config = aai.TranscriptionConfig(
  summarization=True,
  summary_model=aai.SummarizationModel.informative, # optional
  summary_type=aai.SummarizationType.bullets, # optional
  content_safety=True,
  sentiment_analysis=True,
  entity_detection=True,
  iab_categories=True,
  auto_highlights=True
)

transcriber = aai.Transcriber(config=config)
transcript = transcriber.transcribe(save_location)


#%%
sensitive_data = [{
        'Lable': key,
        'Confidence': value}
    for key, value in transcript.content_safety.summary.items()]
sensitive_df = pd.DataFrame(sensitive_data)

#%% Tabs
tab1, tab2, tab3 = st.tabs(["Transcrption | Summary", "Topic | Sensitive Content", "Sentiment Analysis | Entity"])

with tab1:
    st.subheader("Transcrption")
    st.write(transcript.text)
    
    st.subheader("Summary")
    st.write(transcript.summary)


with tab2:
    # Topic Detection
    st.header("Topic")
    topics_df = pd.DataFrame(transcript.iab_categories.summary.items())
    topics_df.columns = ['topic','confidence']
    topics_df["topic"] = topics_df["topic"].str.split(">")
    expanded_topics = topics_df.topic.apply(pd.Series).add_prefix('topic_level_')
    topics_df = topics_df.join(expanded_topics).drop('topic', axis=1).sort_values(['confidence'], ascending=False).fillna('')
    st.dataframe(topics_df)

    # Content Moderation
    st.subheader("Sensitive content")
    sensitive_data = [{
            'Lable': key,
            'Confidence': value}
        for key, value in transcript.content_safety.summary.items()]
    
    sensitive_df = pd.DataFrame(sensitive_data)
    st.write(sensitive_df)
    

with tab3:
    # Sentiment Analysis
    st.subheader("Sentiment Analysis")
    sentiment_data = [{
            'Text': result.text,
            'Sentiment Type': result.sentiment,
            'Confidence': result.confidence}
        for result in transcript.sentiment_analysis]

    sentiment_df = pd.DataFrame(sentiment_data)
    st.dataframe(sentiment_df)

    # Entity Detection
    st.header("Entity")
    # Assuming you have a list of entities called transcript.entities
    entity_data = [{
            'Text': entity.text,
            'Entity Type': entity.entity_type}
        for entity in transcript.entities]
    
    entity_df = pd.DataFrame(entity_data)
    st.dataframe(entity_df)










