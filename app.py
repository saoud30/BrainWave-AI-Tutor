import streamlit as st
import requests
import json
from PIL import Image
import io
import wolframalpha
import os
from dotenv import load_dotenv
from groq import Groq
import logging
import plotly.graph_objects as go
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check if environment variables are loaded
if not os.getenv("GROQ_API_KEY") or not os.getenv("WOLFRAM_ALPHA_APP_ID"):
    st.error("Error: Environment variables are not set. Please check your .env file.")
    logger.error("Missing environment variables")
    st.stop()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Function to get response from Groq API
def get_groq_response(prompt):
    try:
        completion = client.chat.completions.create(
            model="llama3-groq-70b-8192-tool-use-preview",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.5,
            max_tokens=1024,
            top_p=0.65,
            stream=False,
            stop=None,
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling Groq API: {str(e)}")
        st.error(f"Error calling Groq API: {str(e)}")
        return None

# Function to get response from Wolfram Alpha
def get_wolfram_alpha_response(query):
    app_id = os.getenv("WOLFRAM_ALPHA_APP_ID")
    if not app_id:
        logger.error("Wolfram Alpha App ID is not set")
        return "Error: Wolfram Alpha App ID is not set"

    client = wolframalpha.Client(app_id)
    try:
        res = client.query(query)
        if res.success:
            return next(res.results).text
        else:
            logger.info(f"Wolfram Alpha couldn't find an answer for: {query}")
            return "Wolfram Alpha couldn't find an answer to this question."
    except StopIteration:
        logger.info(f"Wolfram Alpha couldn't provide a clear answer for: {query}")
        return "Wolfram Alpha couldn't provide a clear answer to this question."
    except Exception as e:
        logger.error(f"Error calling Wolfram Alpha API: {str(e)}")
        return f"Error calling Wolfram Alpha API: {str(e)}"

# Function to perform sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0.05:
        return "Positive"
    elif sentiment < -0.05:
        return "Negative"
    else:
        return "Neutral"

# Function to generate a concept map
def generate_concept_map(topic):
    prompt = f"Generate a concept map for {topic} in the context of BCA and AI/ML. Format the response as a Python dictionary where keys are main concepts and values are lists of related subconcepts."
    response = get_groq_response(prompt)
    try:
        concept_map = eval(response)
        return concept_map
    except:
        st.error("Failed to generate concept map. Please try again.")
        return None

# Streamlit UI
st.set_page_config(page_title="BrainWave AI Tutor", layout="wide")

# Custom CSS for better visibility while maintaining Streamlit's default background
st.markdown("""
    <style>
    .stTextArea > div > div > textarea {
        background-color: #262730;
        color: #ffffff;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("BrainWave AI Tutor")

# Sidebar for topic selection and additional features
st.sidebar.header("Options")
topic = st.sidebar.selectbox("Select a topic", ["General", "Math", "Artificial Intelligence", "Neural Networks", "Machine Learning", "Linear Algebra"])

# Main content
st.header(f"Ask a question about {topic}")
user_question = st.text_area("Enter your question here:", height=100, key="user_question")

# New feature: Concept Map Generator
if st.sidebar.button("Generate Concept Map"):
    concept_map = generate_concept_map(topic)
    if concept_map:
        st.sidebar.subheader(f"Concept Map for {topic}")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title(f"Concept Map: {topic}")
        ax.axis('off')
        
        for i, (key, values) in enumerate(concept_map.items()):
            ax.text(0.1, 1 - (i+1)*0.1, key, fontsize=12, fontweight='bold')
            for j, value in enumerate(values):
                ax.text(0.3, 1 - (i+1)*0.1 - (j+1)*0.05, f"- {value}", fontsize=10)
        
        st.sidebar.pyplot(fig)

if st.button("Get Answer"):
    if user_question:
        with st.spinner("Thinking..."):
            # Try Wolfram Alpha first for mathematical and scientific questions
            wolfram_response = get_wolfram_alpha_response(user_question)
            
            st.subheader("Wolfram Alpha Response:")
            st.info(wolfram_response)
            
            # Get Groq response for broader context and explanation
            prompt = f"Answer the following question related to {topic} in the context of a BCA course with a minor in AI and ML: {user_question}"
            groq_response = get_groq_response(prompt)
            
            if groq_response:
                st.subheader("AI Assistant Response:")
                st.success(groq_response)
                
                # Sentiment Analysis
                sentiment = analyze_sentiment(groq_response)
                st.write(f"Response Sentiment: {sentiment}")
                
                # Visualization of sentiment
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = TextBlob(groq_response).sentiment.polarity,
                    title = {'text': "Sentiment Analysis"},
                    gauge = {'axis': {'range': [-1, 1]},
                             'bar': {'color': "#1E88E5"},
                             'steps' : [
                                 {'range': [-1, -0.5], 'color': "lightgray"},
                                 {'range': [-0.5, 0.5], 'color': "gray"},
                                 {'range': [0.5, 1], 'color': "darkgray"}],
                             'threshold': {
                                 'line': {'color': "red", 'width': 4},
                                 'thickness': 0.75,
                                 'value': TextBlob(groq_response).sentiment.polarity}}))
                st.plotly_chart(fig)
            
            # Update study tips based on the question
            study_tips_prompt = f"Provide 3 effective study tips for BCA students focusing on the topic: {user_question}"
            study_tips = get_groq_response(study_tips_prompt)
            if study_tips:
                st.sidebar.subheader("Related Study Tips:")
                st.sidebar.info(study_tips)

            # Update learning resources based on the question
            resources_prompt = f"Recommend 3 online learning resources for BCA students interested in learning more about: {user_question}"
            resources = get_groq_response(resources_prompt)
            if resources:
                st.sidebar.subheader("Related Learning Resources:")
                st.sidebar.success(resources)
            
            # Generate a practice question
            practice_prompt = f"Generate a practice question related to {topic} based on the user's question: {user_question}"
            practice_question = get_groq_response(practice_prompt)
            if practice_question:
                st.subheader("Practice Question:")
                st.warning(practice_question)
            
            # New feature: Key Concepts Extraction
            key_concepts_prompt = f"Extract and list 5 key concepts from the answer related to {topic} and the question: {user_question}"
            key_concepts = get_groq_response(key_concepts_prompt)
            if key_concepts:
                st.subheader("Key Concepts:")
                concepts_list = key_concepts.split('\n')
                for concept in concepts_list:
                    st.write(f"• {concept.strip()}")

            # New feature: Further Reading Suggestions
            further_reading_prompt = f"Suggest 3 academic papers or books for further reading on {topic} related to the question: {user_question}. Format as a numbered list with title and brief description."
            further_reading = get_groq_response(further_reading_prompt)
            if further_reading:
                st.subheader("Further Reading:")
                st.markdown(further_reading)

    else:
        st.warning("Please enter a question.")

# New feature: Learning Progress Tracker
if 'learning_progress' not in st.session_state:
    st.session_state.learning_progress = {}

if st.sidebar.button("Update Learning Progress"):
    progress = st.sidebar.slider("Rate your understanding of the current topic", 0, 100, 50)
    st.session_state.learning_progress[topic] = progress
    st.sidebar.success(f"Progress for {topic} updated to {progress}%")

if st.session_state.learning_progress:
    st.sidebar.subheader("Your Learning Progress")
    progress_df = pd.DataFrame.from_dict(st.session_state.learning_progress, orient='index', columns=['Progress'])
    st.sidebar.bar_chart(progress_df)

# Footer
st.markdown("---")
st.markdown("Powered by Groq and Wolfram Alpha")