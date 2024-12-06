# Building chatbot using paid LLM's and open source LLM

# from langchain_openai import ChatOpenAI # Open AI API
from langchain_core.prompts import ChatPromptTemplate # Prompt template
from langchain_core.output_parsers import StrOutputParser # Default output parser whenever a LLM model gives any response
# from langchain_community.llms import Ollama
# from langchain_groq import ChatGroq
import langchain_groq
from langchain_groq import ChatGroq
import streamlit as st # UI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Langsmith tracking (Observable)
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# Defining Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Act as a professional fashion designer advisor. Provide guidance on fashion trends, outfit styling, fabric choices, and color combinations. Avoid unrelated topics."),
        ("user", "Question: {question}")
    ]
)

# UI
st.title("Fashion Designer Advisor 👗👠")
inputText = st.text_input("Ask for fashion advice")

# Using groq inference engine
groqApi = ChatGroq(model="gemma-7b-It", temperature=1)  # 0-2
outputparser = StrOutputParser()
chainSec = prompt | groqApi | outputparser

# Respond to user input
if inputText:
    st.write(chainSec.invoke({'question': inputText}))
