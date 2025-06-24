from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import *
from dotenv import load_dotenv
import os


# Loading the environment
load_dotenv()

# Creat the Chat Interface
gemini_llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                           verbose=LLM_VERBOSE,
                           temperature=LLM_TEMPERATURE,
                           google_api_key=os.getenv("GOOGLE_API_KEY"))

