import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

print(f"API Key loaded: {api_key[:10]}..." if api_key else "No API key found")

# Test 1: Direct Google AI
try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content("What is AI?")
    print("Direct Google AI works!")
    print(f"Response: {response.text[:100]}...")
except Exception as e:
    print(f"Direct Google AI failed: {e}")

# Test 2: LangChain Google AI
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    response = llm.invoke("What is AI?")
    print("LangChain Google AI works!")
    print(f"Response: {response.content[:100]}...")
except Exception as e:
    print(f"LangChain Google AI failed: {e}")

# Test 3: Check API key format
if api_key:
    if api_key.startswith("AIza"):
        print("API key format looks correct")
    else:
        print("API key format looks wrong - should start with 'AIza'")
else:
    print("No API key found in environment")