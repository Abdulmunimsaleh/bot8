from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import io
import random
import string
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
import os
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True)   
import requests
from bs4 import BeautifulSoup
import time
import re
import uvicorn

app = FastAPI(
    title="TripZoori Chatbot API",
    description="API for TripZoori chatbot to be integrated with Tidio",
    version="1.0.0"
)

# Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Function to fetch and parse data from the website
def fetch_website_data(url):
    print("Fetching data from website...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        # Add a delay to avoid being blocked
        time.sleep(1)
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Response status code: {response.status_code}")
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
            return ""
        soup = BeautifulSoup(response.text, 'html.parser')
        # Print the first part of HTML to debug
        print(f"HTML snippet: {soup.prettify()[:500]}...")
        # Extract all text from the website
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        # Extract text from all page elements
        all_text = []
        for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'span', 'div']):
            text = element.get_text(strip=True)
            if text and len(text) > 5:  # Filter out very short snippets
                all_text.append(text)
        # Also get text from any specific content sections that might be relevant
        content_sections = soup.find_all(class_=re.compile('content|description|about|info|details|text'))
        for section in content_sections:
            text = section.get_text(strip=True)
            if text and len(text) > 5:
                all_text.append(text)
        # Look for hotel-related information specifically
        hotel_sections = soup.find_all(string=re.compile('hotel|resort|accommodation|sarova|room|night|price|rate', re.IGNORECASE))
        for section in hotel_sections:
            parent = section.parent
            if parent:
                text = parent.get_text(strip=True)
                if text and len(text) > 5:
                    all_text.append(text)
        # Get welcome message or site header
        welcome_sections = soup.find_all(['h1', 'h2', 'header'], string=re.compile('welcome|hello|trip|zoori', re.IGNORECASE))
        for section in welcome_sections:
            text = section.get_text(strip=True)
            if text:
                all_text.append(f"Welcome message: {text}")
        # If we found text content
        if all_text:
            print(f"Extracted {len(all_text)} text elements")
            full_text = " ".join(all_text)
            # Add fallback content in case the scraping doesn't find specific info
            full_text += " TripZoori is your trip tour guide. We help you plan amazing vacations and find the best hotels, including Sarova hotels."
            return full_text
        else:
            print("Warning: No text content extracted")
            return "TripZoori is your trip tour guide. We help you plan amazing vacations."
    except Exception as e:
        print(f"Error fetching website data: {str(e)}")
        # Return some default information as fallback
        return "TripZoori is your trip tour guide. We help you find the best hotels and vacation spots."

# Initialize global variables
website_url = "https://tripzoori-gittest1.fly.dev/"
raw = ""
sent_tokens = []
word_tokens = []

# Greeting patterns
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = [
    "Hi there! I can provide information from TripZoori.", 
    "Hello! Ask me about TripZoori.",
    "Hey! I'm here to help with TripZoori information.",
    "Greetings! How can I help you with TripZoori today?"
]

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Keyword matching for common questions
def keyword_match(user_response):
    user_response = user_response.lower()
    if any(word in user_response for word in ["welcome", "message"]):
        return "The welcome message of TripZoori is 'Your trip tour guide'."
    if "sarova" in user_response and any(word in user_response for word in ["price", "cost", "rate", "night"]):
        return "Sarova hotels typically range from $150 to $300 per night depending on the room type and season."
    if "how are you" in user_response:
        return "I'm doing great! I'm here to help you with information about TripZoori and travel planning."
    return None

# Generating response
def generate_response(user_response):
    robo_response = ''
    # First check for keyword matches
    keyword_response = keyword_match(user_response)
    if keyword_response:
        return keyword_response
    # If no keyword match, use TF-IDF to find similar content
    sent_tokens.append(user_response)
    try:
        TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
        tfidf = TfidfVec.fit_transform(sent_tokens)
        vals = cosine_similarity(tfidf[-1], tfidf)
        idx = vals.argsort()[0][-2]
        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]
        if req_tfidf == 0:
            # If no match, try to search for keywords in the text
            keywords = [word for word in user_response.split() if len(word) > 3]
            if keywords:
                for sentence in sent_tokens[:-1]:  # Exclude the user query
                    if any(keyword.lower() in sentence.lower() for keyword in keywords):
                        robo_response = sentence
                        break
                if not robo_response:
                    robo_response = "I don't have specific information about that from TripZoori. TripZoori is your trip tour guide for finding great hotels and vacation spots."
            else:
                robo_response = "I'm sorry, I don't understand. Could you rephrase your question about TripZoori?"
        else:
            robo_response = sent_tokens[idx]
    except Exception as e:
        print(f"Error in response generation: {str(e)}")
        robo_response = "TripZoori is your trip tour guide. We can help you find the best hotels and vacation spots."
    sent_tokens.remove(user_response)
    return robo_response

# Initialize function that will run at startup
def initialize_bot():
    global raw, sent_tokens, word_tokens
    # Fetch website data
    raw = fetch_website_data(website_url)
    # Tokenization
    sent_tokens = nltk.sent_tokenize(raw)  # converts to list of sentences
    word_tokens = nltk.word_tokenize(raw)  # converts to list of words
    # Add some predefined information to handle common queries
    additional_info = [
        "TripZoori is your trip tour guide.",
        "The welcome message of TripZoori is 'Your trip tour guide'.",
        "Sarova hotels offer luxury accommodations with prices starting from $150 per night.",
        "TripZoori helps you find the best hotels and vacation destinations.",
        "You can book hotels and tours through the TripZoori website."
    ]
    for info in additional_info:
        sent_tokens.append(info)

@app.on_event("startup")
async def startup_event():
    """Initialize the chatbot when the app starts"""
    initialize_bot()

@app.get("/initialize")
async def initialize_chatbot(url: str = Query("https://tripzoori-gittest1.fly.dev/")):
    """Initialize or reinitialize the chatbot with custom URL"""
    global website_url, raw, sent_tokens, word_tokens
    try:
        # Update URL if provided
        website_url = url
        # Reset tokens
        raw = ""
        sent_tokens = []
        word_tokens = []
        # Initialize
        initialize_bot()
        return {
            "status": "success",
            "message": "Chatbot initialized successfully",
            "data_size": len(raw)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize chatbot: {str(e)}")

@app.get("/chat")
async def chat(message: str = Query(...)):
    """Process a chat message and return a response"""
    global raw
    if not raw:
        # Initialize if not already done
        initialize_bot()
    user_message = message
    # Check for greeting
    greeting_response = greeting(user_message)
    if greeting_response:
        return {
            "status": "success",
            "response": greeting_response
        }
    # Check for thank you message
    if user_message.lower() in ['thanks', 'thank you']:
        return {
            "status": "success",
            "response": "You're welcome! Have a great day!"
        }
    # Check for goodbye
    if user_message.lower() in ['bye', 'goodbye']:
        return {
            "status": "success",
            "response": "Goodbye! Have a wonderful trip!"
        }
    # Generate response
    bot_response = generate_response(user_message)
    return {
        "status": "success",
        "response": bot_response
    }

@app.get("/health")
async def health_check():
    """Check if the API is running"""
    return {
        "status": "healthy",
        "message": "TripZoori Chatbot API is running"
    }

# Optional CORS middleware if needed
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
