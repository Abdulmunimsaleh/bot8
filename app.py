from fastapi import FastAPI, HTTPException, Query
from typing import Optional
import requests
from bs4 import BeautifulSoup
import re

app = FastAPI()

# Constants
WEBSITE_URL = "https://tripzoori-gittest1.fly.dev/"

def scrape_website(url: str = WEBSITE_URL) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Focus on main content areas (adjust selectors based on your website structure)
        main_content = soup.find_all(['main', 'article', 'div', 'p'])
        
        # Get text and clean it
        text_content = []
        for content in main_content:
            text = content.get_text(strip=True)
            if text:
                text_content.append(text)
        
        # Join and clean the text
        text = ' '.join(text_content)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error scraping website: {str(e)}")

def process_message(message: str, context: Optional[str] = None) -> str:
    message = message.lower()
    
    # If we have context from the website
    if context:
        sentences = re.split(r'[.!?]+', context)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Keywords for specific types of information
        if any(word in message for word in ["trip", "travel", "destination", "tour"]):
            relevant = [s for s in sentences if any(word in s.lower() for word in ["trip", "travel", "destination", "tour"])]
            if relevant:
                return " ".join(relevant)
        
        if any(word in message for word in ["price", "cost", "fee", "payment"]):
            relevant = [s for s in sentences if any(word in s.lower() for word in ["price", "cost", "$", "fee", "payment"])]
            if relevant:
                return " ".join(relevant)
        
        # General search in context
        relevant_sentences = [s for s in sentences if any(word in s.lower() for word in message.split())]
        if relevant_sentences:
            return " ".join(relevant_sentences)
    
    # Default responses
    if "hello" in message or "hi" in message:
        return "Hello! I'm the TripZoori assistant. How can I help you with your travel plans today?"
    elif "how are you" in message:
        return "I'm ready to help you plan your perfect trip! What would you like to know about our travel services?"
    elif "bye" in message:
        return "Thank you for considering TripZoori for your travel needs. Have a great day!"
    else:
        return "I can help you with information about our travel services, destinations, and pricing. Could you please be more specific about what you'd like to know?"

@app.get("/")
def read_root():
    return {"message": "Welcome to the TripZoori Chatbot API"}

@app.get("/chat")
def chat(text: str = Query(..., description="User's question")):
    website_data = scrape_website()
    response = process_message(text, website_data)
    
    return {"answer": response, "source_data": website_data if website_data else None}
