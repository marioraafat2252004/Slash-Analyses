import json
import os
import google.generativeai as genai
from utils.csvLoader import load_csv_data
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
csv_data = load_csv_data()

if not csv_data:
    raise RuntimeError("Failed to load CSV files. Ensure all files are correctly formatted.")

# Initialize chat session
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
]

# Update the prompt
system_instruction = f"""
You are an intelligent assistant for an e-commerce platform. Your responsibilities are:

1. Identify whether the user's input is a casual conversation or a product-related query.
2. For casual conversations, respond with friendly and appropriate replies in JSON format:
   {{
     "intent": "casual_message",
     "response": "Friendly reply to the user's message"
   }}

3. For product-related queries:
   - Analyze the input and extract relevant details such as product type, color, category, or attributes from our database {csv_data}.
   - Match the user's query against the provided list of products: {csv_data["products"]}.
   - Recommend exactly 5 products that match the query, sorted by relevance, in the following JSON format:
   {{
     "intent": "product_search",
     "response": "Friendly reply to the user's message",
     "recommendations": {{
       "colours": [], 
       "materials": [], 
       "categories": [], 
       "styles": [], 
       "brands": [], 
       "tags": []
     }},
     "recommendation_count": 5,
     recommended_products_Ids: [id1, id2, ...],
   }}

4. If the query does not match any products, provide alternatives using the closest attributes (e.g., similar color or category).
5. Ensure all recommendations are based solely on the provided data.
"""

# Create the model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b",
    safety_settings=safety_settings,
    generation_config=generation_config,
    system_instruction=system_instruction,
)

chat_sessions = {}


def handle_chat(user_id: str, user_input: str) -> dict:
    if user_id not in chat_sessions:
        chat_sessions[user_id] = model.start_chat(history=[])

    session = chat_sessions[user_id]
    response = session.send_message(user_input)
    session.history.append({"role": "user", "parts": [user_input]})
    session.history.append({"role": "model", "parts": [response.text]})

    try:
        # Clean up the response text
        cleaned_response = response.text.strip("```json\n").strip("```").strip()
        # Parse the cleaned response into JSON
        response_json = json.loads(cleaned_response)
        return response_json
    except json.JSONDecodeError as e:
        # Log the invalid response for debugging
        print(f"JSONDecodeError: {str(e)}")
        print(f"Raw response: {response.text}")
        # Raise an exception with the invalid response
        raise ValueError(f"Invalid JSON response: {response.text}")
