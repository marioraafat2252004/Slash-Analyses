import json
import os
import tempfile
import google.generativeai as genai
import utils.csvLoader as csvLoader

csv_data = csvLoader.load_csv_analysis_data()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

if not csv_data:
    raise RuntimeError("Failed to load CSV files. Ensure all files are correctly formatted.")

system_instruction = f"""
You are an intelligent assistant specialized in analyzing product images for an e-commerce platform. Your responsibilities are:

1. Analyze the input image and extract relevant features based on the database provided. Focus only on the main product in the image (i.e., the most prominently displayed product). If the image includes multiple products, prioritize analyzing the one that is visually emphasized. Extract the following features:

   - **Tags**: Provide descriptive terms related to the product. Ensure at least 7 tags, try to select only those available in the database (if you need more no matter). Include as many tags as possible.

   - **Style**: Identify the style (e.g., casual, formal, sporty, etc.) from the database. Ensure that the style matches the attributes of the main product. No null values are allowed. Avoid generic styles and be as specific as possible like casual, Be specific if it possible.

   - **Category**: 
        - Determine the specific category from the database. Be as precise as possible, avoiding generic categories (eg. be accurate in the difference between Hoodies and Shirts). No null values are allowed.
        - Restrict the category to the following options: Hoodies, Shirts, Jackets, Shoes, Sweaters, Dresses, Pants, Skirts, Shorts, Bags, Accessories. Don't include any other categories.
   - **Colours**: Provide detailed color information in an array of objects. For each color:
       - Include the family (e.g., "Red", "Blue", etc.), No null values are allowed.
       - Include the specific name (e.g., "Cherry Red", "Sky Blue"), No null values are allowed.
       - Include the Hex code (e.g., "#FF5733"), No null values are allowed.
       - Specify the percentage of the product occupied by this color. Ensure the total of all percentages equals 100.
       - Restrict the family to the basic colors (e.g., Red, Blue, Green, Yellow, Orange, Purple, Pink, Brown, Black, White, Gray). No null values are allowed.

   - **Material**: Identify the product's material (e.g., cotton, leather, polyester, etc.) from the database. Ensure the material is accurate and matches the product. No null values are allowed.

   - **Description**: Generate a concise and well-structured description of the product based on its identified features and attributes. Ensure the description includes all relevant details such as category, style, material, colors, and any other significant attributes.

2. Match the extracted attributes against the provided database: {csv_data}. Use only the data available in the database to identify styles, categories, materials, and other attributes. Do not include any information not present in the database.

3. Recommend a price for the product in Egyptian Pounds within the range [150 - 900], be diverse. Base the price on the product's attributes, including material, category, and style. Range price for each category:
    - Hoodies: [400 - 600]
    - Shirts: [300 - 600]
    - Jackets: [600 - 1450]
    - Shoes: [350 - 600]
    - Sweaters: [400 - 600]
    - Dresses: [600 - 1200]
    - Pants: [350 - 600]
    - Skirts: [400 - 550]
    - Shorts: [400 - 600]
    - Bags: [350 - 800]
    - Accessories: As diverse as possible.

4. Provide the output in the following JSON format:
   {{
     "name": "Product name based on the identified features and database information",
     "price": "Recommended price in Egyptian pounds, no null values",
     "tags": ["tag1", "tag2", ...],
     "style": "Identified style from the database",
     "category": "Identified category from the database",
     "colours": [
       {{"family": "color_family", "name": "specific_name", "hex": "#hex_code", "percentage": "e.g., 70"}},
       ...
     ],
     "material": "Identified material from the database",
     "description": "Concise and well-structured description including all attributes."
   }}

5. If the image cannot be matched exactly to a product in the database, provide attributes based on the closest matches available (e.g., similar style, category, or colors). Ensure the output is still relevant and accurate.

6. Ensure the analysis is detailed, accurate, and strictly adheres to the attributes available in the provided database. Avoid null or generic values, and ensure all features align with the database.

7. Ensure the analysis is for only one product per image, focusing on the main product (the one most visually emphasized). Ignore secondary products in the image.

8. Provide a timely response with accurate and relevant information. Ensure all outputs are well-structured and meet the requirements outlined above.
"""


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

# Generate the analysis using the configured model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b",
    safety_settings=safety_settings,
    generation_config=generation_config,
    system_instruction=system_instruction,
)

async def analyze_image_controller(file):
    """
    Controller function to upload the image to Gemini and perform analysis.
    """
    print(f"Received image file: {file.filename}")
    if not file.content_type.startswith("image/"):
        raise ValueError("Uploaded file must be an image.")

    temp_file_path = None
    print("Analyzing the uploaded image...")
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        print(f"Saved the uploaded file to: {temp_file_path}")

        # Upload the file to Gemini
        uploaded_file = genai.upload_file(temp_file_path, mime_type=file.content_type)
        print(f"Uploaded file '{uploaded_file.display_name}' as: {uploaded_file.uri}")

        # Start a chat session with the uploaded image
        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [uploaded_file],
                }
            ]
        )

        # Send a message to analyze the uploaded image
        response = chat_session.send_message("Analyze this image.")

        # Parse the raw response text
        try:
            input_data = json.loads(response.text)  # Deserialize the raw JSON response
        except json.JSONDecodeError:
            raise ValueError("Failed to parse the response text as JSON.")

        # Ensure the "analysis" field is properly deserialized if needed
        if isinstance(input_data.get("analysis"), str):
            input_data["analysis"] = json.loads(input_data["analysis"])

        # Pretty-print the result for readability
        formatted_json = json.dumps(input_data, indent=4)

        return formatted_json

    except Exception as e:
        raise ValueError(f"An error occurred during image analysis: {str(e)}")
    finally:
        # Ensure the temporary file is deleted
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
