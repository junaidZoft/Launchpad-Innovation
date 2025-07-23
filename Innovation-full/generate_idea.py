import os
import json
import logging
from typing import List, Dict, Any, Optional
from groq import Groq
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from a .env file
load_dotenv()

def generate_sdg_ideas_json(
    selected_sdgs: List[str],
    model: str = 'llama3-8b-8192',
    max_retries: int = 3
) -> Optional[List[Dict[str, Any]]]:
    """
    Generates innovative project ideas for selected SDGs using the Groq API
    and returns them as a JSON object (Python list of dictionaries).

    Args:
        selected_sdgs: A list of SDG names to focus on.
        model: The Groq model to use for generation.
        max_retries: The maximum number of times to retry the API call on failure.

    Returns:
        A list of dictionaries, where each dictionary represents a project idea,
        or None if the generation fails.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY not found. Please set it in your .env file.")
        return None

    client = Groq(api_key=api_key)

    # Optimized prompt to request a JSON response
    prompt = f"""
    Generate 5 innovative, student-friendly project ideas for the following UN Sustainable Development Goals (SDGs): {', '.join(selected_sdgs)}.

    Your response MUST be a valid JSON array of objects. Do not include any text or explanations outside of the JSON array.
    
    Each object in the array should represent a single project idea and have the following structure:
    - "title": A short, catchy title for the project.
    - "description": A detailed description (2-3 sentences) of the project. Explain how it addresses the selected SDGs, its feasibility for students aged 14-18, and its potential impact.
    - "technology_focus": The key technology or social innovation involved (e.g., 'Mobile App', 'Data Analysis Platform', 'Community-led Initiative', 'IoT Device').
    - "sdgs_addressed": A list of the specific SDG names it targets from the input list.

    Example of a single JSON object in the array:
    {{
      "title": "Aqua-Alert: Smart Water Quality Monitoring",
      "description": "A project to build low-cost, IoT-based sensors that monitor local water quality in real-time. Students can assemble the sensors and develop a mobile app to visualize the data, alerting communities to pollution and contributing to SDG 6.",
      "technology_focus": "IoT Device",
      "sdgs_addressed": ["Clean Water and Sanitation"]
    }}
    
    Now, generate the JSON array for these SDGs: {', '.join(selected_sdgs)}.
    """

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}: Generating project ideas for SDGs: {selected_sdgs}")
            
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model,
                temperature=0.7,
                max_tokens=2048,
                top_p=1,
                stop=None,
                response_format={"type": "json_object"}, # Use JSON mode if supported by model/API
            )
            
            response_text = chat_completion.choices[0].message.content
            
            # The response should be a string containing a JSON object.
            # The prompt asks for an array, which might be nested under a key like "ideas".
            parsed_json = json.loads(response_text)

            # Check if the response is a dict containing the list, or the list itself
            if isinstance(parsed_json, dict):
                 # Try to find the list within the dict, e.g., under a key like 'ideas' or 'projects'
                for key, value in parsed_json.items():
                    if isinstance(value, list):
                        return value # Return the first list found
            elif isinstance(parsed_json, list):
                return parsed_json # The response is the list itself

            logger.error(f"JSON response was not in the expected format (list of ideas). Response: {response_text}")
            return None


        except json.JSONDecodeError as e:
            logger.error(f"Attempt {attempt + 1}: Failed to decode JSON from response. Error: {e}")
            logger.debug(f"Invalid response text: {chat_completion.choices[0].message.content}")
        except Exception as e:
            logger.error(f"Attempt {attempt + 1}: An unexpected error occurred: {e}")

    logger.error("Failed to generate ideas after multiple retries.")
    return None

if __name__ == "__main__":
    # --- Example Usage ---
    
    # 1. Define the SDGs you are interested in
    my_sdgs = ["Quality Education", "Reduced Inequalities"]
    
    # 2. Call the function to generate ideas
    project_ideas = generate_sdg_ideas_json(my_sdgs)
    
    # 3. Process the result
    if project_ideas:
        print("✅ Successfully generated project ideas!\n")
        # Pretty-print the JSON output
        print(json.dumps(project_ideas, indent=2))
    else:
        print("❌ Failed to generate project ideas.")