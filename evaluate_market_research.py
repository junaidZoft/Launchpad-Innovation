import os
from groq import Groq
from dotenv import load_dotenv
import json

# Load environment variables from .env
load_dotenv()

# Get API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# rubrics.py

Y_AXIS_CATEGORIES = [
    "Not Relevant to the Idea ",
    "Market Size Data",
    "Growth Trends",
    "Market Data Sources",
    "Customer Needs Insights",
    "Competitive Landscape Overview",
    "Market Size Data + Growth Trends",
    "Market Size Data + Market Data Sources",
    "Market Size Data + Customer Needs Insights",
    "Market Size Data + Competitive Landscape Overview",
    "Growth Trends + Market Data Sources",
    "Growth Trends + Customer Needs Insights",
    "Growth Trends + Competitive Landscape Overview",
    "Market Data Sources + Customer Needs Insights",
    "Market Data Sources + Competitive Landscape Overview",
    "Customer Needs Insights + Competitive Landscape Overview",
    "Market Size Data + Growth Trends + Market Data Sources",
    "Market Size Data + Growth Trends + Customer Needs Insights",
    "Market Size Data + Growth Trends + Competitive Landscape Overview",
    "Market Size Data + Market Data Sources + Customer Needs Insights",
    "Market Size Data + Market Data Sources + Competitive Landscape Overview",
    "Market Size Data + Customer Needs Insights + Competitive Landscape Overview",
    "Growth Trends + Market Data Sources + Customer Needs Insights",
    "Growth Trends + Market Data Sources + Competitive Landscape Overview",
    "Growth Trends + Customer Needs Insights + Competitive Landscape Overview",
    "Market Data Sources + Customer Needs Insights + Competitive Landscape Overview",
    "Market Size Data + Growth Trends + Market Data Sources + Customer Needs Insights",
    "Market Size Data + Growth Trends + Market Data Sources + Competitive Landscape Overview",
    "Market Size Data + Growth Trends + Customer Needs Insights + Competitive Landscape Overview",
    "Market Size Data + Market Data Sources + Customer Needs Insights + Competitive Landscape Overview",
    "Growth Trends + Market Data Sources + Customer Needs Insights + Competitive Landscape Overview",
    "All five criteria combined: Market Size Data + Growth Trends + Market Data Sources + Customer Needs Insights + Competitive Landscape Overview"
]


X_AXIS_CATEGORIES = [
    "Not Relevant to the Idea ",
    "Does not have grammar",
    "Does not Demonstrate Understanding Only",
    "Is not Precise and To the Point",
    "Info is not Well-Structured and Is not Easy to Understand",
    "Does not have Grammar + Does not Demonstrate Understanding",
    "Does not have Grammar + Is not Precise and To the Point",
    "Does not have Grammar + Info is not Well-Structured and Is not Easy to Understand",
    "Does not Demonstrate Understanding + Info is not Well-Structured and Is not Easy to Understand",
    "Does not Demonstrate Understanding + Is not Precise and To the Point",
    "Is not Precise and To the Point + Info is not Well-Structured and Is not Easy to Understand",
    "Does not have Grammar + Does not Demonstrate Understanding + Is not Precise and To the Point",
    "Does not have Grammar + Does not Demonstrate Understanding + Info is not Well-Structured and Is not Easy to Understand",
    "Does not have Grammar + Is not Precise and To the Point + Info is not Well-Structured and Is not Easy to Understand",
    "Does not Demonstrate Understanding + Is not Precise and To the Point + Info is not Well-Structured and Is not Easy to Understand",
    "Does not have Grammar + Does not Demonstrate Understanding + Is not Precise and To the Point + Info is not Well-Structured and Is not Easy to Understand",
    "Has some grammar",
    "Demonstrates some Understanding",
    "Is somewhat Precise and To the Point",
    "Info is somewhat Well-Structured and fairly Easy to Understand",
    "Has some Grammar + Demonstrates some Understanding",
    "Has some Grammar + is somewhat Precise and To the Point",
    "Has some Grammar + Info is somewhat Well-Structured and fairly Easy to Understand",
    "Demonstrates some Understanding + Info is somewhat Well-Structured and fairly Easy to Understand",
    "Demonstrates some Understanding + is somewhat Precise and To the Point",
    "Is somewhat Precise and To the Point + Info is somewhat Well-Structured and is somewhat Easy to Understand",
    "Has some Grammar + Demonstrates some Understanding + is somewhat Precise and To the Point",
    "Has some Grammar + Demonstrates some Understanding + Info is somewhat Well-Structured and somewhat Easy to Understand",
    "Has some Grammar + is somewhat Precise and To the Point + Info is somewhat Well-Structured and is somewhat Easy to Understand",
    "Demonstrates some Understanding + is somewhat Precise and To the Point + Info is somewhat Well-Structured and is somewhat Easy to Understand",
    "Has some Grammar + Demonstrates some Understanding + is somewhat Precise and To the Point + Info is somewhat Well-Structured and somewhat Easy to Understand",
    "Has Very good Grammar + Demonstrates Very good Understanding",
    "Has Very good Grammar + Is Precise and To the Point",
    "Has Very good Grammar + Info is Well-Structured and Easy to Understand",
    "Demonstrates Very good Understanding + Is Precise and To the Point",
    "Demonstrates Very Good Understanding + Info is Very Well-Structured and Easy to Understand",
    "Is Precise and To the Point + Info is Very Well-Structured and Easy to Understand",
    "Has very good Grammar + Demonstrates Very Good Understanding + Is Precise and To the Point",
    "Has Very Good Grammar + Demonstrates Very Good Understanding + Info is Very Well-Structured and Easy to Understand",
    "Has Very Good Grammar + Is Precise and To the Point + Info is Very Well-Structured and Easy to Understand",
    "Demonstrates Very Good Understanding + Is Precise and To the Point + Info is Very Well-Structured and Easy to Understand",
    "Has Very Good Grammar + Demonstrates Very Good Understanding + Is Precise and To the Point + Info is Very Well-Structured and Easy to Understand"
]

system_prompt = f"""
You are an expert in market research.

Your task is to classify a free-text market research statement into:

1. Exactly one **X-axis category**, chosen strictly from this list:
{X_AXIS_CATEGORIES}

2. Exactly one **Y-axis category**, chosen strictly from this list:
{Y_AXIS_CATEGORIES}

⚠️ VERY IMPORTANT RULES:
- You MUST choose exactly one value from each list.
- NEVER respond with "none", "not applicable", "n/a", or anything outside these lists.
- If unsure, choose the closest matching category — do NOT return "None".

❌ Never create your own category or explanation.
❌ Never leave a field blank.

Return only valid JSON in this **strict format**:
{{
  "x_category": "<one item from the X list>",
  "y_category": "<one item from the Y list>"
}}
"""


def categorize_market_research(input_text: str) -> dict:
    """
    Categorize a market research statement using Groq API.
    
    Args:
        input_text (str): The market research statement to categorize
        
    Returns:
        dict: JSON response containing x_category and y_category
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set. Please set it in your .env file or environment.")

    if not input_text.strip():
        raise ValueError("Please enter a valid input text.")

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text}
            ]
        )
        output = response.choices[0].message.content
        return json.loads(output)

    except Exception as e:
        raise Exception(f"Error during categorization: {str(e)}")

# Example usage:
if __name__ == "__main__":
    try:
        sample_text = "The global smart home market size was valued at USD 84.5 billion in 2021 and is expected to expand at a compound annual growth rate (CAGR) of 7.1% from 2022 to 2030."
        result = categorize_market_research(sample_text)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")