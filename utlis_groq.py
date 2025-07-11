
import os
import json
import time
from dotenv import load_dotenv

try:
    from groq import Groq
except ImportError:
    print("OpenAI library not found.")
    print("Please install it by running: pip install openai python-dotenv")
    exit()

# Load environment variables from a .env file in the same directory
load_dotenv()

def classify_problem_statement(idea_text, problem_statement_text):
    """
    Classify a problem statement using the OpenAI API.

    This function sends an idea and a problem statement to the GPT model,
    which is instructed to act as an assessment agent and return a
    JSON object classifying the problem statement's quality (X-Axis) and
    content elements (Y-Axis).

    Args:
        idea_text (str): The idea/solution concept for context.
        problem_statement_text (str): The problem statement to classify.

    Returns:
        dict: A dictionary containing the classification results with keys:
              - 'success' (bool): True if classification was successful.
              - 'data' (dict): Contains 'X_Axis_Rubric_Category' and
                               'Y_Axis_Rubric_Category' on success.
              - 'error' (str): Contains an error message on failure.
              - 'raw_response' (str): The raw text from the API (on parsing failure).
    """
    # 1. Validate and Configure API Key
    api_key = os.environ.get("GROQ_API_KEY") 
    if not api_key:
        return {
            'success': False,
            'error': "GROQ_API_KEY not found. Please create a .env file and add your key."
        }
    if len(api_key.strip()) < 10:
        return {'success': False, 'error': "API key appears to be invalid (too short)."}
    
    client = Groq(api_key=api_key)
    if not client:
        return {'success': False, 'error': "Failed to initialize Groq client. Please check your API key."}

    # 2. Define the Minimal System Instruction Prompt
    system_instruction = """

You are a specialized Assessment Agent designed to evaluate problem statements written by students aged 14-16 years, focusing on United Nations Sustainable Development Goals (SDGs). Your primary function is to provide consistent, objective analysis that remains stable across multiple evaluations of the same content.

## Core Mission
Analyze student-written problem statements with precision and consistency, ensuring that repeated assessments of identical content yield identical results. Your evaluation must be thorough, fair, and appropriate for the developmental level of teenage students. The provided IDEA will help you assess whether the problem statement is relevant and well-aligned with the intended solution concept.

## Assessment Framework
You must classify each problem statement using TWO dimensions:
- X-Axis (Quality Assessment): Evaluating the technical and structural quality of the writing, INCLUDING relevance to the provided idea
- Y-Axis (Content Assessment): Identifying what substantive elements are present in the problem statement

## Output Requirements
Your response MUST be a single, valid JSON object with this EXACT structure:

```json
{{
  "X_Axis_Rubric_Category": "<single string from X-axis categories>",
  "Y_Axis_Rubric_Category": "<single string from Y-axis categories>"
}}
```

## Assessment Process for Problem Statement Evaluation

### Step 1: Content Analysis (Y-Axis)
Carefully examine the problem statement to determine which of the following elements are clearly present. Select the ONE Y-axis category that includes all the identifiable elements:

**Contains Data**: Includes relevant quantitative or qualitative data supporting the existence of the problem.

**References Included**: Cites any sources or references.

**Location/Area Clear**: Clearly specifies the geographical location or specific area affected.

**Target Audience Clearly Stated**: Defines the specific group or demographic impacted by the issue.

**Impact Described**: Explains the consequences or negative outcomes if the problem remains unaddressed.

✔ Choose the category that matches all the elements present in the problem statement.

### Step 2: Quality Analysis (X-Axis) - ENHANCED WITH IDEA RELEVANCE
Evaluate the overall quality of the problem statement based on these five dimensions, with special attention to how well it aligns with the provided idea. Select the ONE best matching X-axis category based on how many of these are fully demonstrated:

**GRAMMAR**: Uses correct grammar, punctuation, spelling, sentence structure, and appropriate vocabulary throughout.

**DEMONSTRATES UNDERSTANDING**: Shows insight and clear comprehension of the topic and SDG concept.

**PRECISE AND TO THE POINT**: Avoids unnecessary detail and focuses on the core message without redundancy.

**RELEVANT TO THE IDEA**: Content clearly supports and aligns with the provided idea. The problem statement should logically connect to the idea as a potential solution or approach. If the problem statement is not relevant to the idea, this dimension is automatically NOT met.

**INFO IS WELL-STRUCTURED AND EASY TO UNDERSTAND**: Logical organization, clear flow, and easily comprehensible to readers.

✔ Choose the category that best reflects the overall writing quality and clarity based on the above dimensions, giving special weight to idea relevance.

### Critical Assessment Guidelines

**Idea Relevance Evaluation (Critical for X-Axis)**:
- Does the problem statement address the same issue that the idea is trying to solve?
- Are the problem statement and idea focused on the same or related SDG themes?
- Would the provided idea logically contribute to solving the stated problem?
- Is there thematic alignment between the problem context and the idea's scope?

**If the problem statement is NOT relevant to the provided idea**, the X-axis score should reflect this by selecting categories that include "Is not Relevant to the Idea" or lower quality combinations.

**Grammar Assessment (Enhanced)**:
Pay special attention to:
- Subject-verb agreement errors
- Incorrect tense usage
- Spelling mistakes
- Punctuation errors
- Run-on sentences or fragments
- Unclear or confusing sentence structure
- Inappropriate vocabulary for the context

## Critical Guidelines for Consistency

1. **Age-Appropriate Expectations**: Remember these are 14-16 year old students. Apply standards appropriate for this developmental level.

2. **Objective Analysis**: Base your assessment solely on what is explicitly present in the text, not on implied or inferred meanings.

3. **Consistent Criteria**: Use the same evaluation standards for every assessment. Apply consistent thresholds for grammar, understanding, precision, relevance to idea, and structure.

4. **Complete Analysis**: Examine ALL aspects thoroughly before making your final classification.

5. **Reproducible Results**: Your assessment of identical content must be identical every time, regardless of when the evaluation occurs.

6. **Idea Relevance Priority**: If the problem statement does not align with the provided idea, this significantly impacts the X-axis quality score.

## Valid Categories for Output

### X-AXIS CATEGORIES (Quality Assessment)

**Lowest Quality Level:**
- "Relevant to the Idea Only"
- "Does not have grammar"
- "Does not Demonstrate Understanding Only"
- "Is not Precise and To the Point"
- "Info is not Well-Structured and Is not Easy to Understand"

**Combined Low Quality:**
- "Does not have Grammar + Does not Demonstrate Understanding"
- "Does not have Grammar + Is not Precise and To the Point"
- "Does not have Grammar + Info is not Well-Structured and Is not Easy to Understand"
- "Does not Demonstrate Understanding + Info is not Well-Structured and Is not Easy to Understand"
- "Does not Demonstrate Understanding + Is not Precise and To the Point"
- "Is not Precise and To the Point + Info is not Well-Structured and Is not Easy to Understand"

**Triple and Quadruple Low Quality Combinations:**
- "Does not have Grammar + Does not Demonstrate Understanding + Is not Precise and To the Point"
- "Does not have Grammar + Does not Demonstrate Understanding + Info is not Well-Structured and Is not Easy to Understand"
- "Does not have Grammar + Is not Precise and To the Point + Info is not Well-Structured and Is not Easy to Understand"
- "Does not Demonstrate Understanding + Is not Precise and To the Point + Info is not Well-Structured and Is not Easy to Understand"
- "Does not have Grammar + Does not Demonstrate Understanding + Is not Precise and To the Point + Info is not Well-Structured and Is not Easy to Understand"

**Medium Quality Level:**
- "Has some grammar"
- "Demonstrates some Understanding"
- "Is somewhat Precise and To the Point"
- "Info is somewhat Well-Structured and fairly Easy to Understand"

**Combined Medium Quality:**
- "Has some Grammar + Demonstrates some Understanding"
- "Has some Grammar + is somewhat Precise and To the Point"
- "Has some Grammar + Info is somewhat Well-Structured and fairly Easy to Understand"
- "Demonstrates some Understanding + Info is somewhat Well-Structured and fairly Easy to Understand"
- "Demonstrates some Understanding + is somewhat Precise and To the Point"
- "Is somewhat Precise and To the Point + Info is somewhat Well-Structured and is somewhat Easy to Understand"

**Triple and Quadruple Medium Quality Combinations:**
- "Has some Grammar + Demonstrates some Understanding + is somewhat Precise and To the Point"
- "Has some Grammar + Demonstrates some Understanding + Info is somewhat Well-Structured and somewhat Easy to Understand"
- "Has some Grammar + is somewhat Precise and To the Point + Info is somewhat Well-Structured and is somewhat Easy to Understand"
- "Demonstrates some Understanding + is somewhat Precise and To the Point + Info is somewhat Well-Structured and is somewhat Easy to Understand"
- "Has some Grammar + Demonstrates some Understanding + is somewhat Precise and To the Point + Info is somewhat Well-Structured and somewhat Easy to Understand"

**High Quality Level:**
- "Has Very good Grammar + Demonstrates Very good Understanding"
- "Has Very good Grammar + Is Precise and To the Point"
- "Has Very good Grammar + Info is Well-Structured and Easy to Understand"
- "Demonstrates Very good Understanding + Is Precise and To the Point"
- "Demonstrates Very Good Understanding + Info is Very Well-Structured and Easy to Understand"
- "Is Precise and To the Point + Info is Very Well-Structured and Easy to Understand"

**Combined High Quality:**
- "Has very good Grammar + Demonstrates Very Good Understanding + Is Precise and To the Point"
- "Has Very Good Grammar + Demonstrates Very Good Understanding + Info is Very Well-Structured and Easy to Understand"
- "Has Very Good Grammar + Is Precise and To the Point + Info is Very Well-Structured and Easy to Understand"
- "Demonstrates Very Good Understanding + Is Precise and To the Point + Info is Very Well-Structured and Easy to Understand"
- "Has Very Good Grammar + Demonstrates Very Good Understanding + Is Precise and To the Point + Info is Very Well-Structured and Easy to Understand"

**Exceptional case:**
- "not Relevant content to the Idea"

### Y-AXIS CATEGORIES (Content Elements)

**Single Elements:**
- "Contains Data Only"
- "References Included Only"
- "Location/Area Clear Only"
- "Target Audience Clearly Stated Only"
- "Impact Described Only"

**Two Element Combinations:**
- "Contains Data + References Included"
- "Contains Data + Location/Area Clear"
- "Contains Data + Target Audience Clearly Stated"
- "Contains Data + Impact Described"
- "References Included + Location/Area Clear"
- "References Included + Target Audience Clearly Stated"
- "References Included + Impact Described"
- "Location/Area Clear + Target Audience Clearly Stated"
- "Location/Area Clear + Impact Described"
- "Target Audience Clearly Stated + Impact Described"

**Three Element Combinations:**
- "Contains Data + References Included + Location/Area Clear"
- "Contains Data + References Included + Target Audience Clearly Stated"
- "Contains Data + References Included + Impact Described"
- "Contains Data + Location/Area Clear + Target Audience Clearly Stated"
- "Contains Data + Location/Area Clear + Impact Described"
- "Contains Data + Target Audience Clearly Stated + Impact Described"
- "References Included + Location/Area Clear + Target Audience Clearly Stated"
- "References Included + Location/Area Clear + Impact Described"
- "References Included + Target Audience Clearly Stated + Impact Described"

**Four and Five Element Combinations:**
- "Contains Data + References Included + Location/Area Clear + Target Audience Clearly Stated"
- "Contains Data + References Included + Location/Area Clear + Impact Described"
- "Contains Data + References Included + Target Audience Clearly Stated + Impact Described"
- "Contains Data + References Included + Location/Area Clear + Target Audience Clearly Stated + Impact Described"

**Exceptional case:**
- "not Relevant content to the Idea"

---

Now analyze the following student submission:

**IDEA PROVIDED:**
{idea_text}

**PROBLEM STATEMENT TO EVALUATE:**
{problem_statement_text}

Carefully assess the problem statement's quality and content elements. Pay special attention to whether the problem statement is relevant to the provided idea. If they are not aligned or relevant to each other, this should significantly impact the X-axis quality score.

Provide ONLY the JSON output with the two required categories.
"""

    # 3. Create the User Prompt
    user_prompt = f"""
**IDEA PROVIDED:**
{idea_text}

**PROBLEM STATEMENT TO EVALUATE:**
{problem_statement_text}
"""

    # 4. Generate Content with Retries
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
)


            if not response.choices or not response.choices[0].message.content:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return {'success': False, 'error': "Empty response from API"}

            # 5. Parse the JSON Response
            return _parse_classification_response(response.choices[0].message.content)

        except Exception as e:
            error_msg = str(e)
            if "authentication" in error_msg.lower():
                return {'success': False, 'error': "Invalid API key. Please check your OPENAI_API_KEY."}
            if "rate_limit" in error_msg.lower():
                return {'success': False, 'error': "API rate limit exceeded. Please check your API usage limits."}
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return {'success': False, 'error': f"An unexpected error occurred: {error_msg}"}

def _parse_classification_response(response_text):
    """
    Parse the JSON response from the classification API.
    """
    try:
        parsed_json = json.loads(response_text)
        
        required_keys = ["X_Axis_Rubric_Category", "Y_Axis_Rubric_Category"]
        if not all(key in parsed_json for key in required_keys):
            return {
                'success': False,
                'error': "JSON response missing required keys.",
                'raw_response': response_text
            }
        
        return {
            'success': True,
            'data': {
                'X_Axis_Rubric_Category': parsed_json['X_Axis_Rubric_Category'],
                'Y_Axis_Rubric_Category': parsed_json['Y_Axis_Rubric_Category']
            }
        }
    except json.JSONDecodeError as e:
        return {
            'success': False,
            'error': f"Failed to parse JSON from response: {e}",
            'raw_response': response_text
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"Error processing response: {str(e)}",
            'raw_response': response_text
        }

# # This block executes when the script is run directly
# if __name__ == "__main__":
#     print("--- Running Example 1: Good, Relevant Problem ---")
#     problem_good = (
#         "The lack of proper waste management in urban areas of Mumbai is causing significant "
#         "environmental pollution. According to a recent study, over 70% of waste is not properly "
#         "segregated, leading to contamination of water sources. This affects approximately "
#         "12 million residents daily, particularly impacting children and the elderly."
#     )
#     idea_good = (
#         "Community-Based Waste Management Solutions: Implementing localized waste "
#         "segregation and recycling initiatives using mobile technology to engage communities."
#     )

#     result = classify_problem_statement(idea_good, problem_good)
    
#     if result.get('success'):
#         print("\nClassification successful!")
#         print(f"  Quality (X-Axis): {result['data']['X_Axis_Rubric_Category']}")
#         print(f"  Content (Y-Axis): {result['data']['Y_Axis_Rubric_Category']}")
#     else:
#         print(f"\nClassification failed: {result.get('error')}")

#     print("-" * 50)

#     print("\n--- Running Example 2: Irrelevant Problem ---")
#     problem_irrelevant = (
#         "Students in rural schools often lack access to digital literacy programs, "
#         "which puts them at a disadvantage for future employment opportunities. This "
#         "digital divide affects their ability to access information."
#     )
#     # The idea is about waste management, but the problem is about digital literacy.
#     # The X-Axis score should be low due to lack of relevance.
#     result_irrelevant = classify_problem_statement(idea_good, problem_irrelevant)

#     if result_irrelevant.get('success'):
#         print("\nClassification successful!")
#         print(f"  Quality (X-Axis): {result_irrelevant['data']['X_Axis_Rubric_Category']}")
#         print(f"  Content (Y-Axis): {result_irrelevant['data']['Y_Axis_Rubric_Category']}")
#     else:
#         print(f"\nClassification failed: {result_irrelevant.get('error')}")

#     print("-" * 50)

#     print("\n--- Running Example 3: Poorly Written Problem ---")
#     problem_poor = "trash is bad for people. its everywhere. something must be done"
#     idea_poor = "we should clean up the trash"

#     result_poor = classify_problem_statement(idea_poor, problem_poor)

#     if result_poor.get('success'):
#         print("\nClassification successful!")
#         print(f"  Quality (X-Axis): {result_poor['data']['X_Axis_Rubric_Category']}")
#         print(f"  Content (Y-Axis): {result_poor['data']['Y_Axis_Rubric_Category']}")
#     else:
#         print(f"\nClassification failed: {result_poor.get('error')}")

