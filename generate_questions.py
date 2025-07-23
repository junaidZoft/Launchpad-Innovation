import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def generate_survey_questions(idea, problem_statement, sdgs):
    """
    Generate 5 Yes/No survey questions using Groq API for business idea validation.
    
    Args:
        idea (str): The business idea description
        problem_statement (str): The problem the idea aims to solve
        sdgs (str or list): Sustainable Development Goals (can be string or list)
    
    Returns:
        list: List of 5 Yes/No survey questions as strings
    """
    
    # Load Groq API key from environment variables
    groq_api_key = os.getenv('GROQ_API_KEY')
    
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please add it to your .env file.")
    
    client = Groq(api_key=groq_api_key)
    
    # Handle SDGs input (convert list to string if needed)
    if isinstance(sdgs, list):
        sdgs_str = ", ".join(sdgs)
    else:
        sdgs_str = str(sdgs)
    
    # Create the prompt with emphasis on Yes/No questions
    prompt = f"""You're helping a student prepare for a pitch competition.

They want to ask 5 Yes/No survey questions to get feedback from people (friends, family, or target market) about their idea.

🧠 Idea: {idea}  
📝 Problem Statement: {problem_statement}  
🎯 SDGs: {sdgs_str}

Generate exactly 5 simple, friendly YES/NO survey questions that test if the idea is needed, understandable, and exciting.

IMPORTANT REQUIREMENTS:
- Each question MUST be answerable with only "Yes" or "No"
- Questions should NOT ask for ratings, scales, or open-ended responses
- Questions should be clear and direct
- Keep questions short and easy to understand

Examples of good Yes/No questions:
- "Do you think this problem needs to be solved?"
- "Would you use this product/service?"
- "Is this idea easy to understand?"

Only return the 5 questions as a numbered list with no title, no intro.
Each question should be on a new line starting with a number."""

    try:
        # Make API call to Groq
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama3-8b-8192",  # You can change this to other Groq models like "mixtral-8x7b-32768"
            temperature=0.7,
            max_tokens=500
        )
        
        # Extract and clean the response
        response_text = chat_completion.choices[0].message.content.strip()
        
        # Parse questions from the response
        questions = []
        for line in response_text.split('\n'):
            line = line.strip()
            if line:
                # Remove numbering and bullet points
                cleaned_question = line
                for prefix in ['1.', '2.', '3.', '4.', '5.', '•', '-', '*']:
                    if cleaned_question.startswith(prefix):
                        cleaned_question = cleaned_question[len(prefix):].strip()
                        break
                
                # Remove any remaining numbering patterns
                import re
                cleaned_question = re.sub(r'^\d+\.\s*', '', cleaned_question)
                
                if cleaned_question:
                    questions.append(cleaned_question)
        
        # Ensure we have exactly 5 questions
        questions = questions[:5]
        
        # If we don't have 5 questions, pad with generic Yes/No questions
        while len(questions) < 5:
            generic_yes_no_questions = [
                "Do you think this idea would solve a real problem?",
                "Would you be interested in using this product or service?",
                "Is this idea easy to understand?",
                "Do you believe this idea is innovative and unique?",
                "Would you recommend this idea to your friends or family?"
            ]
            if len(questions) < len(generic_yes_no_questions):
                questions.append(generic_yes_no_questions[len(questions)])
            else:
                break
        
        return questions
        
    except Exception as e:
        raise Exception(f"Error generating questions with Groq API: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Example inputs
    sample_idea = "A mobile app that helps kids learn healthy eating habits through gamification"
    sample_problem = "Many children struggle with making healthy food choices and lack knowledge about nutrition"
    sample_sdgs = ["Good Health and Well-being", "Quality Education"]
    
    # Note: Make sure to create a .env file with GROQ_API_KEY=your_api_key_here
    
    try:
        questions = generate_survey_questions(
            idea=sample_idea,
            problem_statement=sample_problem,
            sdgs=sample_sdgs
        )
        
        print("Generated Survey Questions:")
        for i, question in enumerate(questions, 1):
            print(f"{i}. {question}")
            
    except Exception as e:
        print(f"Error: {e}")