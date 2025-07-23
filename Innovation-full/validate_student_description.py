import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
import logging
from groq import Groq

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

def load_api_keys_from_env() -> Dict[str, str]:
    """
    Load API keys from environment variables.
    
    Returns:
        Dict with API keys for Groq and OpenAI
    """
    try:
        keys = {
            'groq': os.getenv("GROQ_API_KEY"),
            'openai': os.getenv("OPENAI_API_KEY")
        }
        
        available_keys = {k: v for k, v in keys.items() if v}
        logger.info(f"Available API keys: {list(available_keys.keys())}")
        
        return keys
    except Exception as e:
        logger.error(f"Error loading API keys from environment: {e}")
        return {'groq': None, 'openai': None}

def create_validation_prompt(idea: str, problem: str, prototype_description: str) -> str:
    """
    Create a focused prompt template for content validation.
    
    Args:
        idea: Student's innovation idea
        problem: Problem statement
        prototype_description: Prototype description
    
    Returns:
        Formatted prompt for Groq API
    """
    
    prompt = f"""You are an educational content validator for a student innovation platform. Your job is to evaluate if student submissions are appropriate, educational, and coherent.

VALIDATION CRITERIA:
1. SAFETY: Content must be appropriate for educational use (no violence, weapons, adult content, illegal activities, hate speech, harmful substances)
2. COHERENCE: The idea, problem, and prototype must be logically related and form a coherent solution
3. EDUCATIONAL VALUE: Content should demonstrate learning, innovation, problem-solving, or positive impact

STUDENT SUBMISSION:
---
IDEA: {idea}

PROBLEM: {problem}

PROTOTYPE DESCRIPTION: {prototype_description}
---

EVALUATION INSTRUCTIONS:
- Analyze the submission against all three criteria
- Check if the idea logically addresses the stated problem
- Verify the prototype description implements the proposed idea
- Ensure all content is safe and appropriate for students
- Look for educational merit and innovation focus

RESPONSE FORMAT (JSON only):
{{
    "valid": true/false,
    "message": "Brief explanation of decision (max 150 characters)"
}}

EXAMPLES:
Valid: Smart recycling bin with AI sorting → Students don't know how to sort waste → IoT bin with camera recognition
Invalid: Dating app → Students need better grades → Social media platform with messaging

Evaluate the submission now:"""

    return prompt

def validate_student_input(
    idea: str,
    problem: str,
    prototype_description: str,
    model: str = "llama-3.3-70b-versatile"  # Updated to current model
) -> Dict[str, Any]:
    """
    Validate student input using Groq API for content safety and coherence.
    
    Args:
        idea (str): The innovative idea or solution concept
        problem (str): The specific problem statement being solved
        prototype_description (str): Detailed prototype description
        model (str): Groq model to use for validation
    
    Returns:
        Dict containing:
        - valid (bool): Whether the input passes validation
        - message (str): Explanation of the decision
        - error (str): Error message if API call fails
        - api_available (bool): Whether required APIs are available
    """
    
    # Load API keys
    api_keys = load_api_keys_from_env()
    
    if not api_keys['groq']:
        return {
            "valid": False,
            "message": "Groq API key not found in environment variables",
            "error": "Missing GROQ_API_KEY",
            "api_available": False
        }
    
    if not api_keys['openai']:
        logger.warning("OpenAI API key not available for image generation")
    
    # Basic input validation
    if not all([idea, problem, prototype_description]):
        return {
            "valid": False,
            "message": "All fields (idea, problem, prototype description) are required",
            "error": "Missing required fields",
            "api_available": True
        }
    
    # Length checks
    if len(idea.strip()) < 10:
        return {
            "valid": False,
            "message": "Idea must be at least 10 characters long",
            "error": "Idea too short",
            "api_available": True
        }
    
    if len(problem.strip()) < 20:
        return {
            "valid": False,
            "message": "Problem statement must be at least 20 characters long",
            "error": "Problem statement too short",
            "api_available": True
        }
    
    if len(prototype_description.strip()) < 50:
        return {
            "valid": False,
            "message": "Prototype description must be at least 50 characters long",
            "error": "Prototype description too short",
            "api_available": True
        }
    
    # Initialize Groq client
    try:
        client = Groq(api_key=api_keys['groq'])
        logger.info("Groq client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        return {
            "valid": False,
            "message": "Failed to connect to validation service",
            "error": f"Groq client initialization failed: {str(e)}",
            "api_available": False
        }
    
    # Create validation prompt
    validation_prompt = create_validation_prompt(idea, problem, prototype_description)
    
    # Call Groq API for validation
    try:
        logger.info(f"Sending validation request to Groq using model: {model}")
        
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an educational content validator. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": validation_prompt
                }
            ],
            model=model,
            temperature=0.1,  # Low temperature for consistent validation
            max_tokens=200,   # Short response needed
            top_p=0.9
        )
        
        # Extract response content
        response_content = response.choices[0].message.content.strip()
        logger.info(f"Received response from Groq: {response_content[:100]}...")
        
        # Parse JSON response
        try:
            validation_result = json.loads(response_content)
            
            # Ensure required fields exist
            if 'valid' not in validation_result or 'message' not in validation_result:
                raise ValueError("Invalid response format from validation service")
            
            # Clean up message (ensure it's reasonable length)
            message = validation_result['message'][:200]  # Limit message length
            
            return {
                "valid": bool(validation_result['valid']),
                "message": message,
                "error": None,
                "api_available": True,
                "model_used": model,
                "openai_available": api_keys['openai'] is not None
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {response_content}")
            
            # Fallback: try to extract valid/invalid from text
            response_lower = response_content.lower()
            if 'valid": true' in response_lower or '"valid":true' in response_lower:
                return {
                    "valid": True,
                    "message": "Content appears appropriate for educational use",
                    "error": "JSON parsing failed but content seems valid",
                    "api_available": True
                }
            else:
                return {
                    "valid": False,
                    "message": "Content validation failed - please review your submission",
                    "error": "JSON parsing failed and content appears invalid",
                    "api_available": True
                }
    
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        
        # Handle specific error types
        error_message = str(e).lower()
        if "rate limit" in error_message:
            return {
                "valid": False,
                "message": "Validation service is currently busy. Please try again in a moment.",
                "error": "Rate limit exceeded",
                "api_available": True
            }
        elif "api key" in error_message or "unauthorized" in error_message:
            return {
                "valid": False,
                "message": "Validation service configuration error",
                "error": "Invalid API key",
                "api_available": False
            }
        else:
            return {
                "valid": False,
                "message": "Validation service is temporarily unavailable",
                "error": f"API error: {str(e)}",
                "api_available": True
            }

# Convenience function with different models
def validate_with_model(idea: str, problem: str, prototype_description: str, model_name: str = None) -> Dict[str, Any]:
    """
    Validate using specific Groq model.
    
    Available models (as of January 2025):
    - llama-3.3-70b-versatile (default, best for complex reasoning)
    - llama-3.1-8b-instant (faster, good for simple validation)
    - gemma2-9b-it (Google's Gemma model)
    - llama3-70b-8192 (Legacy Llama 3 with 8K context)
    - llama3-8b-8192 (Legacy Llama 3 8B with 8K context)
    """
    
    available_models = {
        "llama-70b": "llama-3.3-70b-versatile",  # Updated to current model
        "llama-8b": "llama-3.1-8b-instant", 
        "gemma": "gemma2-9b-it",  # Added Gemma option
        "llama3-70b": "llama3-70b-8192",  # Legacy model
        "llama3-8b": "llama3-8b-8192",  # Legacy model
        "default": "llama-3.3-70b-versatile"  # Updated default
    }
    
    model = available_models.get(model_name, available_models["default"])
    
    return validate_student_input(idea, problem, prototype_description, model)

# Example usage and testing
if __name__ == "__main__":
    print("=== Testing Student Input Validation with Groq (Updated Models) ===\n")
    
    # Test 1: Good educational content
    print("Test 1: Valid Educational Content")
    result1 = validate_student_input(
        idea="Smart water bottle that reminds students to stay hydrated during study sessions",
        problem="Students often forget to drink water while studying, leading to dehydration and reduced focus",
        prototype_description="IoT-enabled water bottle with LED indicators, smartphone app integration, customizable reminder intervals, and hydration tracking. Features include study mode integration, daily goals, and health insights."
    )
    print(f"Valid: {result1['valid']}")
    print(f"Message: {result1['message']}")
    print(f"Error: {result1['error']}")
    print(f"Model used: {result1.get('model_used', 'unknown')}")
    print()
    
    # Test 2: Unrelated content
    print("Test 2: Unrelated Content")
    result2 = validate_student_input(
        idea="Gaming console for entertainment",
        problem="Students need better study habits",
        prototype_description="Social media platform for sharing photos and videos with friends"
    )
    print(f"Valid: {result2['valid']}")
    print(f"Message: {result2['message']}")
    print(f"Model used: {result2.get('model_used', 'unknown')}")
    print()
    
    # Test 3: Inappropriate content
    print("Test 3: Inappropriate Content")
    result3 = validate_student_input(
        idea="Weapon detection system for schools",
        problem="School security concerns",
        prototype_description="AI-powered gun detection cameras with automatic alert systems"
    )
    print(f"Valid: {result3['valid']}")
    print(f"Message: {result3['message']}")
    print(f"Model used: {result3.get('model_used', 'unknown')}")
    print()
    
    # Test 4: Using fast model
    print("Test 4: Using Fast Model (8B)")
    result4 = validate_with_model(
        idea="Smart study planner app with AI recommendations",
        problem="Students struggle with time management and academic planning",
        prototype_description="Mobile app with calendar integration, task prioritization, study session scheduling, and personalized learning analytics",
        model_name="llama-8b"
    )
    print(f"Valid: {result4['valid']}")
    print(f"Message: {result4['message']}")
    print(f"Model used: {result4.get('model_used', 'unknown')}")
    print()
    
    # Test 5: Using Gemma model
    print("Test 5: Using Gemma Model")
    result5 = validate_with_model(
        idea="Environmental monitoring device for classrooms",
        problem="Poor air quality in classrooms affects student concentration and health",
        prototype_description="IoT sensor array measuring CO2, temperature, humidity, and air quality with real-time alerts and data visualization dashboard for teachers",
        model_name="gemma"
    )
    print(f"Valid: {result5['valid']}")
    print(f"Message: {result5['message']}")
    print(f"Model used: {result5.get('model_used', 'unknown')}")
    print()
    
 