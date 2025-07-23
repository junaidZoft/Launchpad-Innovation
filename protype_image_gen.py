from openai import OpenAI
import base64
from PIL import Image
import io
import logging
from typing import List, Dict, Any
from enum import Enum
import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Style options enum
class StyleOption(str, Enum):
    PHOTOREALISTIC = "Photorealistic Concept"
    MOCKUP_3D = "3D Mockup"
    WHITEBOARD = "Whiteboard Sketch"
    UI_MOCKUP = "User Interface (UI) Mockup"
    INFOGRAPHIC = "Infographic Style"

# Enhanced style guidance for innovation and entrepreneurship
STYLE_GUIDANCE = {
    StyleOption.PHOTOREALISTIC: "Create a high-resolution, photorealistic prototype visualization showing the innovation in a real-world context with actual users. Focus on demonstrating the problem-solution fit, user interaction, and market viability. Include environmental context that shows the target market using the product naturally.",
    StyleOption.MOCKUP_3D: "Design a professional 3D prototype render that showcases the innovation's key features, materials, and scalability potential. Emphasize manufacturability, cost-effectiveness, and user-centered design principles. Show the product from multiple angles if beneficial for understanding the innovation.",
    StyleOption.WHITEBOARD: "Create a detailed innovation sketch that combines product visualization with business model elements. Include annotations about key features, value propositions, target user needs, and competitive advantages. Use entrepreneur-style sketching with clear feature callouts and benefit explanations.",
    StyleOption.UI_MOCKUP: "Design a comprehensive digital innovation mockup showing user journey, key functionalities, and business model integration. Focus on user experience, market differentiation, and scalability. Include elements that demonstrate the innovation's unique value proposition and competitive advantage.",
    StyleOption.INFOGRAPHIC: "Create an innovation-focused infographic that combines product visualization with business model canvas elements. Show the problem-solution fit, target market segments, revenue streams, and implementation roadmap. Use entrepreneurial visual language with clear value propositions."
}

def download_image_as_base64(url: str) -> str:
    """Download image from URL and convert to base64"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        image_bytes = response.content
        
        # Validate it's a valid image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to base64
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
        return base64_string
    except Exception as e:
        logger.error(f"Error downloading image: {e}")
        raise ValueError(f"Failed to download image: {e}")

def validate_image_data(b64_string: str) -> bool:
    """Validate if base64 string contains valid image data"""
    try:
        image_bytes = base64.b64decode(b64_string)
        image = Image.open(io.BytesIO(image_bytes))
        # Verify it's a valid image by checking format
        return image.format in ['PNG', 'JPEG', 'JPG', 'WEBP']
    except Exception as e:
        logger.error(f"Image validation failed: {e}")
        return False

def create_generation_prompt(idea: str, problem: str, prototype_description: str, style: StyleOption) -> str:
    """Create an enhanced prompt for innovation and entrepreneurship focused image generation"""
    style_guide = STYLE_GUIDANCE[style]
    
    # Enhanced prompt template for innovation and entrepreneurship
    final_prompt = f"""
    INNOVATION PROTOTYPE VISUALIZATION

    INNOVATION CONTEXT:
    💡 Idea: {idea}
    🎯 Problem Statement: {problem}
    📋 Prototype Description: {prototype_description}
    🎨 Visualization Style: {style}

    STYLE REQUIREMENTS:
    {style_guide}

    INNOVATION & ENTREPRENEURSHIP FOCUS:
    • Demonstrate clear problem-solution fit and user value
    • Show real-world application and user interaction scenarios
    • Highlight innovative features and competitive differentiation
    • Include professional presentation quality suitable for stakeholders
    • Emphasize scalability potential and market viability
    • Show user-centered design principles and accessibility
    • Include context that demonstrates the innovation's impact

    DELIVERABLE:
    Create a compelling, professional prototype visualization that clearly demonstrates how this innovation solves the stated problem. The image should be suitable for presentation to investors, customers, and stakeholders in the innovation ecosystem. Focus on showing the prototype in action, solving real user problems, with clear value proposition communication.

    Make the visualization engaging, technically accurate, and commercially viable while maintaining the specified artistic style.
    """
    
    return final_prompt.strip()

def generate_prototype_images(
    idea: str,
    problem: str,
    prototype_description: str,
    style: StyleOption = StyleOption.PHOTOREALISTIC,
    num_images: int = 2,
    api_key: str = None
) -> Dict[str, Any]:
    """
    Generate prototype images based on the provided specifications.
    
    Args:
        idea (str): The innovative idea or solution concept (1-1000 chars)
        problem (str): The specific problem statement being solved (1-1000 chars)
        prototype_description (str): Detailed prototype description including key features (1-2000 chars)
        style (StyleOption): Visualization style for the prototype
        num_images (int): Number of images to generate (1-4)
        api_key (str): OpenAI API key (if None, will try to get from environment)
    
    Returns:
        Dict containing:
        - success (bool): Whether the operation was successful
        - images (List[Dict]): Generated images with base64 data and variation numbers
        - prompt_used (str): The prompt used for image generation
        - num_generated (int): Number of images generated
        - error (str): Error message if success is False
    """
    
    # Validate inputsaz
    if not idea or len(idea) > 1000:
        return {"success": False, "error": "Idea must be between 1-1000 characters"}
    
    if not problem or len(problem) > 1000:
        return {"success": False, "error": "Problem must be between 1-1000 characters"}
    
    if not prototype_description or len(prototype_description) > 2000:
        return {"success": False, "error": "Prototype description must be between 1-2000 characters"}
    
    if not isinstance(style, StyleOption):
        return {"success": False, "error": "Invalid style option"}
    
    if num_images < 1 or num_images > 4:
        return {"success": False, "error": "Number of images must be between 1-4"}
    
    # Get API key
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        return {"success": False, "error": "OpenAI API key is required"}
    
    # Initialize OpenAI client
    try:
        openai_client = OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return {"success": False, "error": f"Failed to initialize OpenAI client: {e}"}
    
    logger.info(f"Received prototype generation request for {num_images} image(s)")
    
    try:
        # Create generation prompt
        final_prompt = create_generation_prompt(idea, problem, prototype_description, style)
        logger.info(f"Generated prompt for style: {style}")
        
        # Generate images
        try:
            logger.info(f"Requesting {num_images} image(s) from OpenAI")
            
            # Use gpt-image-1 model with correct parameters
            response = openai_client.images.generate(
                model="gpt-image-1",
                prompt=final_prompt,
                n=num_images,  # gpt-image-1 supports multiple images
                size="1024x1024"
            )
            
            logger.info(f"Successfully generated {len(response.data)} image(s)")
            
            # Use the response data directly
            all_images = response.data
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            
            # Handle specific OpenAI errors
            if "rate limit" in str(e).lower():
                return {"success": False, "error": "Rate limit exceeded. Please try again later."}
            elif "invalid" in str(e).lower() and "key" in str(e).lower():
                return {"success": False, "error": "Invalid API key"}
            else:
                return {"success": False, "error": f"Image generation failed: {str(e)}"}
        
        # Process generated images
        processed_images = []
        for i, img_data in enumerate(all_images):
            try:
                # Handle both URL and base64 responses
                if hasattr(img_data, 'url') and img_data.url:
                    # Download image from URL and convert to base64
                    image_base64 = download_image_as_base64(img_data.url)
                elif hasattr(img_data, 'b64_json') and img_data.b64_json:
                    # Use base64 data directly
                    image_base64 = img_data.b64_json
                else:
                    logger.warning(f"Unknown image data format for variation {i+1}")
                    continue
                
                # Validate image data
                if not validate_image_data(image_base64):
                    logger.warning(f"Invalid image data for variation {i+1}")
                    continue
                
                processed_images.append({
                    "image_base64": image_base64,
                    "variation_number": i + 1
                })
                
            except Exception as e:
                logger.error(f"Error processing image {i+1}: {e}")
                continue
        
        if not processed_images:
            logger.error("No valid images were generated")
            return {"success": False, "error": "No valid images were generated"}
        
        logger.info(f"Successfully processed {len(processed_images)} image(s)")
        
        return {
            "success": True,
            "images": processed_images,
            "prompt_used": final_prompt,
            "num_generated": len(processed_images),
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in generate_prototype_images: {e}")
        return {"success": False, "error": f"An unexpected error occurred: {str(e)}"}


# Example usage
if __name__ == "__main__":
    # Example usage of the function
    result = generate_prototype_images(
        idea="Smart water bottle that tracks hydration and reminds users to drink water",
        problem="People often forget to stay hydrated throughout the day, leading to health issues",
        prototype_description="A sleek water bottle with LED indicators, smartphone connectivity, and gentle vibration reminders. Features include water intake tracking, personalized hydration goals, and integration with fitness apps.",
        style=StyleOption.PHOTOREALISTIC,
        num_images=2
    )
    
    if result["success"]:
        print(f"Successfully generated {result['num_generated']} images")
        print(f"Prompt used: {result['prompt_used'][:100]}...")
        for img in result["images"]:
            print(f"Generated image {img['variation_number']} (base64 length: {len(img['image_base64'])})")
    else:
        print(f"Error: {result['error']}")