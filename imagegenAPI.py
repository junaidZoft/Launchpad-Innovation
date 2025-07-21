from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, validator
from openai import OpenAI
import base64
from PIL import Image
import io
import logging
from typing import List, Optional
from enum import Enum
import uvicorn
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

# Initialize FastAPI app
app = FastAPI(
    title="Prototype Image Generator API",
    description="Generate prototype images using OpenAI's DALL-E model",
    version="1.0.0"
)

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable is not set")
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Initialize OpenAI client
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    raise ValueError(f"Failed to initialize OpenAI client: {e}")

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

# Request models
class PrototypeRequest(BaseModel):
    idea: str = Field(..., min_length=1, max_length=1000, description="The innovative idea or solution concept")
    problem: str = Field(..., min_length=1, max_length=1000, description="The specific problem statement being solved")
    prototype_description: str = Field(..., min_length=1, max_length=2000, description="Detailed prototype description including key features")
    style: StyleOption = Field(..., description="Visualization style for the prototype")
    num_images: int = Field(default=2, ge=1, le=4, description="Number of images to generate (1-4)")
    
    @validator('prototype_description')
    def validate_prototype_description(cls, v):
        if not v.strip():
            raise ValueError('Prototype description cannot be empty')
        return v.strip()

# Response models
class ImageData(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image data")
    variation_number: int = Field(..., description="Image variation number")

class PrototypeResponse(BaseModel):
    success: bool = Field(..., description="Whether the operation was successful")
    images: List[ImageData] = Field(..., description="Generated images")
    prompt_used: str = Field(..., description="The prompt used for image generation")
    num_generated: int = Field(..., description="Number of images generated")

class ErrorResponse(BaseModel):
    success: bool = Field(default=False)
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")

# Helper functions
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

def create_generation_prompt(request: PrototypeRequest) -> str:
    """Create an enhanced prompt for innovation and entrepreneurship focused image generation"""
    style_guide = STYLE_GUIDANCE[request.style]
    
    # Enhanced prompt template for innovation and entrepreneurship
    final_prompt = f"""
    INNOVATION PROTOTYPE VISUALIZATION

    INNOVATION CONTEXT:
    💡 Idea: {request.idea}
    🎯 Problem Statement: {request.problem}
    📋 Prototype Description: {request.prototype_description}
    🎨 Visualization Style: {request.style}

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

# API Routes
@app.post(
    "/generate-prototype",
    response_model=PrototypeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        401: {"model": ErrorResponse, "description": "Unauthorized - Invalid API Key"},
        429: {"model": ErrorResponse, "description": "Rate Limited"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    }
)
async def generate_prototype(request: PrototypeRequest):
    """
    Generate prototype images based on the provided specifications.
    
    This endpoint creates visual prototypes using OpenAI's DALL-E model.
    """
    logger.info(f"Received prototype generation request for {request.num_images} image(s)")
    
    try:
        # Create generation prompt
        final_prompt = create_generation_prompt(request)
        logger.info(f"Generated prompt for style: {request.style}")
        
        # Generate images
        try:
            logger.info(f"Requesting {request.num_images} image(s) from OpenAI")
            
            # Use image-1 model with correct parameters
            response = openai_client.images.generate(
                model="image-1",
                prompt=final_prompt,
                n=request.num_images,  # image-1 supports multiple images
                size="1024x1024"
            )
            
            logger.info(f"Successfully generated {len(response.data)} image(s)")
            
            # Use the response data directly
            all_images = response.data
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            
            # Handle specific OpenAI errors
            if "rate limit" in str(e).lower():
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=ErrorResponse(
                        error="Rate limit exceeded. Please try again later.",
                        error_code="RATE_LIMIT_EXCEEDED"
                    ).dict()
                )
            elif "invalid" in str(e).lower() and "key" in str(e).lower():
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=ErrorResponse(
                        error="Invalid API key",
                        error_code="INVALID_API_KEY"
                    ).dict()
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=ErrorResponse(
                        error=f"Image generation failed: {str(e)}",
                        error_code="GENERATION_FAILED"
                    ).dict()
                )
        
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
                
                processed_images.append(ImageData(
                    image_base64=image_base64,
                    variation_number=i + 1
                ))
                
            except Exception as e:
                logger.error(f"Error processing image {i+1}: {e}")
                continue
        
        if not processed_images:
            logger.error("No valid images were generated")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ErrorResponse(
                    error="No valid images were generated",
                    error_code="NO_VALID_IMAGES"
                ).dict()
            )
        
        logger.info(f"Successfully processed {len(processed_images)} image(s)")
        
        return PrototypeResponse(
            success=True,
            images=processed_images,
            prompt_used=final_prompt,
            num_generated=len(processed_images)
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_prototype: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error=f"An unexpected error occurred: {str(e)}",
                error_code="INTERNAL_ERROR"
            ).dict()
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Prototype Image Generator API"}

@app.get("/styles")
async def get_available_styles():
    """Get available visualization styles for innovation prototyping"""
    return {
        "styles": [
            {
                "key": style.value,
                "description": STYLE_GUIDANCE[style],
                "best_for": {
                    StyleOption.PHOTOREALISTIC: "Final presentations, investor pitches, market validation",
                    StyleOption.MOCKUP_3D: "Product development, manufacturing planning, technical reviews",
                    StyleOption.WHITEBOARD: "Brainstorming sessions, concept development, team collaboration",
                    StyleOption.UI_MOCKUP: "Digital products, user experience design, software solutions",
                    StyleOption.INFOGRAPHIC: "Business model presentation, process explanation, stakeholder communication"
                }[style]
            }
            for style in StyleOption
        ]
    }

@app.post("/validate-innovation")
async def validate_innovation_concept(request: PrototypeRequest):
    """Validate innovation concept and provide feedback before prototype generation"""
    feedback = {
        "concept_strength": "medium",
        "recommendations": [],
        "missing_elements": [],
        "market_potential": "unknown"
    }
    
    # Analyze problem statement
    if len(request.problem) < 50:
        feedback["recommendations"].append("Expand your problem statement to include specific pain points and user impact")
    
    # Analyze idea clarity and innovation
    innovation_keywords = ["new", "innovative", "unique", "different", "better", "improved", "novel", "smart", "automated", "efficient"]
    if not any(keyword in request.idea.lower() for keyword in innovation_keywords):
        feedback["recommendations"].append("Highlight what makes your solution innovative and different from existing alternatives")
    
    # Check prototype description completeness
    if len(request.prototype_description) < 100:
        feedback["recommendations"].append("Provide more detailed prototype description including key features and user benefits")
    
    # Check for technical feasibility indicators
    technical_keywords = ["technology", "system", "platform", "device", "software", "hardware", "algorithm", "data", "sensor", "app"]
    if not any(keyword in request.prototype_description.lower() for keyword in technical_keywords):
        feedback["missing_elements"].append("Technical implementation details or technology stack")
    
    # Check for user-focused language
    user_keywords = ["user", "customer", "people", "solve", "help", "benefit", "experience", "interface", "interaction"]
    if not any(keyword in request.prototype_description.lower() for keyword in user_keywords):
        feedback["missing_elements"].append("User-centered design and experience considerations")
    
    # Determine concept strength
    if len(feedback["recommendations"]) == 0 and len(feedback["missing_elements"]) <= 1:
        feedback["concept_strength"] = "strong"
        feedback["market_potential"] = "high"
    elif len(feedback["recommendations"]) <= 2 and len(feedback["missing_elements"]) <= 2:
        feedback["concept_strength"] = "medium"
        feedback["market_potential"] = "medium"
    else:
        feedback["concept_strength"] = "needs_improvement"
        feedback["market_potential"] = "low"
    
    feedback["ready_for_prototyping"] = feedback["concept_strength"] in ["strong", "medium"]
    
    return feedback

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    logger.error(f"Validation error: {exc}")
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=ErrorResponse(
            error=str(exc),
            error_code="VALIDATION_ERROR"
        ).dict()
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )