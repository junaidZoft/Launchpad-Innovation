"""
Market Fit Evaluator - Optimized Groq Version

A high-performance evaluator for assessing market fit of student projects using Groq API.
Optimized for speed with comprehensive error handling, caching, and batch processing.
"""

import os
import json
import time
import logging
import asyncio
import hashlib
from typing import Dict, Any, Optional, Tuple, List
from functools import lru_cache
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import aiohttp
from contextlib import asynccontextmanager

try:
    from groq import Groq, AsyncGroq
except ImportError as e:
    raise ImportError(
        "Groq library not found. "
        "Install with: pip install groq python-dotenv aiohttp"
    ) from e

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logging.warning("python-dotenv not found. Environment variables must be set manually.")


class ClassificationError(Exception):
    """Custom exception for classification errors."""
    pass


class APIError(ClassificationError):
    """API-related errors."""
    pass


class ConfigurationError(ClassificationError):
    """Configuration-related errors."""
    pass


class ValidationError(ClassificationError):
    """Input validation errors."""
    pass


@dataclass
class MarketFitResult:
    """Structured result from market fit evaluation."""
    success: bool
    market_potential: Optional[str] = None
    customer_validation: Optional[str] = None
    business_viability: Optional[str] = None
    competitive_advantage: Optional[str] = None
    error: Optional[str] = None
    raw_response: Optional[str] = None
    processing_time: Optional[float] = None
    tokens_used: Optional[int] = None
    model_used: Optional[str] = None


class QualityLevel(Enum):
    """X-axis quality levels for validation."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    EXCEPTIONAL = "exceptional"


class ContentLevel(Enum):
    """Y-axis content levels for validation."""
    SINGLE = "single"
    DOUBLE = "double"
    TRIPLE = "triple"
    QUADRUPLE = "quadruple"
    QUINTUPLE = "quintuple"
    EXCEPTIONAL = "exceptional"


class OptimizedProblemStatementClassifier:
    """
    High-performance classifier using Groq API for fast inference.
    
    Features:
    - Ultra-fast Groq API integration
    - Advanced caching with TTL
    - Batch processing with connection pooling
    - Optimized prompt templates
    - Performance monitoring and metrics
    - Async/await support throughout
    """
    
    # Available Groq models (optimized for speed)
    FAST_MODELS = [
        "llama3-70b-8192",      # Fastest, high quality
        "llama3-8b-8192",       # Very fast, good quality
        "mixtral-8x7b-32768",   # Good balance
        "gemma-7b-it",          # Lightweight
    ]
    
    def _init_(
        self,
        api_key: Optional[str] = None,
        model_name: str = "llama3-70b-8192",  # Default to fastest model
        max_retries: int = 2,  # Reduced retries for speed
        timeout: float = 15.0,  # Reduced timeout
        enable_caching: bool = True,
        cache_size: int = 256,  # Increased cache size
        cache_ttl: int = 3600,  # Cache TTL in seconds
        log_level: str = "INFO",
        max_concurrent: int = 10  # For batch processing
    ):
        """Initialize the optimized classifier."""
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model_name = model_name
        self.max_retries = max_retries
        self.timeout = timeout
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self.max_concurrent = max_concurrent
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        self._validate_configuration()
        
        # Initialize Groq clients
        self.client = Groq(api_key=self.api_key)
        self.async_client = AsyncGroq(api_key=self.api_key)
        
        # Initialize cache with TTL
        self._cache = {}
        self._cache_timestamps = {}
        
        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_response_time': 0.0,
            'total_tokens': 0,
            'fastest_response': float('inf'),
            'slowest_response': 0.0
        }
        
        # Thread pool for sync operations
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        
        self.logger.info(f"Groq Classifier initialized with model: {self.model_name}")
    
    def _validate_configuration(self) -> None:
        """Validate configuration parameters."""
        if not self.api_key:
            raise ConfigurationError(
                "GROQ_API_KEY not found. Set it as an environment variable or pass it directly."
            )
        
        if len(self.api_key.strip()) < 10:
            raise ConfigurationError("API key appears to be invalid (too short)")
        
        if self.model_name not in self.FAST_MODELS:
            self.logger.warning(f"Model {self.model_name} not in recommended fast models: {self.FAST_MODELS}")
        
        if self.max_retries < 1:
            raise ConfigurationError("max_retries must be at least 1")
        
        if self.timeout <= 0:
            raise ConfigurationError("timeout must be positive")
    
    @lru_cache(maxsize=1)
    def _get_optimized_prompt(self) -> str:
        """Get the optimized system prompt (cached)."""
        return '''
You are a specialized Assessment Agent designed to evaluate problem statements written by students aged 14-16 years, focusing on United Nations Sustainable Development Goals (SDGs). Your primary function is to provide consistent, objective analysis that remains stable across multiple evaluations of the same content.

## Core Mission
Analyze student-written problem statements with precision and consistency, ensuring that repeated assessments of identical content yield identical results. Your evaluation must be thorough, fair, and appropriate for the developmental level of teenage students. The provided IDEA will help you assess whether the problem statement is relevant and well-aligned with the intended solution concept.

## Assessment Framework
You must classify each problem statement using TWO dimensions:
- X-Axis (Quality Assessment): Evaluating the technical and structural quality of the writing, INCLUDING relevance to the provided idea
- Y-Axis (Content Assessment): Identifying what substantive elements are present in the problem statement

## Output Requirements
Your response MUST be a single, valid JSON object with this EXACT structure:

json
{{
  "X_Axis_Rubric_Category": "<single string from X-axis categories>",
  "Y_Axis_Rubric_Category": "<single string from Y-axis categories>"
}}


## Assessment Process for Problem Statement Evaluation

### Step 1: Content Analysis (Y-Axis)
Carefully examine the problem statement to determine which of the following elements are clearly present. Select the ONE Y-axis category that includes all the identifiable elements:

*Contains Data*: Includes relevant quantitative or qualitative data supporting the existence of the problem.

*References Included*: Cites any sources or references.

*Location/Area Clear*: Clearly specifies the geographical location or specific area affected.

*Target Audience Clearly Stated*: Defines the specific group or demographic impacted by the issue.

*Impact Described*: Explains the consequences or negative outcomes if the problem remains unaddressed.

✔ Choose the category that matches all the elements present in the problem statement.

### Step 2: Quality Analysis (X-Axis) - ENHANCED WITH IDEA RELEVANCE
Evaluate the overall quality of the problem statement based on these five dimensions, with special attention to how well it aligns with the provided idea. Select the ONE best matching X-axis category based on how many of these are fully demonstrated:

*GRAMMAR*: Uses correct grammar, punctuation, spelling, sentence structure, and appropriate vocabulary throughout.

*DEMONSTRATES UNDERSTANDING*: Shows insight and clear comprehension of the topic and SDG concept.

*PRECISE AND TO THE POINT*: Avoids unnecessary detail and focuses on the core message without redundancy.

*RELEVANT TO THE IDEA*: Content clearly supports and aligns with the provided idea. The problem statement should logically connect to the idea as a potential solution or approach. If the problem statement is not relevant to the idea, this dimension is automatically NOT met.

*INFO IS WELL-STRUCTURED AND EASY TO UNDERSTAND*: Logical organization, clear flow, and easily comprehensible to readers.

✔ Choose the category that best reflects the overall writing quality and clarity based on the above dimensions, giving special weight to idea relevance.

### Critical Assessment Guidelines

*Idea Relevance Evaluation (Critical for X-Axis)*:
- Does the problem statement address the same issue that the idea is trying to solve?
- Are the problem statement and idea focused on the same or related SDG themes?
- Would the provided idea logically contribute to solving the stated problem?
- Is there thematic alignment between the problem context and the idea's scope?

*If the problem statement is NOT relevant to the provided idea*, the X-axis score should reflect this by selecting categories that include "Is not Relevant to the Idea" or lower quality combinations.

*Grammar Assessment (Enhanced)*:
Pay special attention to:
- Subject-verb agreement errors
- Incorrect tense usage
- Spelling mistakes
- Punctuation errors
- Run-on sentences or fragments
- Unclear or confusing sentence structure
- Inappropriate vocabulary for the context

## Critical Guidelines for Consistency

1. *Age-Appropriate Expectations*: Remember these are 14-16 year old students. Apply standards appropriate for this developmental level.

2. *Objective Analysis*: Base your assessment solely on what is explicitly present in the text, not on implied or inferred meanings.

3. *Consistent Criteria*: Use the same evaluation standards for every assessment. Apply consistent thresholds for grammar, understanding, precision, relevance to idea, and structure.

4. *Complete Analysis*: Examine ALL aspects thoroughly before making your final classification.

5. *Reproducible Results*: Your assessment of identical content must be identical every time, regardless of when the evaluation occurs.

6. *Idea Relevance Priority*: If the problem statement does not align with the provided idea, this significantly impacts the X-axis quality score.

## Valid Categories for Output

### X-AXIS CATEGORIES (Quality Assessment)

*Lowest Quality Level:*
- "Target Audience Alignment only"
- "Problem-Solution Alignment only"
- "Customer Validation Evidence only"
- "Unique Value Proposition only"
- "Outlines initial steps for entering the market only"

*Combined Low Quality:*
- "Target Audience Alignment + Problem-Solution Alignment"
- "Target Audience Alignment + Customer Validation Evidence"
- "Target Audience Alignment + Unique Value Proposition"
- "Target Audience Alignment + Outlines initial steps"
- "Problem-Solution Alignment + Customer Validation Evidence"
- "Problem-Solution Alignment + Unique Value Proposition"
- "Problem-Solution Alignment + Outlines initial steps"
- "Customer Validation Evidence + Unique Value Proposition"
- "Customer Validation Evidence + Outlines initial steps"
- "Unique Value Proposition + Outlines initial steps"

*Triple and Quadruple Low Quality Combinations:*
- "Target Audience Alignment + Problem-Solution Alignment + Outlines initial steps"
- "Target Audience Alignment + Customer Validation Evidence + Unique Value Proposition"
- "Target Audience Alignment + Customer Validation Evidence + Outlines initial steps"
- "Target Audience Alignment + Unique Value Proposition + Outlines initial steps"
- "Problem-Solution Alignment + Customer Validation Evidence + Unique Value Proposition"
- "Problem-Solution Alignment + Customer Validation Evidence + Outlines initial steps"
- "Problem-Solution Alignment + Unique Value Proposition + Outlines initial steps"
- "Customer Validation Evidence + Unique Value Proposition + Outlines initial steps"

*Medium Quality Level:*
- "Target Audience Alignment + Problem-Solution Alignment + Customer Validation Evidence + Unique Value Proposition"
- "Target Audience Alignment + Problem-Solution Alignment + Customer Validation Evidence + Outlines initial steps"
- "Target Audience Alignment + Problem-Solution Alignment + Unique Value Proposition + Outlines initial steps"
- "Target Audience Alignment + Customer Validation Evidence + Unique Value Proposition + Outlines initial steps"
- "Problem-Solution Alignment + Customer Validation Evidence + Unique Value Proposition + Outlines initial steps"

*Combined High Quality:*
- "Target Audience Alignment + Problem-Solution Alignment + Customer Validation Evidence + Unique Value Proposition + Outlines initial steps"
- "Target Audience Alignment + Problem-Solution Alignment + Customer Validation EvidenceTarget Audience Alignment + Problem-Solution Alignment + Unique Value Proposition"

*Exceptional case:*
- "not Relevant content to the Idea"

### Y-AXIS CATEGORIES (Content Elements)

*Single Elements:*
- "Relevant to the Idea Only"
- "Does not have grammar"
- "Does not Demonstrate Understanding Only"
- "Is not Precise and To the Point"
- "Info is not Well-Structured and Is not Easy to Understand"
- "Has some grammar"
- "Demonstrates some Understanding"
- "Is somewhat Precise and To the Point"
- "Info is somewhat Well-Structured and fairly Easy to Understand"

*Two Element Combinations:*
- "Does not have Grammar + Does not Demonstrate Understanding" 
- "Does not have Grammar + Is not Precise and To the Point"
- "Does not have Grammar + Info is not Well-Structured and Is not Easy to Understand"
- "Does not Demonstrate Understanding + Info is not Well-Structured and Is not Easy to Understand"
- "Does not Demonstrate Understanding + Is not Precise and To the Point"
- "Is not Precise and To the Point + Info is not Well-Structured and Is not Easy to Understand"
- "Has some Grammar + Demonstrates some Understanding"
- "Has some Grammar + is somewhat Precise and To the Point"
- "Has some Grammar + Info is somewhat Well-Structured and fairly Easy to Understand"
- "Demonstrates some Understanding + Info is somewhat Well-Structured and fairly Easy to Understand"
- "Demonstrates some Understanding + is somewhat Precise and To the Point"
- "Is somewhat Precise and To the Point + Info is somewhat Well-Structured and is somewhat Easy to Understand"
- "Has Very good Grammar + Demonstrates Very good Understanding"
- "Has Very good Grammar + Is Precise and To the Point"
- "Has Very good Grammar + Info is Well-Structured and Easy to Understand"
- "Demonstrates Very good Understanding + Is Precise and To the Point"
- "Demonstrates Very Good Understanding + Info is Very Well-Structured and Easy to Understand"
- "Is Precise and To the Point + Info is Very Well-Structured and Easy to Understand"

*Three Element Combinations:*
- "Does not have Grammar + Does not Demonstrate Understanding + Is not Precise and To the Point"
- "Does not have Grammar + Does not Demonstrate Understanding + Info is not Well-Structured and Is not Easy to Understand"
- "Does not have Grammar + Is not Precise and To the Point + Info is not Well-Structured and Is not Easy to Understand"
- "Does not Demonstrate Understanding + Is not Precise and To the Point + Info is not Well-Structured and Is not Easy to Understand"
- "Does not have Grammar + Does not Demonstrate Understanding + Is not Precise and To the Point + Info is not Well-Structured and Is not Easy to Understand"
- "Has some Grammar + Demonstrates some Understanding + is somewhat Precise and To the Point"
- "Has some Grammar + Demonstrates some Understanding + Info is somewhat Well-Structured and somewhat Easy to Understand"
- "Has some Grammar + is somewhat Precise and To the Point + Info is somewhat Well-Structured and is somewhat Easy to Understand"
- "Demonstrates some Understanding + is somewhat Precise and To the Point + Info is somewhat Well-Structured and is somewhat Easy to Understand"
- "Has some Grammar + Demonstrates some Understanding + is somewhat Precise and To the Point + Info is somewhat Well-Structured and somewhat Easy to Understand"
- "Has very good Grammar + Demonstrates Very Good Understanding + Is Precise and To the Point"
- "Has Very Good Grammar + Demonstrates Very Good Understanding + Info is Very Well-Structured and Easy to Understand"
- "Has Very Good Grammar + Is Precise and To the Point + Info is Very Well-Structured and Easy to Understand"
- "Demonstrates Very Good Understanding + Is Precise and To the Point + Info is Very Well-Structured and Easy to Understand"

*Four and Five Element Combinations:*
- "Has Very Good Grammar + Demonstrates Very Good Understanding + Is Precise and To the Point + Info is Very Well-Structured and Easy to Understand"

*Exceptional case:*
- "not Relevant content to the Idea"

---

Now analyze the following market fit description:

{market_fit_description}

Evaluate the market fit based on these key aspects:
1. Market Potential (size, growth, trends)
2. Customer Validation (evidence of demand, user feedback)
3. Business Viability (revenue model, cost structure)
4. Competitive Advantage (unique selling points, barriers to entry)

Provide ONLY a JSON output with the following structure:
{
    "market_potential": "<high|medium|low> - Brief justification",
    "customer_validation": "<strong|moderate|weak> - Brief justification",
    "business_viability": "<viable|uncertain|challenging> - Brief justification",
    "competitive_advantage": "<strong|moderate|weak> - Brief justification"
}
'''
    
    def _validate_inputs(self, market_fit_description: str) -> None:
        """Validate input parameters with optimized checks."""
        if not market_fit_description or not isinstance(market_fit_description, str) or len(market_fit_description.strip()) < 50:
            raise ValidationError("market_fit_description must be a non-empty string with at least 50 characters")
        
        # Quick length check
        if len(market_fit_description) > 8000:
            raise ValidationError("market_fit_description exceeds maximum length of 8000 characters")
    
    def _sanitize_text(self, text: str) -> str:
        """Fast text sanitization."""
        return ' '.join(text.strip().split())
    
    def _create_cache_key(self, idea_text: str, problem_statement_text: str) -> str:
        """Create optimized cache key."""
        combined = f"{idea_text[:200]}|{problem_statement_text[:500]}"  # Limit key size
        return hashlib.blake2b(combined.encode(), digest_size=16).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if cache_key not in self._cache_timestamps:
            return False
        
        return (time.time() - self._cache_timestamps[cache_key]) < self.cache_ttl
    
    def _get_cached_result(self, cache_key: str) -> Optional[MarketFitResult]:
        """Get cached result with TTL check."""
        if not self.enable_caching or cache_key not in self._cache:
            self.metrics['cache_misses'] += 1
            return None
        
        if not self._is_cache_valid(cache_key):
            # Remove expired cache entry
            del self._cache[cache_key]
            del self._cache_timestamps[cache_key]
            self.metrics['cache_misses'] += 1
            return None
        
        self.metrics['cache_hits'] += 1
        return self._cache[cache_key]
    
    def _cache_result(self, cache_key: str, result: MarketFitResult) -> None:
        """Cache result with TTL."""
        if not self.enable_caching:
            return
        
        # Simple LRU eviction
        if len(self._cache) >= self.cache_size:
            oldest_key = min(self._cache_timestamps.keys(), 
                           key=self._cache_timestamps.get)
            del self._cache[oldest_key]
            del self._cache_timestamps[oldest_key]
        
        self._cache[cache_key] = result
        self._cache_timestamps[cache_key] = time.time()
    
    def _parse_json_response(self, response_text: str) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Fast JSON parsing with fallback."""
        try:
            # Clean response
            text = response_text.strip()
            
            # Remove markdown formatting
            if text.startswith('json'):
                text = text[7:]
            if text.endswith(''):
                text = text[:-3]
            text = text.strip()
            
            # Find JSON boundaries
            start = text.find('{')
            end = text.rfind('}') + 1
            
            if start == -1 or end <= start:
                return False, {}, "No JSON found in response"
            
            json_str = text[start:end]
            parsed = json.loads(json_str)
            
            # Quick validation
            required_keys = ["X_Axis_Rubric_Category", "Y_Axis_Rubric_Category"]
            if not all(k in parsed and isinstance(parsed[k], str) and parsed[k].strip() 
                      for k in required_keys):
                return False, {}, "Invalid or missing required fields"
            
            return True, parsed, None
            
        except json.JSONDecodeError as e:
            return False, {}, f"JSON parsing error: {str(e)}"
        except Exception as e:
            return False, {}, f"Response parsing error: {str(e)}"
    
    async def _make_api_call_async(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Make async API call to Groq."""
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a precise assessment agent. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low for consistency
                max_tokens=1000,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            usage_info = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            
            return response.choices[0].message.content, usage_info
            
        except Exception as e:
            raise APIError(f"Groq API call failed: {str(e)}")
    
    def _make_api_call_sync(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Make synchronous API call to Groq."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a precise assessment agent. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            usage_info = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            
            return response.choices[0].message.content, usage_info
            
        except Exception as e:
            raise APIError(f"Groq API call failed: {str(e)}")
    
    async def evaluate_market_fit_async(self, market_fit_description: str) -> MarketFitResult:
        """Async classification method for maximum speed."""
        start_time = time.time()
        self.metrics['total_requests'] += 1
        
        try:
            # Fast validation and sanitization
            self._validate_inputs(idea_text, problem_statement_text)
            idea_text = self._sanitize_text(idea_text)
            problem_statement_text = self._sanitize_text(problem_statement_text)
            
            # Check cache
            cache_key = self._create_cache_key(idea_text, problem_statement_text)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            # Build optimized prompt
            prompt = self._get_optimized_prompt().format(
                idea_text=idea_text,
                problem_statement_text=problem_statement_text
            )
            
            # Make API call with retries
            last_error = None
            for attempt in range(self.max_retries):
                try:
                    response_text, usage_info = await self._make_api_call_async(prompt)
                    
                    # Parse response
                    success, parsed_data, error = self._parse_json_response(response_text)
                    
                    if success:
                        processing_time = time.time() - start_time
                        
                        result = MarketFitResult(
                            success=True,
                            market_potential=parsed_data['market_potential'],
                            customer_validation=parsed_data['customer_validation'],
                            business_viability=parsed_data['business_viability'],
                            competitive_advantage=parsed_data['competitive_advantage'],
                            raw_response=response_text,
                            processing_time=processing_time,
                            tokens_used=usage_info.get('total_tokens', 0),
                            model_used=self.model_name
                        )
                        
                        # Update metrics
                        self._update_metrics(processing_time, usage_info.get('total_tokens', 0), True)
                        
                        # Cache result
                        self._cache_result(cache_key, result)
                        
                        return result
                    else:
                        raise APIError(f"Response parsing failed: {error}")
                
                except Exception as e:
                    last_error = e
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(0.5 * (attempt + 1))  # Fast exponential backoff
            
            # All retries failed
            processing_time = time.time() - start_time
            self._update_metrics(processing_time, 0, False)
            
            return MarketFitResult(
                success=False,
                error=f"Failed after {self.max_retries} attempts: {str(last_error)}",
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_metrics(processing_time, 0, False)
            
            return MarketFitResult(
                success=False,
                error=str(e),
                processing_time=processing_time
            )
    
    def evaluate_market_fit_sync(self, market_fit_description: str) -> MarketFitResult:
        """Synchronous market fit evaluation method."""
        return asyncio.run(self.evaluate_market_fit_async(market_fit_description))
    
    # Alias for backwards compatibility
    def evaluate_market_fit(self, market_fit_description: str) -> MarketFitResult:
        """Main market fit evaluation method."""
        return self.evaluate_market_fit_sync(market_fit_description)
    
    async def evaluate_batch_async(self, descriptions: List[str]) -> List[MarketFitResult]:
        """High-speed batch evaluation."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def evaluate_single(description: str) -> MarketFitResult:
            async with semaphore:
                return await self.evaluate_market_fit_async(description)
        
        tasks = [evaluate_single(desc) for desc in descriptions]
        return await asyncio.gather(*tasks, return_exceptions=False)
    
    def evaluate_batch_sync(self, descriptions: List[str]) -> List[MarketFitResult]:
        """Synchronous batch evaluation."""
        return asyncio.run(self.evaluate_batch_async(descriptions))
    
    def _update_metrics(self, processing_time: float, tokens: int, success: bool) -> None:
        """Update performance metrics."""
        if success:
            self.metrics['successful_requests'] += 1
            
            # Update response time stats
            if self.metrics['successful_requests'] == 1:
                self.metrics['average_response_time'] = processing_time
            else:
                n = self.metrics['successful_requests']
                current_avg = self.metrics['average_response_time']
                self.metrics['average_response_time'] = (current_avg * (n - 1) + processing_time) / n
            
            # Update fastest/slowest
            self.metrics['fastest_response'] = min(self.metrics['fastest_response'], processing_time)
            self.metrics['slowest_response'] = max(self.metrics['slowest_response'], processing_time)
            
            # Update token usage
            self.metrics['total_tokens'] += tokens
        else:
            self.metrics['failed_requests'] += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        total_requests = self.metrics['total_requests']
        if total_requests == 0:
            return {"message": "No requests processed yet"}
        
        success_rate = (self.metrics['successful_requests'] / total_requests) * 100
        cache_hit_rate = (self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses'])) * 100 if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0 else 0
        
        return {
            'total_requests': total_requests,
            'successful_requests': self.metrics['successful_requests'],
            'failed_requests': self.metrics['failed_requests'],
            'success_rate': f"{success_rate:.2f}%",
            'cache_hits': self.metrics['cache_hits'],
            'cache_misses': self.metrics['cache_misses'],
            'cache_hit_rate': f"{cache_hit_rate:.2f}%",
            'average_response_time': f"{self.metrics['average_response_time']:.3f}s",
            'fastest_response': f"{self.metrics['fastest_response']:.3f}s" if self.metrics['fastest_response'] != float('inf') else "N/A",
            'slowest_response': f"{self.metrics['slowest_response']:.3f}s",
            'total_tokens_used': self.metrics['total_tokens'],
            'model_used': self.model_name,
            'cache_size': len(self._cache)
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_response_time': 0.0,
            'total_tokens': 0,
            'fastest_response': float('inf'),
            'slowest_response': 0.0
        }
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        self._cache_timestamps.clear()
        self.logger.info("Cache cleared")
    
    def _del_(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Factory functions
def create_fast_classifier(api_key: Optional[str] = None, **kwargs) -> OptimizedProblemStatementClassifier:
    """Create a classifier optimized for speed."""
    defaults = {
        'model_name': 'llama3-8b-8192',  # Fastest model
        'max_retries': 2,
        'timeout': 10.0,
        'cache_size': 512,
        'max_concurrent': 15
    }
    defaults.update(kwargs)
    return OptimizedProblemStatementClassifier(api_key=api_key, **defaults)


def create_balanced_classifier(api_key: Optional[str] = None, **kwargs) -> OptimizedProblemStatementClassifier:
    """Create a classifier with balanced speed and quality."""
    defaults = {
        'model_name': 'llama3-70b-8192',  # Good balance
        'max_retries': 2,
        'timeout': 15.0,
        'cache_size': 256,
        'max_concurrent': 10
    }
    defaults.update(kwargs)
    return OptimizedProblemStatementClassifier(api_key=api_key, **defaults)


# Test samples
SAMPLE_DATA = [
    {
        'idea': 'Smart Water Management System: IoT-enabled sensors and mobile app for monitoring water quality and usage in rural communities, promoting SDG 6 (Clean Water and Sanitation).',
        'problem': 'Water scarcity in rural Maharashtra affects over 2.5 million people according to state government data from 2023. Villages like Ahmednagar face daily water shortages during summer months (April-June), forcing women and children to walk 3-5 kilometers daily to collect water. This impacts school attendance rates, which drop by 35% during peak summer, particularly affecting girls aged 10-14. Without proper monitoring systems, existing bore wells often run dry or produce contaminated water, leading to waterborne diseases that affect approximately 40% of households annually.'
    },
    {
        'idea': 'Community Recycling Hub: Local waste sorting and recycling center using AI-powered sorting technology to promote circular economy and SDG 12 (Responsible Consumption).',
        'problem': 'Urban waste management crisis in Delhi generates 11,000 tons of waste daily but only processes 61% effectively. Most residential areas lack proper segregation systems, leading to 85% mixed waste that ends up in landfills. This creates methane emissions equivalent to 2.3 million tons CO2 annually and affects air quality for 30 million residents. Local recyclers struggle with contaminated materials, reducing recycling efficiency to just 12% compared to global averages of 35%.'
    },
    {
        'idea': 'Digital Learning Platform for Girls: Mobile education app providing STEM courses in local languages to promote girls\' education and SDG 4 (Quality Education).',
        'problem': 'Gender disparity in STEM education affects millions of girls globally, with only 35% of STEM higher education students being female according to UNESCO data.'
    }
]


async def run_comprehensive_test():
    """Run comprehensive tests with sample data."""
    print("🚀 Testing Optimized Problem Statement Classifier with Groq API")
    print("=" * 70)
    
    # Create classifier instance
    try:
        classifier = create_balanced_classifier(log_level="INFO")
        print(f"✅ Classifier created successfully with model: {classifier.model_name}")
    except Exception as e:
        print(f"❌ Failed to create classifier: {e}")
        return
    
    # Test 1: Single classification
    print("\n📋 Test 1: Single Classification")
    print("-" * 40)
    
    sample = SAMPLE_DATA[0]
    start_time = time.time()
    
    result = await classifier.classify_async(sample['idea'], sample['problem'])
    
    if result.success:
        print("✅ Classification successful!")
        print(f"   Quality (X-Axis): {result.x_axis_category}")
        print(f"   Content (Y-Axis): {result.y_axis_category}")
        print(f"   Processing time: {result.processing_time:.3f}s")
        print(f"   Tokens used: {result.tokens_used}")
        print(f"   Model: {result.model_used}")
    else:
        print(f"❌ Classification failed: {result.error}")
    
    # Test 2: Batch classification
    print(f"\n📦 Test 2: Batch Classification ({len(SAMPLE_DATA)} samples)")
    print("-" * 50)
    
    batch_requests = [(sample['idea'], sample['problem']) for sample in SAMPLE_DATA]
    batch_start = time.time()
    
    batch_results = await classifier.classify_batch_async(batch_requests)
    batch_time = time.time() - batch_start
    
    print(f"⏱  Total batch time: {batch_time:.3f}s")
    print(f"📊 Results:")
    
    for i, result in enumerate(batch_results):
        if result.success:
            print(f"   Sample {i+1}: ✅ X: {result.x_axis_category[:30]}...")
            print(f"              Y: {result.y_axis_category[:30]}...")
            print(f"              Time: {result.processing_time:.3f}s")
        else:
            print(f"   Sample {i+1}: ❌ Error: {result.error}")
    
    # Test 3: Cache performance
    print(f"\n🔄 Test 3: Cache Performance Test")
    print("-" * 40)
    
    # Run same request twice to test caching
    sample = SAMPLE_DATA[1]
    
    # First call (cache miss)
    cache_start = time.time()
    result1 = await classifier.classify_async(sample['idea'], sample['problem'])
    first_time = time.time() - cache_start
    
    # Second call (should be cache hit)
    cache_start = time.time()
    result2 = await classifier.classify_async(sample['idea'], sample['problem'])
    second_time = time.time() - cache_start
    
    print(f"   First call (cache miss): {first_time:.3f}s")
    print(f"   Second call (cache hit): {second_time:.3f}s")
    print(f"   Speed improvement: {(first_time/second_time):.1f}x faster")
    print(f"   Results identical: {result1.x_axis_category == result2.x_axis_category and result1.y_axis_category == result2.y_axis_category}")
    
    # Test 4: Performance report
    print(f"\n📈 Performance Report")
    print("-" * 30)
    
    report = classifier.get_performance_report()
    for key, value in report.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n🎉 Testing completed successfully!")
    return classifier


def sync_test():
    """Synchronous test runner."""
    print("Running synchronous tests...")
    
    try:
        classifier = create_fast_classifier()
        
        # Quick sync test
        sample = SAMPLE_DATA[0]
        result = classifier.classify_sync(sample['idea'], sample['problem'])
        
        if result.success:
            print("✅ Sync test passed!")
            print(f"   Processing time: {result.processing_time:.3f}s")
            print(f"   X-Axis: {result.x_axis_category}")
            print(f"   Y-Axis: {result.y_axis_category}")
        else:
            print(f"❌ Sync test failed: {result.error}")
        
        print(f"\nPerformance: {classifier.get_performance_report()}")
        
    except Exception as e:
        print(f"❌ Sync test error: {e}")


if __name__ == "_main_":
    print("Choose test mode:")
    print("1. Async comprehensive test (recommended)")
    print("2. Quick sync test")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(run_comprehensive_test())
    else:
        sync_test()