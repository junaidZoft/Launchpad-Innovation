from __future__ import annotations

import os
import json
import time
import logging
import hashlib
import asyncio
from collections import OrderedDict
from typing import List, Tuple, Optional, Dict, Any

# ────────────────────────────────────────────────────────────────────────────
#  .env loading (silently ignored if python-dotenv is missing)
# ────────────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()               # pulls variables from .env into os.environ
except ImportError:
    pass                        # fallback to system environment variables only

# ────────────────────────────────────────────────────────────────────────────
#  Groq SDK
# ────────────────────────────────────────────────────────────────────────────
try:
    from groq import Groq, AsyncGroq
    from groq.types.chat import ChatCompletion
except ImportError as err:
    raise ImportError(
        "groq package not found. Install with:\n"
        "   pip install groq"
    ) from err

# ────────────────────────────────────────────────────────────────────────────
#  Exceptions
# ────────────────────────────────────────────────────────────────────────────
class ClassificationError(Exception):
    """Raised when classification fails or returns invalid data."""


# ────────────────────────────────────────────────────────────────────────────
#  Classifier
# ────────────────────────────────────────────────────────────────────────────
class OptimizedPrototypeClassifier:
    """
    Lightweight, LRU-cached classifier for student prototype descriptions,
    powered by Groq and Llama 3.

    Usage (async):
        classifier = OptimizedPrototypeClassifier()
        x_cat, y_cat = await classifier.classify(description)
    """

    # ------- Prompt template (constant) ------------------------------------
    # NOTE: The prompt is still crucial for guiding the model, even with JSON mode.
    _PROMPT: str = """IMPORTANT: Return EXACT strings from the lists below. Do not simplify. Your entire response must be a single, valid JSON object and nothing else.

**CONTENT SAFETY CHECK**: This tool is designed for students aged 12-15. If the description contains inappropriate content (violence, adult themes, harmful activities, or content unsuitable for middle school students), return:
{{"X_Axis_Rubric_Category": "Content Not Appropriate for Age Group", "Y_Axis_Rubric_Category": "Content Safety Exception"}}

**RELEVANCE CHECK**: If the description is not about a prototype, product, or invention suitable for a school project, return:
{{"X_Axis_Ruric_Category": "Not Relevant", "Y_Axis_Rubric_Category": "Not Relevant"}}

Analyze this student prototype description and classify it on TWO dimensions based on the provided rubrics.

**COMMUNICATION QUALITY (X-AXIS)** - Choose EXACTLY one string:
1. "Does not have grammar"
2. "Does not Demonstrate Understanding Only"
3. "Is not Precise and To the Point"
4. "Lacks Visual Clarity"
5. "Info is not Well-Structured and Is not Easy to Understand"
6. "Does not have Grammar + Lacks Visual Clarity"
7. "Has some Grammar"
8. "Demonstrates some Understanding"
9. "Is somewhat Precise and To the Point"
10. "Has some Visual Clarity"
11. "Info is somewhat Well-Structured and fairly Easy to Understand"
12. "Has some Grammar + Has some Visual Clarity"
13. "Has Very Good Grammar + Demonstrates Very Good Understanding"
14. "Has Very Good Grammar + Has Clear Visual Clarity"
15. "Is Precise and To the Point + Has Clear Visual Clarity"
16. "Has Very Good Grammar + Demonstrates Very Good Understanding + Has Clear Visual Clarity"

**PROJECT COMPLETENESS (Y-AXIS)** - Choose EXACTLY one string:
1. "Clear Product/Service Description"
2. "Prototype Status"
3. "Unique Value Statement"
4. "User Interaction Description"
5. "Technical/Practical Feasibility Evidence"
6. "Product Description + Prototype Status"
7. "Product Description + Unique Value Statement"
8. "Product Description + User Interaction Description"
9. "Product Description + Technical/Practical Feasibility Evidence"
10. "Prototype Status + Unique Value Statement"
11. "Prototype Status + User Interaction Description"
12. "Prototype Status + Technical/Practical Feasibility Evidence"
13. "Unique Value Statement + User Interaction Description"
14. "Unique Value Statement + Technical/Practical Feasibility Evidence"
15. "User Interaction Description + Technical/Practical Feasibility Evidence"
16. "Product Description + Prototype Status + Unique Value Statement"
17. "Product Description + Prototype Status + User Interaction Description"
18. "Product Description + Prototype Status + Technical/Practical Feasibility Evidence"
19. "Product Description + Unique Value Statement + User Interaction Description"
20. "Product Description + Unique Value Statement + Technical/Practical Feasibility Evidence"
21. "Product Description + User Interaction Description + Technical/Practical Feasibility Evidence"
22. "Prototype Status + Unique Value Statement + User Interaction Description"
23. "Prototype Status + Unique Value Statement + Technical/Practical Feasibility Evidence"
24. "Prototype Status + User Interaction Description + Technical/Practical Feasibility Evidence"
25. "Unique Value Statement + User Interaction Description + Technical/Practical Feasibility Evidence"
26. "Product Description + Prototype Status + Unique Value Statement + User Interaction Description"
27. "Product Description + Prototype Status + Unique Value Statement + Technical/Practical Feasibility Evidence"
28. "Product Description + Prototype Status + User Interaction Description + Technical/Practical Feasibility Evidence"
29. "Product Description + Unique Value Statement + User Interaction Description + Technical/Practical Feasibility Evidence"
30. "Prototype Status + Unique Value Statement + User Interaction Description + Technical/Practical Feasibility Evidence"
31. "Clear Product/Service Description + Prototype Status + Unique Value Statement + User Interaction Description + Technical/Practical Feasibility Evidence"

**STUDENT DESCRIPTION TO ANALYZE:**
{description}

Return ONLY a valid JSON object with keys "X_Axis_Rubric_Category" and "Y_Axis_Rubric_Category".
"""

    # ------- Constructor ---------------------------------------------------
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "llama3-70b-8192",  # Note: Use a valid model from Groq's platform
        *,
        max_retries: int = 2,
        timeout: float = 20.0,
        cache_size: int = 50,
        log_level: str = "INFO",
    ) -> None:
        # Logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        # API key from arg or .env
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ClassificationError(
                "GROQ_API_KEY not set. Add it to your .env or environment variables."
            )

        # Groq configuration (sync and async clients)
        self.client = Groq(api_key=api_key)
        self.async_client = AsyncGroq(api_key=api_key)
        self.model_name = model_name

        # Runtime settings
        self.max_retries = max_retries
        self.timeout = timeout
        self.generation_params: Dict[str, Any] = {
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 512,
            "top_p": 0.9,
        }

        # Simple LRU cache
        self._cache: OrderedDict[str, Tuple[str, str]] = OrderedDict()
        self._cache_size = cache_size

    # ------- Helpers -------------------------------------------------------
    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:16]

    def _parse_response_and_update_cache(
        self, response_text: str, cache_key: str
    ) -> Tuple[str, str]:
        """Parses the model's JSON response and updates the cache."""
        # The response is already clean JSON thanks to response_format
        payload = json.loads(response_text)
        x_cat = payload["X_Axis_Rubric_Category"]
        y_cat = payload["Y_Axis_Rubric_Category"]

        # Update cache
        self._cache[cache_key] = (x_cat, y_cat)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)  # evict oldest
        
        self.logger.debug(f"Cache miss - added key {cache_key}")
        return x_cat, y_cat

    # ------- Public API (Async) --------------------------------------------
    async def classify(self, description: str) -> Tuple[str, str]:
        """
        Classify a single prototype description.

        Returns:
            (x_category, y_category)
        """
        description = description.strip()
        if not description:
            raise ClassificationError("Description cannot be empty.")

        key = self._hash(description)
        if key in self._cache:
            self._cache.move_to_end(key)
            self.logger.debug(f"Cache hit for key {key}")
            return self._cache[key]

        prompt = self._PROMPT.format(description=description[:2000])
        last_err: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                start = time.monotonic()
                response: ChatCompletion = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=self.timeout,
                    **self.generation_params
                )
                self.logger.debug(f"Async model latency: {time.monotonic() - start:.2f}s")
                content = response.choices[0].message.content
                if not content:
                    raise ClassificationError("API returned an empty response.")
                return self._parse_response_and_update_cache(content, key)

            except Exception as err:
                last_err = err
                self.logger.warning(f"Attempt {attempt}/{self.max_retries} failed (async): {err}")
                await asyncio.sleep(1)

        raise ClassificationError(
            f"Classification failed after {self.max_retries} attempts: {last_err}"
        )

    async def classify_batch(
        self,
        descriptions: List[str],
        *,
        max_concurrent: int = 10, # Groq is fast, can often handle more
    ) -> List[Tuple[str, str]]:
        """Classify multiple descriptions concurrently (bounded concurrency)."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _worker(desc: str) -> Tuple[str, str]:
            async with semaphore:
                return await self.classify(desc)

        tasks = [_worker(d) for d in descriptions]
        return await asyncio.gather(*tasks)

    # ------- Synchronous API (Fallback) ------------------------------------
    def classify_sync(self, description: str) -> Tuple[str, str]:
        """
        Synchronous version of classify() for non-async environments.
        """
        description = description.strip()
        if not description:
            raise ClassificationError("Description cannot be empty.")

        key = self._hash(description)
        if key in self._cache:
            self._cache.move_to_end(key)
            self.logger.debug(f"Cache hit for key {key}")
            return self._cache[key]

        prompt = self._PROMPT.format(description=description[:2000])
        last_err: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                start = time.monotonic()
                response: ChatCompletion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=self.timeout,
                    **self.generation_params
                )
                self.logger.debug(f"Sync model latency: {time.monotonic() - start:.2f}s")
                content = response.choices[0].message.content
                if not content:
                    raise ClassificationError("API returned an empty response.")
                return self._parse_response_and_update_cache(content, key)

            except Exception as err:
                last_err = err
                self.logger.warning(f"Attempt {attempt}/{self.max_retries} failed (sync): {err}")
                time.sleep(1)

        raise ClassificationError(
            f"Classification failed after {self.max_retries} attempts: {last_err}"
        )


# ────────────────────────────────────────────────────────────────────────────
#  Quick Demo (run `python your_script_name.py` to test)
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    async def _demo() -> None:
        # Create a dummy .env file for the demo if it doesn't exist
        if not os.path.exists(".env") and not os.getenv("GROQ_API_KEY"):
            print("WARNING: GROQ_API_KEY not found.")
            print("Please create a .env file with GROQ_API_KEY='your-key' to run the demo.")
            return
        
        # Note: You can change the model here if you wish, e.g., to "llama3-8b-8192" for testing
        classifier = OptimizedPrototypeClassifier(model_name="llama3-70b-8192")

        sample = (
            "Our prototype is a smart water bottle that tracks daily hydration, "
            "notifies users via an LED light ring, and syncs with a mobile app "
            "over Bluetooth to show historical data. It's built with an Arduino Nano, "
            "a flow meter, and a custom 3D printed case. The goal is to make "
            "tracking water intake effortless."
        )

        try:
            print("--- Running Async Classification (1st time) ---")
            x, y = await classifier.classify(sample)
            print("X-Axis Category:", x)
            print("Y-Axis Category:", y)
            print()
            
            print("--- Running Sync Classification (should be a cache hit) ---")
            start_time = time.monotonic()
            x_sync, y_sync = classifier.classify_sync(sample)
            duration = time.monotonic() - start_time
            print(f"Completed in {duration:.6f} seconds.")
            print("X-Axis Category:", x_sync)
            print("Y-Axis Category:", y_sync)
            print()

            print("--- Running Batch Classification ---")
            samples = [
                "a simple todo list app",
                "a robot that waters plants based on soil moisture",
                "this is not a prototype description its just a random sentence",
            ]
            results = await classifier.classify_batch(samples)
            for i, (desc, (x_res, y_res)) in enumerate(zip(samples, results)):
                print(f"Result for sample {i+1} ('{desc[:30]}...'):")
                print(f"  X: {x_res}")
                print(f"  Y: {y_res}")

        except ClassificationError as err:
            print("ERROR:", err)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    asyncio.run(_demo())