import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
from tavily import TavilyClient
from groq import Groq
import logging

#configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

def check_input_consistency(
    target_market: str,
    market_summary: str,
    idea: str,
    problem_statement: str
) -> str:
    """
    Check if the business inputs are consistent with each other.
    
    Args:
        target_market: Who is your target market
        market_summary: What you already know about the market
        idea: The business idea
        problem_statement: The problem your idea solves
    
    Returns:
        Consistency check result as string
    """
    try:
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        if not GROQ_API_KEY:
            return "Error: Missing GROQ_API_KEY environment variable. Please check your .env file."
        
        groq_client = Groq(api_key=GROQ_API_KEY)
        
        consistency_prompt = f"""You're analyzing a business idea for consistency. Check if these inputs align well:

Target Market: {target_market}
Market Knowledge: {market_summary}
Business Idea: {idea}
Problem Statement: {problem_statement}

Respond with ONLY one of these formats:
"✅ CONSISTENT: All inputs align well and make sense together."
"❌ INCONSISTENT: [Brief explanation of what doesn't match]"

Keep response under 50 words."""

        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": consistency_prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=100
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Error checking consistency: {str(e)}"


def conduct_market_research(
    target_market: str,
    market_summary: str,
    idea: str,
    problem_statement: str
) -> Dict[str, Any]:
    """
    Conduct comprehensive market research using web search and AI analysis.
    
    Args:
        target_market: Who is your target market
        market_summary: What you already know about the market
        idea: The business idea
        problem_statement: The problem your idea solves
    
    Returns:
        Dictionary containing market analysis and web search results
    """
    try:
        # Get API keys from environment variables (loaded from .env)
        TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        
        if not TAVILY_API_KEY or not GROQ_API_KEY:
            return {
                "error": "Missing required API keys. Please ensure TAVILY_API_KEY and GROQ_API_KEY are set in your .env file.",
                "instructions": "Create a .env file with:\nTAVILY_API_KEY=your_tavily_key_here\nGROQ_API_KEY=your_groq_key_here"
            }
        
        tavily = TavilyClient(api_key=TAVILY_API_KEY)
        groq_client = Groq(api_key=GROQ_API_KEY)
        
        # Perform web search
        search_query = f"market trends competitors analysis {target_market} {idea}"
        logging.info(f"Conducting web search with query: {search_query}")
        logging.info(f"search_query length: {len(search_query)}")
        
        if len(search_query) >= 400:
            # cut the search query to fit within the limit
            search_query = search_query[:400]
        
        search_results = tavily.search(query=search_query, max_results=5)
        
        # Process search results
        web_info = ""
        sources = []
        for result_item in search_results.get("results", []):
            title = result_item.get("title", "")
            content = result_item.get("content", "")
            url = result_item.get("url", "")
            
            web_info += f"Title: {title}\nContent: {content}\nURL: {url}\n\n"
            sources.append({"title": title, "url": url, "content": content[:200] + "..."})
        
        if not web_info:
            web_info = "No relevant market data found from web search."
        
        # Generate market analysis
        market_prompt = f"""You're a senior business analyst creating a comprehensive market report.

Business Context:
- Target Market: {target_market}
- Current Knowledge: {market_summary}
- Business Idea: {idea}
- Problem Statement: {problem_statement}

Real Market Data from Web Search:
{web_info}

Create a detailed market analysis with these sections:

## Executive Summary
Brief overview of market opportunity (2-3 sentences)

## Market Trends 🚀
Identify 4-5 key trends affecting this market using the web data. Include specific examples and statistics where available.

## Competitive Landscape 🏢
List 4-6 real competitors or similar businesses from the search results:
- Company name and brief description
- Their target market
- Key differentiators

## Market Size & Opportunity 📊
Estimate market size, growth potential, and opportunity gaps based on available data.

## Target Audience Insights 👥
Detailed analysis of the target market including demographics, behaviors, and pain points.

## Market Entry Strategy 🎯
Recommend 3-4 specific strategies for entering this market successfully.

## Risk Analysis ⚠️
Identify 2-3 potential market risks and mitigation strategies.

Use data from the web search to support your analysis. Keep it professional and comprehensive, around 600-800 words total."""

        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": market_prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.5,
            max_tokens=1000
        )
        
        return {
            "market_analysis": response.choices[0].message.content.strip(),
            "web_sources": sources,
            "search_query": search_query,
            "metadata": {
                "target_market": target_market,
                "idea": idea,
                "problem_statement": problem_statement,
                "timestamp": "2025-07-22"
            }
        }
        
    except Exception as e:
        return {"error": f"Error conducting market research: {str(e)}"}


def generate_market_insights(
    target_market: str,
    market_summary: str,
    idea: str,
    problem_statement: str
) -> Dict[str, Any]:
    """
    Main function to generate complete market research insights.
    
    Args:
        target_market: Who is your target market
        market_summary: What you already know about the market
        idea: The business idea
        problem_statement: The problem your idea solves
    
    Returns:
        Complete market research results including consistency check and analysis
    """
    # Check input consistency
    consistency_result = check_input_consistency(target_market, market_summary, idea, problem_statement)
    
    # Conduct market research
    research_results = conduct_market_research(target_market, market_summary, idea, problem_statement)
    
    return {
        "consistency_check": consistency_result,
        **research_results
    }


def verify_env_setup() -> Dict[str, bool]:
    """
    Verify that required environment variables are properly loaded.
    
    Returns:
        Dictionary showing status of each required environment variable
    """
    required_vars = ["TAVILY_API_KEY", "GROQ_API_KEY"]
    status = {}
    
    for var in required_vars:
        value = os.getenv(var)
        status[var] = {
            "exists": value is not None,
            "has_value": bool(value and value.strip()),
            "preview": f"{value[:8]}..." if value and len(value) > 8 else value
        }
    
    return status


# Example usage and test function
def test_market_research():
    """Test the market research functions with sample data"""
    
    print("=== ENVIRONMENT SETUP CHECK ===")
    env_status = verify_env_setup()
    for var, status in env_status.items():
        status_icon = "✅" if status["has_value"] else "❌"
        print(f"{status_icon} {var}: {'Found' if status['has_value'] else 'Missing'}")
        if status["has_value"]:
            print(f"   Preview: {status['preview']}")
    print()
    
    test_data = {
        "target_market": "Health-conscious teenagers and young adults aged 13-25",
        "market_summary": "Many young people want healthy snack options but struggle to find convenient, affordable, and tasty alternatives to junk food.",
        "idea": "A subscription service that delivers personalized healthy snack boxes with gamification elements",
        "problem_statement": "Young people lack access to convenient, affordable, and appealing healthy snack options"
    }
    
    print("=== TESTING MARKET RESEARCH ===")
    
    # Test consistency check
    consistency = check_input_consistency(**test_data)
    print(f"Consistency Check: {consistency}\n")
    
    # Test full market research
    results = generate_market_insights(**test_data)
    
    print("=== MARKET RESEARCH RESULTS ===")
    print(f"Consistency: {results.get('consistency_check')}")
    
    if "error" in results:
        print(f"Error: {results['error']}")
        if "instructions" in results:
            print(f"\nSetup Instructions:\n{results['instructions']}")
    else:
        print(f"\nMarket Analysis:\n{results.get('market_analysis')}")
        print(f"\nSources Found: {len(results.get('web_sources', []))}")
    
    return results


if __name__ == "__main__":
    # Test the market research with environment variable loading
    test_market_research()