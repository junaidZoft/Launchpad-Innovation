import streamlit as st
import os
import json
from typing import List
from dotenv import load_dotenv
from tavily import TavilyClient
# Changed import from google.generativeai to openai
from groq import Groq

try:
    # Assuming the OpenAI-compatible classifier is saved as classifier.py
    from utlis_groq import classify_problem_statement
except ImportError:
    st.error("Error importing the 'classify_problem_statement' function from classifier.py. Please ensure the file is in the same directory.")
    st.stop()

# === Load API keys ===
load_dotenv()
# Changed from GEMINI_API_KEY to OPENAI_API_KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# === Configure APIs ===
@st.cache_resource
def setup_apis():
    """Setup and cache API clients"""
    # Configure Groq client
    groq_client = Groq(api_key=GROQ_API_KEY)
    
    # Configure Tavily client
    tavily_client = None
    if TAVILY_API_KEY:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        
    return groq_client, tavily_client

# === SDG List (Unchanged) ===
SDG_LIST = [
    "No Poverty", "Zero Hunger", "Good Health and Well-being", "Quality Education",
    "Gender Equality", "Clean Water and Sanitation", "Affordable and Clean Energy",
    "Decent Work and Economic Growth", "Industry, Innovation and Infrastructure",
    "Reduced Inequalities", "Sustainable Cities and Communities",
    "Responsible Consumption and Production", "Climate Action", "Life Below Water",
    "Life on Land", "Peace, Justice and Strong Institutions", "Partnerships for the Goals"
]

# === Problem Statement Evaluation Criteria (Unchanged) ===
PROBLEM_STATEMENT_CRITERIA = """
EFFECTIVE PROBLEM STATEMENT ASSESSMENT CRITERIA
1. **Contains Data**: The problem statement includes relevant quantitative or qualitative data supporting the existence of the problem.
2. **References Included**: The statement cites credible sources or references that validate the problem.
3. **Location/Area Clear**: Clearly specifies the geographical location or specific area where the problem exists.
4. **Target Audience Clearly Stated**: Defines the specific group or demographic affected by the problem.
5. **Impact Described**: Explains the consequences or negative impact if the problem remains unaddressed.
6. **GRAMMAR**: The response uses correct grammar, spelling, and punctuation throughout.
7. **DEMONSTRATES UNDERSTANDING**: The answer reflects a clear comprehension of the question or topic, showing insight and awareness.
8. **PRECISE AND TO THE POINT**: The information is concise, avoiding unnecessary details and focusing on the core message.
9. **RELEVANT TO THE IDEA**: The content directly relates to and supports the main idea or purpose being assessed.
10. **INFO IS WELL-STRUCTURED AND EASY TO UNDERSTAND**: The response is logically organized, making it straightforward and accessible for the reader to follow.
"""

# === Market Fit Evaluation Rubric (Unchanged) ===
MARKET_FIT_RUBRIC = """Student Response Evaluation Template**

You are a supportive business mentor evaluating a young entrepreneur's market analysis. A student (aged 14-15) has submitted a response about their business idea. Please evaluate their work using this 10-point rubric, providing encouraging yet constructive feedback.

**Evaluation Criteria (1-10 scale, where 1 = needs significant improvement, 10 = excellent):**

1. **Target Audience Clarity** - Do they clearly identify who their customers are and what those customers need?
2. **Problem-Solution Connection** - Do they explain how their idea solves a real problem for their target customers?
3. **Market Research Evidence** - Do they include any supporting data, research, or validation (surveys, interviews, observations)?
4. **Unique Value Proposition** - Do they explain what makes their idea different or better than existing solutions?
5. **Market Entry Strategy** - Do they outline realistic first steps for launching their idea (MVP, pilot testing, initial customers)?
6. **Communication Quality** - Is their writing clear with proper grammar, spelling, and punctuation?
7. **Business Understanding** - Do they demonstrate solid comprehension of basic business concepts?
8. **Focus and Conciseness** - Do they stay on topic and communicate efficiently without unnecessary details?
9. **Relevance and Consistency** - Does all content directly relate to supporting their business idea?
10. **Organization and Clarity** - Is their response well-structured and easy to follow?

**Feedback Guidelines:**
- Always address the student directly using "you" and "your"
- Use encouraging, constructive language appropriate for teenagers
- Provide specific examples from their response
- Offer concrete suggestions for improvement
- Acknowledge strengths before addressing areas for growth
- Format feedback as numbered points (1-10) with scores and detailed comments

Remember: These students are learning entrepreneurship basics. Your goal is to build their confidence while developing their business thinking and communication skills.
"""
# === Helper Functions (Updated for Groq) ===
def generate_project_ideas(selected_sdgs: List[str]) -> str:
    """Generate project ideas based on selected SDGs using Groq"""
    groq_client, _ = setup_apis()
    
    system_prompt = "You are an educational assistant helping students brainstorm project ideas."
    user_prompt = f"""
    Generate 5 student-friendly, realistic project ideas based on the following Sustainable Development Goals: {', '.join(selected_sdgs)}.
    Each idea should be:
    - Feasible for students to implement
    - Ethical and socially responsible
    - Involve either technology or social innovation
    - Clearly address one or more of the selected SDGs
    
    Format the ideas as a numbered list.
    Strictly avoid any harmful or dangerous content.
    """
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating ideas: {str(e)}")
        return ""

def evaluate_problem_statement(idea: str, problem_statement: str) -> dict:
    """Evaluate problem statement by calling the imported classifier function"""
    try:
        # This function now calls the OpenAI-based classifier from classifier.py
        result = classify_problem_statement(idea, problem_statement)
        print(f"Evaluation Result: {result}")
        if result['success']:
           return {
               "success": True,
               "evaluation": result['data'],
           }
        else:
           return {
               "success": False,
               "error": result.get('error', 'An unknown error occurred in the classifier.')
           }
    except Exception as e:
        print(f"Error evaluating problem statement: {str(e)}")
        return {"success": False, "error": str(e)}

def generate_market_research(selected_sdgs: List[str], idea: str, problem_statement: str, 
                           target_market: str, research_question: str) -> dict:
    """Generate market research insights using OpenAI and Tavily"""
    openai_client, tavily_client = setup_apis()
    
    try:
        # Tavily Search (if available)
        web_summary = "No web search available - Tavily API key not configured."
        source_urls = []
        
        if tavily_client:
            search_query = f"{research_question} for {target_market} related to SDGs {' '.join(selected_sdgs)} market research"
            with st.spinner("Searching the web for latest insights..."):
                try:
                    tavily_result = tavily_client.search(query=search_query, include_answer=True, include_sources=True, search_depth="advanced")
                    web_summary = tavily_result.get("answer", "No summary available.")
                    sources = tavily_result.get("sources", [])
                    source_urls = [src.get('url', '') for src in sources if src.get('url')]
                except Exception as e:
                    st.warning(f"Web search error: {e}")
                    web_summary = "No summary available due to search API error."
        
        # OpenAI Analysis
        system_prompt = "You are a market research analyst."
        user_prompt = f"""
        Based on the information provided, generate comprehensive market research insights.

        Web Research Summary: {web_summary}
        Web Sources: {', '.join(source_urls) if source_urls else "N/A"}
        
        Project Details:
        - SDGs: {', '.join(selected_sdgs)}
        - Idea: {idea}
        - Problem Statement: {problem_statement}
        - Target Market: {target_market}
        - Research Question: {research_question}

        Please provide:
        1. Detailed market research insights
        2. Competitor analysis
        3. Market opportunities and challenges
        4. Recommendations for market entry

        Format your response clearly with these sections.
        """
        
        with st.spinner("Generating market research insights with OpenAI..."):
            response = openai_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            return {
                "success": True,
                "web_summary": web_summary,
                "market_research": response.choices[0].message.content.strip(),
                "web_sources": source_urls,
                "sdgs": selected_sdgs, "idea": idea, "problem_statement": problem_statement,
                "target_market": target_market, "research_question": research_question
            }
    
    except Exception as e:
        return {"success": False, "error": str(e)}

def generate_presentation_questions(idea: str, problem_statement: str, market_research: str) -> List[str]:
    """Generate student presentation questions using OpenAI"""
    openai_client, _ = setup_apis()
    
    system_prompt = "You are a presentation coach helping a student prepare."
    user_prompt = f"""
    Generate 5 short, direct questions that a student could ask their audience during a presentation about their project.

    Project Idea: {idea}
    Problem Statement: {problem_statement}
    Market Research: {market_research}

    Each question should:
    - Start with question words (What, How, Why, When, Where, Which, Who)
    - Be one sentence only
    - Be concise and clear
    - Focus on getting actionable feedback from the audience
    - Relate to the problem, solution, and market context

    Return only a numbered list of 5 questions.
    """
    
    try:
        response = openai_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        questions = []
        for line in response.choices[0].message.content.strip().split("\n"):
            if line.strip() and line[0].isdigit():
                question_text = line.split(".", 1)[1].strip()
                questions.append(question_text)
        return questions
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return []

def evaluate_market_fit(student_response: str) -> str:
    """Evaluate student's market fit response using OpenAI"""
    openai_client, _ = setup_apis()
    
    try:
        response = openai_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": MARKET_FIT_RUBRIC},
                {"role": "user", "content": f"Student Response:\n{student_response.strip()}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating feedback: {e}"

# === Streamlit App (Updated for OpenAI) ===
def main():
    st.set_page_config(page_title="Integrated SDG Student Platform", page_icon="🌍", layout="wide")
    
    st.title("🌍 Integrated SDG Student Platform")
    st.markdown("Complete workflow: Select SDGs → Choose Ideas → Write Problem Statement → Evaluate → Market Research → Generate Questions → Market Fit Analysis")
    

    
    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 1
    
    # Progress indicator
    steps = ["Select SDGs", "Choose Idea", "Problem Statement", "Evaluation", "Market Research", "Questions", "Market Fit"]
    current_step = st.session_state.step
    
    progress = current_step / len(steps)
    st.progress(progress)
    st.write(f"**Step {current_step}/{len(steps)}: {steps[current_step-1]}**")
    
    # --- Workflow Steps ---
    
    if current_step == 1:
        st.header("🎯 Step 1: Select SDGs")
        st.write("Choose up to 3 Sustainable Development Goals for your project:")
        
        if 'selected_sdgs' not in st.session_state:
            st.session_state.selected_sdgs = []
        
        col1, col2 = st.columns(2)
        selected_sdgs = []
        
        for i, sdg in enumerate(SDG_LIST):
            col_idx = i % 2
            with [col1, col2][col_idx]:
                disabled = len(st.session_state.selected_sdgs) >= 3 and sdg not in st.session_state.selected_sdgs
                if st.checkbox(sdg, value=sdg in st.session_state.selected_sdgs, disabled=disabled, key=f"sdg_{sdg}"):
                    selected_sdgs.append(sdg)
        
        st.session_state.selected_sdgs = selected_sdgs
        
        if selected_sdgs:
            st.success(f"✅ Selected {len(selected_sdgs)}/3 SDGs: {', '.join(selected_sdgs)}")
        else:
            st.info("Please select at least 1 SDG")
        
        if st.button("Next: Generate Ideas", type="primary", disabled=not selected_sdgs):
            st.session_state.step = 2
            st.rerun()
    
    elif current_step == 2:
        st.header("💡 Step 2: Choose Your Project Idea")
        st.write(f"**Selected SDGs:** {', '.join(st.session_state.selected_sdgs)}")
        
        if st.button("🚀 Generate Project Ideas", type="primary"):
            with st.spinner("Generating project ideas with OpenAI..."):
                st.session_state.generated_ideas = generate_project_ideas(st.session_state.selected_sdgs)
        
        if 'generated_ideas' in st.session_state:
            st.subheader("Generated Project Ideas:")
            st.write(st.session_state.generated_ideas)
            
            chosen_idea = st.text_area("Describe your chosen project idea:", height=150, placeholder="Enter your selected idea or modify one...")
            if chosen_idea and st.button("Next: Write Problem Statement", type="primary"):
                st.session_state.chosen_idea = chosen_idea
                st.session_state.step = 3
                st.rerun()
        
        if st.button("← Back to SDG Selection"):
            st.session_state.step = 1
            st.rerun()
    
    elif current_step == 3:
        st.header("📝 Step 3: Write Problem Statement")
        st.write(f"**Selected SDGs:** {', '.join(st.session_state.selected_sdgs)}")
        st.write(f"**Chosen Idea:** {st.session_state.chosen_idea}")
        
        with st.expander("📋 Problem Statement Criteria", expanded=True):
            st.write(PROBLEM_STATEMENT_CRITERIA)
        
        problem_statement = st.text_area("Write your problem statement:", height=200, placeholder="Write a comprehensive problem statement...")
        
        if problem_statement and st.button("Next: Evaluate Problem Statement", type="primary"):
            st.session_state.problem_statement = problem_statement
            st.session_state.step = 4
            st.rerun()
        
        if st.button("← Back to Choose Idea"):
            st.session_state.step = 2
            st.rerun()
            
    elif current_step == 4:
        st.header("📊 Step 4: Problem Statement Evaluation")
        st.write(f"**Idea:** {st.session_state.chosen_idea}")
        st.write(f"**Problem Statement:** {st.session_state.problem_statement}")

        if 'evaluation_result' not in st.session_state:
            with st.spinner("Evaluating your problem statement with OpenAI..."):
                evaluation = evaluate_problem_statement(st.session_state.chosen_idea, st.session_state.problem_statement)
                if evaluation['success']:
                    st.session_state.evaluation_result = evaluation['evaluation']
                    st.success("✅ Evaluation completed!")
                else:
                    st.error(f"Evaluation failed: {evaluation['error']}")
                    st.session_state.evaluation_result = None # Mark as failed

        if st.session_state.get('evaluation_result'):
            st.json(st.session_state.evaluation_result)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Problem Statement"):
                st.session_state.step = 3
                del st.session_state.evaluation_result
                st.rerun()
        with col2:
            if 'evaluation_result' in st.session_state and st.session_state.evaluation_result:
                if st.button("Next: Market Research", type="primary"):
                    st.session_state.step = 5
                    st.rerun()

    elif current_step == 5:
        st.header("🔍 Step 5: Market Research")
        
        col1, col2 = st.columns(2)
        with col1:
            target_market = st.text_input("Target Market:", placeholder="e.g., Small farmers in rural areas")
        with col2:
            research_question = st.text_input("Research Question:", placeholder="What market insights do you want to discover?")
        
        if st.button("🚀 Generate Market Research", type="primary", disabled=not (target_market and research_question)):
            research_results = generate_market_research(st.session_state.selected_sdgs, st.session_state.chosen_idea, st.session_state.problem_statement, target_market, research_question)
            if research_results['success']:
                st.session_state.market_research = research_results
                st.success("✅ Market research completed!")
            else:
                st.error(f"Market research failed: {research_results['error']}")
        
        if 'market_research' in st.session_state:
            res = st.session_state.market_research
            with st.expander("🌐 Web Research Summary", expanded=True): st.write(res['web_summary'])
            with st.expander("📊 Market Research Analysis", expanded=True): st.write(res['market_research'])
            if res['web_sources']:
                with st.expander("🔗 Sources"):
                    for i, url in enumerate(res['web_sources'], 1): st.write(f"{i}. {url}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Evaluation"):
                st.session_state.step = 4
                st.rerun()
        with col2:
            if 'market_research' in st.session_state:
                if st.button("Next: Generate Questions", type="primary"):
                    st.session_state.step = 6
                    st.rerun()

    elif current_step == 6:
        st.header("❓ Step 6: Presentation Questions")
        
        if st.button("🎯 Generate Presentation Questions", type="primary"):
            with st.spinner("Generating questions with OpenAI..."):
                questions = generate_presentation_questions(st.session_state.chosen_idea, st.session_state.problem_statement, st.session_state.market_research['market_research'])
                if questions:
                    st.session_state.presentation_questions = questions
                    st.success("✅ Questions generated!")
        
        if 'presentation_questions' in st.session_state:
            st.subheader("Questions for Your Presentation:")
            for i, question in enumerate(st.session_state.presentation_questions, 1): st.write(f"{i}. {question}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Market Research"):
                st.session_state.step = 5
                st.rerun()
        with col2:
            if 'presentation_questions' in st.session_state:
                if st.button("Next: Market Fit Analysis", type="primary"):
                    st.session_state.step = 7
                    st.rerun()

    elif current_step == 7:
        st.header("📈 Step 7: Market Fit Analysis")
        st.info("*Write why you believe your idea is needed in the market and how your idea is unique. Use any data or current knowledge you have. Outline how you will enter the market.*")
        
        market_fit_response = st.text_area("Your Market Fit Analysis:", height=300, placeholder="Write your analysis here...")
        
        if st.button("📊 Get Feedback", type="primary", disabled=not market_fit_response):
            with st.spinner("Analyzing your response with OpenAI..."):
                feedback = evaluate_market_fit(market_fit_response)
                st.session_state.market_fit_feedback = feedback
        
        if 'market_fit_feedback' in st.session_state:
            st.success("✅ Feedback generated!")
            st.markdown("### 📋 Feedback:")
            st.write(st.session_state.market_fit_feedback)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Questions"):
                st.session_state.step = 6
                st.rerun()
        with col2:
            if st.button("🎉 Complete Project"):
                st.balloons()
                st.success("🎉 Congratulations! You've completed all steps of the SDG project workflow!")
                with st.expander("📋 Project Summary", expanded=True):
                    # Check if session state variables exist before accessing them
                    if 'selected_sdgs' in st.session_state:
                        st.write(f"**SDGs:** {', '.join(st.session_state.selected_sdgs)}")
                    if 'chosen_idea' in st.session_state:
                        st.write(f"**Idea:** {st.session_state.chosen_idea}")
                    if 'problem_statement' in st.session_state:
                        st.write(f"**Problem Statement:** {st.session_state.problem_statement}")
                    if 'market_research' in st.session_state:
                        st.write(f"**Target Market:** {st.session_state.market_research['target_market']}")
                    if 'presentation_questions' in st.session_state:
                        st.write("**Presentation Questions:**")
                        for i, q in enumerate(st.session_state.presentation_questions, 1): 
                            st.write(f"{i}. {q}")
    
        # --- Sidebar ---
        with st.sidebar:
            st.header("🎯 Progress")
            for i, step_name in enumerate(steps, 1):
                if i < current_step: 
                    st.write(f"✅ {i}. {step_name}")
                elif i == current_step: 
                    st.write(f"📍 {i}. {step_name}")
                else: 
                    st.write(f"⭕ {i}. {step_name}")
            
            st.markdown("---")
            
            if st.button("🔄 Reset All", type="secondary"):
                keys_to_clear = list(st.session_state.keys())
                for key in keys_to_clear:
                    del st.session_state[key]
                st.rerun()
            
            st.header("🔌 API Status")
            st.write("✅ Groq (Llama 70B): Connected")
            # Check if TAVILY_API_KEY is defined before using it
            try:
                tavily_status = "✅ Tavily Search: Connected" if TAVILY_API_KEY else "⚠️ Tavily Search: Not configured"
            except NameError:
                tavily_status = "⚠️ Tavily Search: Not configured (API key not defined)"
            st.write(tavily_status)

if __name__ == "__main__":
    main()