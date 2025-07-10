import streamlit as st
import os
import json
from typing import List, Optional
from dotenv import load_dotenv
from tavily import TavilyClient
import google.generativeai as genai

try:
    from utilis_gemini import classify_problem_statement
except ImportError:
    st.error("Error importing marking_ps_gemini module. Please ensure it is installed and accessible.")

# === Load API keys ===
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# === Configure APIs ===
@st.cache_resource
def setup_apis():
    """Setup and cache API clients"""
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    tavily_client = None
    if TAVILY_API_KEY:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    return gemini_model, tavily_client

# === SDG List ===
SDG_LIST = [
    "No Poverty",
    "Zero Hunger", 
    "Good Health and Well-being",
    "Quality Education",
    "Gender Equality",
    "Clean Water and Sanitation",
    "Affordable and Clean Energy",
    "Decent Work and Economic Growth",
    "Industry, Innovation and Infrastructure",
    "Reduced Inequalities",
    "Sustainable Cities and Communities",
    "Responsible Consumption and Production",
    "Climate Action",
    "Life Below Water",
    "Life on Land",
    "Peace, Justice and Strong Institutions",
    "Partnerships for the Goals"
]

# === Problem Statement Evaluation Criteria ===
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

# === Market Fit Evaluation Rubric ===
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

# === Helper Functions ===
def generate_project_ideas(selected_sdgs: List[str]) -> str:
    """Generate project ideas based on selected SDGs"""
    gemini_model, _ = setup_apis()
    
    prompt = f"""
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
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating ideas: {str(e)}")
        return ""

def evaluate_problem_statement(idea: str, problem_statement: str) -> dict:
    """Evaluate problem statement against the criteria"""
    gemini_model, _ = setup_apis()
    
    try:
        result = classify_problem_statement(idea, problem_statement)
        if result['success']:
           print(f"Evaluation Result: {result['data']}")
           return {
               "success": True,
               "evaluation": result['data'],
           }
    except Exception as e:
        print(f"Error evaluating problem statement: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }
        

def generate_market_research(selected_sdgs: List[str], idea: str, problem_statement: str, 
                           target_market: str, research_question: str) -> dict:
    """Generate market research insights"""
    gemini_model, tavily_client = setup_apis()
    
    try:
        # Tavily Search (if available)
        web_summary = "No web search available - Tavily API key not configured."
        source_urls = []
        
        if tavily_client:
            search_query = f"{research_question} {target_market} SDG {' '.join(selected_sdgs)} market research"
            
            with st.spinner("Searching the web for latest insights..."):
                try:
                    tavily_result = tavily_client.search(
                        query=search_query,
                        include_answer=True,
                        include_sources=True,
                        search_depth="advanced"
                    )
                    web_summary = tavily_result.get("answer", "No summary available.")
                    sources = tavily_result.get("sources", [])
                    source_urls = [src.get('url', '') for src in sources if src.get('url')]
                except Exception as e:
                    st.warning(f"Web search error: {e}")
                    web_summary = "No summary available due to search API error."
        
        # Gemini Analysis
        prompt = f"""
        You are a market research analyst. Based on the information provided, generate comprehensive market research insights.

        Web Research Summary: {web_summary}
        Web Sources: {', '.join(source_urls)}
        
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
        
        with st.spinner("Generating market research insights..."):
            response = gemini_model.generate_content(prompt)
            
            return {
                "success": True,
                "web_summary": web_summary,
                "market_research": response.text.strip(),
                "web_sources": source_urls,
                "sdgs": selected_sdgs,
                "idea": idea,
                "problem_statement": problem_statement,
                "target_market": target_market,
                "research_question": research_question
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def generate_presentation_questions(idea: str, problem_statement: str, market_research: str) -> List[str]:
    """Generate student presentation questions"""
    gemini_model, _ = setup_apis()
    
    prompt = f"""
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
        response = gemini_model.generate_content(prompt)
        questions = []
        
        for line in response.text.strip().split("\n"):
            if line.strip() and line[0].isdigit():
                question_text = line.split(".", 1)[1].strip()
                questions.append(question_text)
        
        return questions
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return []

def evaluate_market_fit(student_response: str) -> str:
    """Evaluate student's market fit response"""
    gemini_model, _ = setup_apis()
    
    full_prompt = MARKET_FIT_RUBRIC + "\n\nStudent Response:\n" + student_response.strip()
    
    try:
        response = gemini_model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error generating feedback: {e}"

# === Streamlit App ===
def main():
    st.set_page_config(
        page_title="Integrated SDG Student Platform",
        page_icon="🌍",
        layout="wide"
    )
    
    st.title("🌍 Integrated SDG Student Platform")
    st.markdown("Complete workflow: Select SDGs → Choose Ideas → Write Problem Statement → Evaluate → Market Research → Generate Questions → Market Fit Analysis")
    
    # Check API keys
    if not GEMINI_API_KEY:
        st.error("⚠️ Please set your GEMINI_API_KEY")
        st.stop()
    
    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 1
    
    # Progress indicator
    steps = ["Select SDGs", "Choose Idea", "Problem Statement", "Evaluation", "Market Research", "Questions", "Market Fit"]
    current_step = st.session_state.step
    
    # Progress bar
    progress = current_step / len(steps)
    st.progress(progress)
    st.write(f"**Step {current_step}/{len(steps)}: {steps[current_step-1]}**")
    
    # Step 1: Select SDGs
    if current_step == 1:
        st.header("🎯 Step 1: Select SDGs")
        st.write("Choose up to 3 Sustainable Development Goals for your project:")
        
        # Initialize SDG selection
        if 'selected_sdgs' not in st.session_state:
            st.session_state.selected_sdgs = []
        
        # SDG selection with checkboxes
        col1, col2 = st.columns(2)
        selected_sdgs = []
        
        for i, sdg in enumerate(SDG_LIST):
            col_idx = i % 2
            with [col1, col2][col_idx]:
                # Disable if already 3 selected and this one isn't selected
                disabled = len(st.session_state.selected_sdgs) >= 3 and sdg not in st.session_state.selected_sdgs
                
                is_checked = st.checkbox(
                    sdg,
                    value=sdg in st.session_state.selected_sdgs,
                    disabled=disabled,
                    key=f"sdg_{sdg}"
                )
                
                if is_checked:
                    selected_sdgs.append(sdg)
        
        # Update session state
        st.session_state.selected_sdgs = selected_sdgs
        
        # Show selection
        if selected_sdgs:
            st.success(f"✅ Selected {len(selected_sdgs)}/3 SDGs: {', '.join(selected_sdgs)}")
        else:
            st.info("Please select at least 1 SDG")
        
        # Next button
        if st.button("Next: Generate Ideas", type="primary", disabled=not selected_sdgs):
            st.session_state.step = 2
            st.rerun()
    
    # Step 2: Choose Idea
    elif current_step == 2:
        st.header("💡 Step 2: Choose Your Project Idea")
        st.write(f"**Selected SDGs:** {', '.join(st.session_state.selected_sdgs)}")
        
        # Generate ideas button
        if st.button("🚀 Generate Project Ideas", type="primary"):
            with st.spinner("Generating project ideas..."):
                ideas = generate_project_ideas(st.session_state.selected_sdgs)
                st.session_state.generated_ideas = ideas
        
        # Display generated ideas
        if 'generated_ideas' in st.session_state:
            st.subheader("Generated Project Ideas:")
            st.write(st.session_state.generated_ideas)
            
            # Idea selection
            st.subheader("Enter Your Chosen Idea:")
            chosen_idea = st.text_area(
                "Describe your chosen project idea:",
                height=150,
                placeholder="Enter your selected idea or modify one of the generated ideas..."
            )
            
            if chosen_idea and st.button("Next: Write Problem Statement", type="primary"):
                st.session_state.chosen_idea = chosen_idea
                st.session_state.step = 3
                st.rerun()
        
        # Back button
        if st.button("← Back to SDG Selection"):
            st.session_state.step = 1
            st.rerun()
    
    # Step 3: Problem Statement
    elif current_step == 3:
        st.header("📝 Step 3: Write Problem Statement")
        st.write(f"**Selected SDGs:** {', '.join(st.session_state.selected_sdgs)}")
        st.write(f"**Chosen Idea:** {st.session_state.chosen_idea}")
        
        # Show criteria
        with st.expander("📋 Problem Statement Criteria", expanded=True):
            st.write(PROBLEM_STATEMENT_CRITERIA)
        
        # Problem statement input
        problem_statement = st.text_area(
            "Write your problem statement:",
            height=200,
            placeholder="Write a comprehensive problem statement that addresses the criteria above..."
        )
        
        if problem_statement and st.button("Next: Evaluate Problem Statement", type="primary"):
            st.session_state.problem_statement = problem_statement
            st.session_state.step = 4
            st.rerun()
        
        # Back button
        if st.button("← Back to Choose Idea"):
            st.session_state.step = 2
            st.rerun()
    
    # Step 4: Evaluation
    elif current_step == 4:
        st.header("📊 Step 4: Problem Statement Evaluation")
        st.write(f"**Idea:** {st.session_state.chosen_idea}")
        st.write(f"**Problem Statement:** {st.session_state.problem_statement}")
        
        if st.button("🔍 Evaluate Problem Statement", type="primary"):
            with st.spinner("Evaluating your problem statement..."):
                evaluation = evaluate_problem_statement(
                    st.session_state.chosen_idea,
                    st.session_state.problem_statement
                )
                
                if evaluation['success']:
                    st.session_state.evaluation_result = evaluation['evaluation']
                    st.success("✅ Evaluation completed!")
                    st.write(evaluation['evaluation'])
                else:
                    st.error(f"Evaluation failed: {evaluation['error']}")
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Problem Statement"):
                st.session_state.step = 3
                st.rerun()
        
        with col2:
            if 'evaluation_result' in st.session_state:
                if st.button("Next: Market Research", type="primary"):
                    st.session_state.step = 5
                    st.rerun()
    
    # Step 5: Market Research
    elif current_step == 5:
        st.header("🔍 Step 5: Market Research")
        
        # Market research parameters
        col1, col2 = st.columns(2)
        
        with col1:
            target_market = st.text_input(
                "Target Market:",
                placeholder="e.g., Small farmers in rural areas"
            )
        
        with col2:
            research_question = st.text_input(
                "Research Question:",
                placeholder="What market insights do you want to discover?"
            )
        
        if st.button("🚀 Generate Market Research", type="primary"):
            if target_market and research_question:
                with st.spinner("Generating market research..."):
                    research_results = generate_market_research(
                        st.session_state.selected_sdgs,
                        st.session_state.chosen_idea,
                        st.session_state.problem_statement,
                        target_market,
                        research_question
                    )
                    
                    if research_results['success']:
                        st.session_state.market_research = research_results
                        st.success("✅ Market research completed!")
                        
                        # Display results
                        with st.expander("🌐 Web Research Summary", expanded=True):
                            st.write(research_results['web_summary'])
                        
                        with st.expander("📊 Market Research Analysis", expanded=True):
                            st.write(research_results['market_research'])
                        
                        if research_results['web_sources']:
                            with st.expander("🔗 Sources"):
                                for i, url in enumerate(research_results['web_sources'], 1):
                                    st.write(f"{i}. {url}")
                    else:
                        st.error(f"Market research failed: {research_results['error']}")
            else:
                st.error("Please fill in both target market and research question")
        
        # Navigation buttons
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
    
    # Step 6: Generate Questions
    elif current_step == 6:
        st.header("❓ Step 6: Presentation Questions")
        
        if st.button("🎯 Generate Presentation Questions", type="primary"):
            with st.spinner("Generating questions..."):
                questions = generate_presentation_questions(
                    st.session_state.chosen_idea,
                    st.session_state.problem_statement,
                    st.session_state.market_research['market_research']
                )
                
                if questions:
                    st.session_state.presentation_questions = questions
                    st.success("✅ Questions generated!")
                    
                    st.subheader("Questions for Your Presentation:")
                    for i, question in enumerate(questions, 1):
                        st.write(f"{i}. {question}")
        
        # Navigation buttons
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
    
    # Step 7: Market Fit Analysis
    elif current_step == 7:
        st.header("📈 Step 7: Market Fit Analysis")
        
        st.markdown("### ✍ Student Prompt:")
        st.info("*Write why you believe your idea is needed in the market and how your idea is unique. Use any data or current knowledge you have. Outline how you will enter the market.*")
        
        market_fit_response = st.text_area(
            "Your Market Fit Analysis:",
            height=300,
            placeholder="Write your analysis here..."
        )
        
        if st.button("📊 Get Feedback", type="primary"):
            if market_fit_response:
                with st.spinner("Analyzing your response..."):
                    feedback = evaluate_market_fit(market_fit_response)
                    st.success("✅ Feedback generated!")
                    st.markdown("### 📋 Feedback:")
                    st.write(feedback)
            else:
                st.error("Please write your market fit analysis")
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Questions"):
                st.session_state.step = 6
                st.rerun()
        
        with col2:
            if st.button("🎉 Complete Project"):
                st.balloons()
                st.success("🎉 Congratulations! You've completed all steps of the SDG project workflow!")
                
                # Show project summary
                with st.expander("📋 Project Summary", expanded=True):
                    st.write(f"**SDGs:** {', '.join(st.session_state.selected_sdgs)}")
                    st.write(f"**Idea:** {st.session_state.chosen_idea}")
                    st.write(f"**Problem Statement:** {st.session_state.problem_statement}")
                    if 'market_research' in st.session_state:
                        st.write(f"**Target Market:** {st.session_state.market_research['target_market']}")
                    if 'presentation_questions' in st.session_state:
                        st.write("**Presentation Questions:**")
                        for i, q in enumerate(st.session_state.presentation_questions, 1):
                            st.write(f"{i}. {q}")
    
    # Sidebar - Progress and Reset
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
            # Clear all session state
            keys_to_clear = [
                'step', 'selected_sdgs', 'generated_ideas', 'chosen_idea', 
                'problem_statement', 'evaluation_result', 'market_research', 
                'presentation_questions'
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.step = 1
            st.rerun()
        
        # API Status
        st.header("🔌 API Status")
        st.write("✅ Gemini AI: Connected")
        if TAVILY_API_KEY:
            st.write("✅ Tavily Search: Connected")
        else:
            st.write("⚠️ Tavily Search: Not configured")

if __name__ == "__main__":
    main()