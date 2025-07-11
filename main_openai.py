# Save this file as app.py
import streamlit as st
import os
import re
from typing import List
from dotenv import load_dotenv
from tavily import TavilyClient
from openai import OpenAI

# --- Import custom modules ---
try:
    from utlis_openai import classify_problem_statement
except ImportError:
    st.error("Error importing 'classify_problem_statement' from utlis_openai.py. Please ensure the file exists in the project directory.")
    st.stop()
    
try:
    from image_generator import add_prototype_generation_step
except ImportError:
    st.error("Error importing 'add_prototype_generation_step' from image_generator.py. Please ensure the file exists in the project directory.")
    st.stop()

# === Load API keys from .env file ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# === Configure API Clients (Cached) ===
@st.cache_resource
def setup_apis():
    """Setup and cache API clients"""
    if not OPENAI_API_KEY:
        st.error("OpenAI API key is not set. Please add it to your .env file.")
        st.stop()
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    tavily_client = None
    if TAVILY_API_KEY:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    else:
        st.warning("Tavily API key not found. Web search will be disabled.")
        
    return openai_client, tavily_client

# === Constants and Prompts ===
SDG_LIST = [
    "No Poverty", "Zero Hunger", "Good Health and Well-being", "Quality Education",
    "Gender Equality", "Clean Water and Sanitation", "Affordable and Clean Energy",
    "Decent Work and Economic Growth", "Industry, Innovation and Infrastructure",
    "Reduced Inequalities", "Sustainable Cities and Communities",
    "Responsible Consumption and Production", "Climate Action", "Life Below Water",
    "Life on Land", "Peace, Justice and Strong Institutions", "Partnerships for the Goals"
]

PROBLEM_STATEMENT_CRITERIA = """
**EFFECTIVE PROBLEM STATEMENT ASSESSMENT CRITERIA:**
1. **Contains Data**: Includes relevant quantitative or qualitative data.
2. **References Included**: Cites credible sources or references.
3. **Location/Area Clear**: Specifies the geographical location or area.
4. **Target Audience Clearly Stated**: Defines the specific group affected.
5. **Impact Described**: Explains the negative consequences if unaddressed.
"""

MARKET_FIT_RUBRIC = """
You are a supportive business mentor evaluating a young entrepreneur's market analysis. A student (aged 14-15) has written a response about their business idea. Please evaluate their work using this 10-point rubric, providing encouraging yet constructive feedback.
Evaluation Criteria (1-10 scale, where 1 = needs significant improvement, 10 = excellent):

Target Audience Clarity - Do they clearly identify who their customers are and what those customers need?
Problem-Solution Connection - Do they explain how their idea solves a real problem for their target customers?
Market Research Evidence - Do they include any supporting data, research, or validation (surveys, interviews, observations)?
Unique Value Proposition - Do they explain what makes their idea different or better than existing solutions?
Market Entry Strategy - Do they outline realistic first steps for launching their idea (MVP, pilot testing, initial customers)?
Communication Quality - Is their writing clear with proper grammar, spelling, and punctuation?
Business Understanding - Do they demonstrate solid comprehension of basic business concepts?
Focus and Conciseness - Do they stay on topic and communicate efficiently without unnecessary details?
Relevance and Consistency - Does all content directly relate to supporting their business idea?
Organization and Clarity - Is their response well-structured and easy to follow?

Feedback Guidelines:
Use encouraging, constructive language appropriate for teenagers. Provide specific examples from their response. Offer concrete suggestions for improvement. Acknowledge their strengths before addressing weaknesses. If the response is off-topic or inappropriate, gently redirect them back to the business concept. Format feedback as numbered points (1-10) with scores and detailed comments.
Remember: These students are learning entrepreneurship basics. Focus on building their confidence while helping them improve their business thinking and communication skills.
you are speaking to the student , so never use "the student's response".
"""

# === Helper Functions ===
def generate_project_ideas(selected_sdgs: List[str]) -> str:
    openai_client, _ = setup_apis()
    system_prompt = "You are an educational assistant helping students brainstorm project ideas."
    
    # --- UPDATED PROMPT TO GET FULL CONTENT ---
    user_prompt = f"""
    Generate 5 student-friendly, realistic project ideas based on the following Sustainable Development Goals: {', '.join(selected_sdgs)}.
    Each idea should be:
    - Feasible for students to implement, ethical, and socially responsible.
    - Involve either technology or social innovation.
    - Clearly address one or more of the selected SDGs.

    Format each idea on a new line as a numbered list, using '::' as a separator, like this:
    1. Idea Title :: A brief, one-sentence description of the idea.

    Strictly avoid any harmful or dangerous content.
    """
    try:
        response = openai_client.chat.completions.create(model="gpt-4.1", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.7, max_tokens=1000)
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating ideas: {str(e)}"); return "Error generating ideas."

def evaluate_problem_statement_wrapper(idea, problem_statement):
    try:
        result = classify_problem_statement(idea, problem_statement)
        return {"success": True, "evaluation": result.get('data', result)} if isinstance(result, dict) and result.get('success') else {"success": False, "error": result.get('error', 'Classification failed.')}
    except Exception as e:
        return {"success": False, "error": str(e)}

def generate_market_research(selected_sdgs, idea, problem_statement, target_market, research_question):
    openai_client, tavily_client = setup_apis()
    web_summary, source_urls = "Web search was not performed as Tavily API key is not configured.", []
    if tavily_client:
        search_query = f"{research_question} for {target_market} related to SDGs: {', '.join(selected_sdgs)}"
        try:
            with st.spinner("Searching the web..."):
                tavily_result = tavily_client.search(query=search_query, include_answer=True, include_sources=True, search_depth="basic")
                web_summary = tavily_result.get("answer", "No summary could be generated.")
                source_urls = [src.get('url', '') for src in tavily_result.get("sources", []) if src.get('url')]
        except Exception as e:
            st.warning(f"Web search failed: {e}."); web_summary = "Web search unavailable."
    system_prompt = "You are a market research analyst helping a student."
    user_prompt = f"Web Research Summary:\n{web_summary}\n\nProject Details:\n- SDGs: {', '.join(selected_sdgs)}\n- Idea: {idea}\n- Problem: {problem_statement}\n- Target Market: {target_market}\n- Question: {research_question}\n\nProvide a structured report covering: Market Size, Customer Analysis, Competition, Entry Strategy, and Challenges."
    with st.spinner("Analyzing market data..."):
        try:
            response = openai_client.chat.completions.create(model="gpt-4.1", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.7, max_tokens=1500)
            return {"success": True, "web_summary": web_summary, "market_research": response.choices[0].message.content.strip(), "web_sources": source_urls, "target_market": target_market, "research_question": research_question}
        except Exception as e:
            return {"success": False, "error": str(e)}

def generate_presentation_questions(idea, problem_statement, market_research):
    openai_client, _ = setup_apis()
    system_prompt = "You are a presentation coach."
    user_prompt = f"Generate 5 engaging, open-ended questions a student can ask their audience.\n\nProject Idea: {idea}\nMarket Research: {market_research[:500]}...\n\nReturn ONLY questions, numbered 1-5."
    try:
        response = openai_client.chat.completions.create(model="gpt-4.1", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.7, max_tokens=500)
        content = response.choices[0].message.content.strip()
        return [line.split(".", 1)[-1].strip() for line in content.split("\n") if line.strip() and line[0].isdigit()][:5]
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}"); return []

def evaluate_market_fit(student_response):
    openai_client, _ = setup_apis()
    try:
        response = openai_client.chat.completions.create(model="gpt-4.1", messages=[{"role": "system", "content": MARKET_FIT_RUBRIC}, {"role": "user", "content": f"Student Response:\n\n{student_response.strip()}"}], temperature=0.7, max_tokens=1500)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating feedback: {e}"

# === Main Streamlit App ===
def main():
    st.set_page_config(page_title="SDG Student Platform", page_icon="🌍", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""<style>.main-header{background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);padding:1rem;border-radius:10px;margin-bottom:2rem}.main-header h1{color:#fff;text-align:center;margin:0}.step-card{background:#f8f9fa;padding:1.5rem;border-radius:10px;border-left:4px solid #667eea;margin:1rem 0}.success-box{background:#d4edda;border:1px solid #c3e6cb;color:#155724;padding:1rem;border-radius:5px;margin:1rem 0}.info-box{background:#d1ecf1;border:1px solid #bee5eb;color:#0c5460;padding:1rem;border-radius:5px;margin:1rem 0}</style>""", unsafe_allow_html=True)
    st.markdown("""<div class="main-header"><h1>🌍 SDG Student Platform</h1><p style="text-align:center;color:#fff;margin:0">Complete workflow: From SDG selection to prototype visualization</p></div>""", unsafe_allow_html=True)
    
    if not OPENAI_API_KEY: st.error("⚠️ Please set your OPENAI_API_KEY in the .env file"); st.stop()
    if 'step' not in st.session_state: st.session_state.step = 1
    
    steps = ["Select SDGs", "Choose Idea", "Problem Statement", "Evaluation", "Market Research", "Questions", "Market Fit", "Prototype Visualization"]
    current_step = st.session_state.step
    st.progress(current_step / len(steps))
    st.markdown(f'<div style="text-align: center; padding: 1rem;"><h3>Step {current_step}/{len(steps)}: {steps[current_step-1]}</h3></div>', unsafe_allow_html=True)

    if current_step == 1:
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.header("🎯 Select Sustainable Development Goals")
        st.write("Choose up to 3 SDGs that your project will address:")
        if 'selected_sdgs' not in st.session_state: st.session_state.selected_sdgs = []
        cols = st.columns(3)
        current_selection = []
        for i, sdg in enumerate(SDG_LIST):
            with cols[i % 3]:
                if st.checkbox(sdg, value=(sdg in st.session_state.selected_sdgs), key=f"sdg_{i}"):
                    current_selection.append(sdg)
        st.session_state.selected_sdgs = current_selection[:3]
        if st.session_state.selected_sdgs: st.markdown(f"<div class='success-box'>✅ Selected {len(st.session_state.selected_sdgs)}/3 SDGs: {', '.join(st.session_state.selected_sdgs)}</div>", unsafe_allow_html=True)
        else: st.markdown("<div class='info-box'>Please select at least 1 SDG to continue.</div>", unsafe_allow_html=True)
        if st.button("Next: Generate Ideas", type="primary", disabled=not st.session_state.selected_sdgs):
            st.session_state.step = 2; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # --- UPDATED STEP 2 LOGIC TO DISPLAY FULL CONTENT ---
    elif current_step == 2:
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.header("💡 Choose Your Project Idea")
        st.write(f"**Selected SDGs:** {', '.join(st.session_state.selected_sdgs)}")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🚀 Generate New Project Ideas", type="primary"):
                with st.spinner("Generating creative project ideas..."):
                    st.session_state.generated_ideas = generate_project_ideas(st.session_state.selected_sdgs)
        with col2:
            if st.button("← Back to SDG Selection"): st.session_state.step = 1; st.rerun()
        
        if 'generated_ideas' in st.session_state:
            st.subheader("✅ Your Generated Ideas")
            
            # Parse ideas into a dictionary of {title: description}
            ideas_dict = {}
            for line in st.session_state.generated_ideas.split('\n'):
                if '::' in line:
                    clean_line = re.sub(r'^\s*\d+\.\s*', '', line).strip()
                    parts = clean_line.split('::', 1)
                    if len(parts) == 2:
                        ideas_dict[parts[0].strip()] = parts[1].strip()

            if not ideas_dict:
                st.warning("Could not parse the generated ideas. Please try generating again.")
            else:
                radio_options = list(ideas_dict.keys()) + ["Write my own idea"]
                chosen_option = st.radio("Select one of the generated ideas or choose to write your own:", options=radio_options, key="idea_choice")
                
                # Display the full description of the selected idea
                if chosen_option in ideas_dict:
                    st.markdown(f"<div class='info-box'><b>Description:</b> {ideas_dict[chosen_option]}</div>", unsafe_allow_html=True)
                
                final_idea = ""
                if chosen_option == "Write my own idea":
                    final_idea = st.text_area("Describe your custom project idea here:", height=100, key="custom_idea_input")
                else:
                    final_idea = chosen_option
                
                if st.button("Next: Write Problem Statement", type="primary", disabled=not final_idea):
                    st.session_state.chosen_idea = final_idea; st.session_state.step = 3; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    elif current_step == 3:
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.header("📝 Write Your Problem Statement")
        st.markdown(f"**Idea:** {st.session_state.chosen_idea}")
        with st.expander("📋 Problem Statement Criteria", expanded=True): st.markdown(PROBLEM_STATEMENT_CRITERIA)
        problem_statement = st.text_area("Write your comprehensive problem statement:", height=200, key="problem_statement_input")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Back to Choose Idea"): st.session_state.step = 2; st.rerun()
        with col2:
            if st.button("Next: Evaluate Problem Statement", type="primary", disabled=not problem_statement):
                st.session_state.problem_statement = problem_statement; st.session_state.step = 4; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    elif current_step >= 4:
        # Steps 4-8 remain unchanged and are omitted here for brevity
        # but would be included in the actual running file.
        st.markdown('<div class="step-card">...</div>', unsafe_allow_html=True)


    # ... The rest of the main function for steps 4-8 and the sidebar remains the same ...
    # (Included here for completeness)
    if current_step == 4:
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.header("📊 Problem Statement Evaluation")
        if 'evaluation_result' not in st.session_state:
            with st.spinner("Evaluating your problem statement..."):
                st.session_state.evaluation_result = evaluate_problem_statement_wrapper(st.session_state.chosen_idea, st.session_state.problem_statement)
        result = st.session_state.evaluation_result
        if result['success']: st.success("✅ Evaluation completed!"); st.json(result['evaluation'])
        else: st.error(f"❌ Evaluation failed: {result['error']}")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Back to Problem Statement"):
                st.session_state.step = 3; del st.session_state.evaluation_result; st.rerun()
        with col2:
            if result['success']:
                if st.button("Next: Market Research", type="primary"): st.session_state.step = 5; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    elif current_step == 5:
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.header("🔍 Market Research")
        col1, col2 = st.columns(2)
        target_market = col1.text_input("Target Market:", placeholder="e.g., Small farmers in rural Kenya", key="target_market_input")
        research_question = col2.text_input("Research Question:", placeholder="e.g., Current challenges for market access", key="research_question_input")
        if st.button("🚀 Generate Market Research", type="primary", disabled=not (target_market and research_question)):
            st.session_state.market_research = generate_market_research(st.session_state.selected_sdgs, st.session_state.chosen_idea, st.session_state.problem_statement, target_market, research_question)
        if 'market_research' in st.session_state:
            res = st.session_state.market_research
            if res['success']:
                st.success("✅ Market research completed!")
                with st.expander("🌐 Web Research Summary", expanded=True): st.write(res['web_summary'])
                with st.expander("📊 Market Analysis", expanded=True): st.markdown(res['market_research'])
                if res.get('web_sources'):
                    with st.expander("🔗 Sources"):
                        for i, url in enumerate(res['web_sources'], 1): st.write(f"{i}. {url}")
            else: st.error(f"❌ Market research failed: {res['error']}")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Back to Evaluation"): st.session_state.step = 4; st.rerun()
        with col2:
            if 'market_research' in st.session_state and st.session_state.market_research['success']:
                if st.button("Next: Generate Questions", type="primary"): st.session_state.step = 6; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    elif current_step == 6:
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.header("❓ Presentation Questions")
        if st.button("🎯 Generate Presentation Questions", type="primary"):
            with st.spinner("Generating engaging questions..."):
                st.session_state.presentation_questions = generate_presentation_questions(st.session_state.chosen_idea, st.session_state.problem_statement, st.session_state.market_research['market_research'])
        if 'presentation_questions' in st.session_state and st.session_state.presentation_questions:
            st.success("✅ Questions generated!")
            for i, question in enumerate(st.session_state.presentation_questions, 1): st.write(f"**{i}.** {question}")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Back to Market Research"): st.session_state.step = 5; st.rerun()
        with col2:
            if 'presentation_questions' in st.session_state and st.session_state.presentation_questions:
                if st.button("Next: Market Fit Analysis", type="primary"): st.session_state.step = 7; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    elif current_step == 7:
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.header("📈 Market Fit Analysis")
        market_fit_response = st.text_area("Your Market Fit Analysis:", height=300, key="market_fit_input", help="Explain why your idea is needed, what makes it unique, and your market entry strategy.")
        if st.button("📊 Get Feedback", type="primary", disabled=not market_fit_response):
            with st.spinner("Analyzing your response..."):
                st.session_state.market_fit_feedback = evaluate_market_fit(market_fit_response)
        if 'market_fit_feedback' in st.session_state:
            st.success("✅ Feedback generated!")
            st.markdown(st.session_state.market_fit_feedback)
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Back to Questions"): st.session_state.step = 6; st.rerun()
        with col2:
            if 'market_fit_feedback' in st.session_state:
                if st.button("Next: Generate Prototypes", type="primary"): st.session_state.step = 8; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    elif current_step == 8:
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.header("🎨 Prototype Visualization")
        target_market = st.session_state.get('market_research', {}).get('target_market', 'general users')
        generation_successful = add_prototype_generation_step(st.session_state.chosen_idea, st.session_state.problem_statement, target_market, OPENAI_API_KEY)
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Back to Market Fit"):
                st.session_state.step = 7
                if 'prototype_generation_result' in st.session_state: del st.session_state.prototype_generation_result
                st.rerun()
        with col2:
            if generation_successful:
                if st.button("🎉 Complete Project", type="primary"):
                    st.balloons()
                    st.success("🎉 Congratulations! You've completed the full SDG project workflow!")
                    with st.expander("📋 Final Project Summary", expanded=True):
                        st.write(f"**SDGs:** {', '.join(st.session_state.selected_sdgs)}")
                        st.write(f"**Idea:** {st.session_state.chosen_idea}")
                        st.write(f"**Problem Statement:** {st.session_state.problem_statement}")
                        if 'market_research' in st.session_state: st.write(f"**Target Market:** {st.session_state.market_research['target_market']}")
                        if 'presentation_questions' in st.session_state:
                            st.write("**Presentation Questions:**"); [st.write(f"- {q}") for q in st.session_state.presentation_questions]
                        if 'prototype_generation_result' in st.session_state: st.write(f"**Prototypes:** {len(st.session_state.prototype_generation_result.get('images',[]))} visualization(s) created.")
                    if st.button("🔄 Start New Project"):
                        for key in list(st.session_state.keys()): del st.session_state[key]
                        st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with st.sidebar:
        st.header("🎯 Progress Tracker")
        for i, step_name in enumerate(steps, 1):
            if i < current_step: st.write(f"✅ {i}. {step_name}")
            elif i == current_step: st.write(f"📍 **{i}. {step_name}**")
            else: st.write(f"⏳ {i}. {step_name}")
        st.markdown("---")
        if 'selected_sdgs' in st.session_state and st.session_state.selected_sdgs:
            st.subheader("📊 Project Summary")
            st.write(f"**SDGs:** {', '.join(st.session_state.selected_sdgs)}")
            if 'chosen_idea' in st.session_state: st.write(f"**Idea:** {st.session_state.chosen_idea[:100]}...")
            if 'problem_statement' in st.session_state: st.write(f"**Problem:** {st.session_state.problem_statement[:100]}...")
            if 'market_research' in st.session_state: st.write(f"**Target:** {st.session_state.market_research.get('target_market', 'N/A')}")
        st.markdown("---")
        if st.button("🔄 Reset All Progress"):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()