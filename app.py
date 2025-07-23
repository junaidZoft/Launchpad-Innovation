import streamlit as st
import json
import logging
import os
import sys
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from generate_idea import generate_sdg_ideas_json
except ImportError:
    logging.error("Failed to import generate_sdg_ideas_json from generate_idea.py. Ensure the file exists and is correctly named.")
    generate_sdg_ideas_json = None

try:
    from evaluate_problem_statement import OptimizedProblemStatementClassifier
except ImportError:
    st.error("❌ Could not import from evaluate_problem_statement.py. Please ensure the file exists and is in the same directory.")
    OptimizedProblemStatementClassifier = None
    
try:
    from market_research import generate_market_insights, verify_env_setup
except ImportError:
    logging.error("Failed to import from market_research.py. Ensure the file exists and is correctly named.")

try:
    from generate_questions import generate_survey_questions
except ImportError:
    logging.error("Failed to import generate_survey_questions. Ensure the file exists and is correctly named.")

try:
    from generate_marketfit_feedback import get_market_fit_feedback
except ImportError:
    logging.error("Failed to import get_market_fit_feedback. Ensure the file exists and is correctly named.")


# Page configuration
st.set_page_config(
    page_title="SDG Innovation Hub",
    page_icon="🌍",
    layout="wide"
)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 1
if 'selected_sdgs' not in st.session_state:
    st.session_state.selected_sdgs = []
if 'generated_ideas' not in st.session_state:
    st.session_state.generated_ideas = []
if 'selected_idea' not in st.session_state:
    st.session_state.selected_idea = ""
if 'problem_statement' not in st.session_state:
    st.session_state.problem_statement = ""
if 'problem_statement_classification_result' not in st.session_state:
    st.session_state.problem_statement_classification_result = None
if 'target_market' not in st.session_state:
    st.session_state.target_market = ""
if 'market_summary' not in st.session_state:
    st.session_state.market_summary = ""
if 'research_completed' not in st.session_state:
    st.session_state.research_completed = False
if 'research_questions' not in st.session_state:
    st.session_state.research_questions = []
if 'survey_questions' not in st.session_state:
    st.session_state.survey_questions = []
if 'marketfit' not in st.session_state:
    st.session_state.marketfit = ""
if 'market_fit_feedback' not in st.session_state:
    st.session_state.market_fit_feedback = ""

# SDG List
sdgs = [
    "1. No Poverty",
    "2. Zero Hunger", 
    "3. Good Health and Well-being",
    "4. Quality Education",
    "5. Gender Equality",
    "6. Clean Water and Sanitation",
    "7. Affordable and Clean Energy",
    "8. Decent Work and Economic Growth",
    "9. Industry, Innovation and Infrastructure",
    "10. Reduced Inequalities",
    "11. Sustainable Cities and Communities",
    "12. Responsible Consumption and Production",
    "13. Climate Action",
    "14. Life Below Water",
    "15. Life on Land",
    "16. Peace, Justice and Strong Institutions",
    "17. Partnerships for the Goals"
]

# Navigation
def navigate_to_page(page_num):
    st.session_state.page = page_num

# Sidebar navigation
st.sidebar.title("Navigation")
pages = [
    "1. SDG Selection & Ideas",
    "2. Problem Statement", 
    "3. Market Research",
    "4. Survey Questions",
    "5. Market Fit Analysis",
    "6. Prototype generation"
]

for i, page_name in enumerate(pages, 1):
    if st.sidebar.button(page_name, key=f"nav_{i}"):
        navigate_to_page(i)

# Main content based on current page
if st.session_state.page == 1:
    # PAGE 1: SDG Selection and Idea Generation
    st.title("🌍 SDG Innovation Hub")
    st.header("Step 1: Select SDGs and Generate Ideas")
    
    st.subheader("Choose exactly 2 Sustainable Development Goals")
    st.write("Select up to 2 SDGs from the list below:")
    
    # Create checkboxes for all 17 SDGs
    selected_sdgs = []
    
    # Display SDGs in 3 columns for better layout
    col1, col2, col3 = st.columns(3)
    
    for i, sdg in enumerate(sdgs):
        if i % 3 == 0:
            with col1:
                if st.checkbox(sdg, key=f"sdg_check_{i}"):
                    selected_sdgs.append(sdg)
        elif i % 3 == 1:
            with col2:
                if st.checkbox(sdg, key=f"sdg_check_{i}"):
                    selected_sdgs.append(sdg)
        else:
            with col3:
                if st.checkbox(sdg, key=f"sdg_check_{i}"):
                    selected_sdgs.append(sdg)
    
    # Check if user selected more than 2 SDGs
    if len(selected_sdgs) > 2:
        st.error("⚠️ Please select exactly 2 SDGs. You have selected too many.")
        st.session_state.selected_sdgs = []
    elif len(selected_sdgs) == 2:
        st.session_state.selected_sdgs = selected_sdgs
        st.success(f"✅ Selected SDGs: {selected_sdgs[0]} and {selected_sdgs[1]}")
        
        if st.button("🚀 Generate Ideas", type="primary"):
            if generate_sdg_ideas_json is not None:
                try:
                    result = generate_sdg_ideas_json(selected_sdgs)
                    if result:
                        st.session_state.generated_ideas = []
                        for idea in result:
                            title = idea.get("title", "Untitled")
                            description = idea.get("description", "No description provided.")
                            sdgs_addressed = ", ".join(idea.get("sdgs_addressed", []))
                            st.session_state.generated_ideas.append(f"**{title}**: \n{description}")
                        st.success("Ideas generated successfully!")
                    else:
                        st.error("❌ Failed to generate ideas. No results returned.")
                except Exception as e:
                    st.error(f"❌ Error generating ideas: {str(e)}")
                    logging.error(f"Error in generate_sdg_ideas_json: {str(e)}")
            else:
                st.error("❌ Idea generation module not available. Please check imports.")
            st.rerun()
            
    elif len(selected_sdgs) == 1:
        st.info("📋 Please select one more SDG to proceed.")
        st.session_state.selected_sdgs = []
    else:
        st.info("📋 Please select 2 SDGs to get started.")
        st.session_state.selected_sdgs = []
    
    if st.session_state.generated_ideas:
        st.subheader("Generated Ideas")
        selected_idea = st.radio(
            "Select one idea to proceed:",
            st.session_state.generated_ideas,
            key="idea_selection"
        )
        
        st.session_state.selected_idea = selected_idea
        
        if st.button("Proceed to Problem Statement →", type="primary"):
            navigate_to_page(2)
            st.rerun()

elif st.session_state.page == 2:
    # PAGE 2: Problem Statement
    st.title("📝 Problem Statement")
    st.header("Step 2: Define the Problem")
    
    if st.session_state.selected_idea:
        st.info(f"**Selected Idea:** {st.session_state.selected_idea}")
    
    st.subheader("Write Problem Statement")
    problem_statement = st.text_area(
        "Describe the problem your idea addresses:",
        value=st.session_state.problem_statement,
        height=150,
        placeholder="Enter a detailed problem statement here...",
        key="problem_statement_input"
    )
    
    st.session_state.problem_statement = problem_statement
    
    if problem_statement:
        if st.button("🔍 Evaluate Problem Statement", type="primary"):
            if OptimizedProblemStatementClassifier is not None:
                try:
                    # Initialize the classifier
                    classifier = OptimizedProblemStatementClassifier(
                        api_key=None,  # Will use GROQ_API_KEY from environment
                        model_name="llama3-70b-8192",  # or any other model
                        max_retries=2,
                        timeout=15.0,
                        enable_caching=True
                    )
                    
                    # Call the classification method
                    result = classifier.classify_problem_statement(
                        idea_text=st.session_state.selected_idea,
                        problem_statement_text=st.session_state.problem_statement
                    )
                    
                    # Handle the result
                    if result and hasattr(result, 'success') and result.success:
                        # Classification was successful
                        st.success("✅ Problem statement evaluated successfully!")
                        
                        # Display the results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Quality Assessment (X-Axis)")
                            st.write(f"**Category:** {getattr(result, 'x_axis_category', 'N/A')}")
                        
                        with col2:
                            st.subheader("Content Assessment (Y-Axis)")
                            st.write(f"**Category:** {getattr(result, 'y_axis_category', 'N/A')}")
                        
                        # Store result in session state for later use
                        st.session_state.problem_statement_classification_result = result
                        
                        # Optional: Show raw response in expandable section
                        if hasattr(result, 'raw_response') and result.raw_response:
                            with st.expander("📄 View Raw Response"):
                                st.text(result.raw_response)
                            
                    else:
                        # Classification failed
                        error_msg = getattr(result, 'error', 'Unknown error') if result else 'No result returned'
                        st.error(f"❌ Evaluation failed: {error_msg}")
                        processing_time = getattr(result, 'processing_time', None) if result else None
                        if processing_time:
                            st.info(f"Processing time: {processing_time:.3f}s")
                        
                except Exception as e:
                    st.error(f"❌ Classifier error: {str(e)}")
                    logging.error(f"Classifier error: {str(e)}")
            else:
                st.error("❌ Problem statement classifier not available. Please check imports.")
    
    # Show proceed button if classification result exists
    if st.session_state.problem_statement_classification_result:
        if st.button("Proceed to Market Research →", type="primary"):
            navigate_to_page(3)
            st.rerun()
elif st.session_state.page == 3:
    # PAGE 3: Market Research
    st.title("🎯 Market Research")
    st.header("Step 3: Define Target Market and Research")
    
    if st.session_state.selected_idea:
        st.info(f"**Selected Idea:** {st.session_state.selected_idea}")
    
    st.subheader("Target Market")
    target_market = st.text_area(
        "Define your target market:",
        value=st.session_state.target_market,
        height=100,
        placeholder="Describe your target market demographics, geography, psychographics...",
        key="target_market_input"
    )
    
    st.subheader("What do you already know")
    market_summary = st.text_area(
        "What do you already know?",
        value=st.session_state.market_summary,
        height=68,
        placeholder="Describe what you already know about the market, competitors, etc.",
        key="market_summary_input"
    )
    
    # Update session state
    st.session_state.target_market = target_market
    st.session_state.market_summary = market_summary
    
    # Check if basic fields are filled
    basic_fields_filled = all([
        target_market.strip(), 
        market_summary.strip(), 
        st.session_state.problem_statement, 
        st.session_state.selected_idea
    ])
    
    # Check character limits
    char_limit_ok = len(target_market) <= 400 and len(market_summary) <= 400
    
    if not char_limit_ok:
        st.error("❌ Target market and market summary must be under 400 characters each.")
    
    if not basic_fields_filled:
        st.warning("Please fill in all fields before proceeding.")
    
    # Market Research Generation
    if st.button("🔬 Generate Market Research", type="primary"):
        if basic_fields_filled and char_limit_ok:
            with st.spinner("Conducting market research..."):
                results = generate_market_insights(
                    target_market=target_market,
                    market_summary=market_summary,
                    idea=st.session_state.selected_idea,
                    problem_statement=st.session_state.problem_statement
                )
            
            if "error" in results:
                st.error(f"Error: {results['error']}")
                if "instructions" in results:
                    st.info(results['instructions'])
            else:
                # Show consistency check
                st.subheader("📋 Consistency Check")
                if "✅" in results.get('consistency_check', ''):
                    st.success(results['consistency_check'])
                else:
                    st.warning(results['consistency_check'])
                
                # Show market analysis
                st.subheader("📊 Market Analysis")
                st.markdown(results.get('market_analysis', 'No analysis available'))
                
                # Show sources
                if results.get('web_sources'):
                    st.subheader("🔗 Sources")
                    for i, source in enumerate(results['web_sources'], 1):
                        with st.expander(f"Source {i}: {source['title']}"):
                            st.write(source['content'])
                            st.write(f"**URL:** {source['url']}")
                
                st.session_state.market_research_results = results
                st.success("✅ Market research completed!")
    
    # Evaluation section - only show if market research exists
    if hasattr(st.session_state, 'market_research_results') and st.session_state.market_research_results:
        st.divider()
        
        if st.button("📋 Evaluate Market Research", type="secondary"):
            with st.spinner("Evaluating market research..."):
                try:
                    from evaluate_market_research import categorize_market_research
                    
                    # Combine market data for evaluation
                    data = market_summary + " " + target_market
                    result = categorize_market_research(data)
                    
                    # Store evaluation results
                    st.session_state.market_research_evaluated = True
                    st.session_state.categorization_result = result
                    
                    logging.info("Market research evaluation completed.")
                    logging.info(f"Evaluation result: {result}")
                    
                    st.success("✅ Market research evaluation completed!")
                    
                    # Show detailed results in expandable section
                    with st.expander("🔍 View Detailed Results"):
                        st.json(result)
                        
                except ImportError as e:
                    st.error(f"Error importing evaluation module: {str(e)}")
                    st.info("Make sure 'evaluate_market_research.py' file exists and is properly configured.")
                except Exception as e:
                    st.error(f"Error evaluating market research: {str(e)}")
                    st.session_state.market_research_evaluated = False
    

    # Navigation section - always show at bottom
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("← Back to Problem Statement", type="secondary"):
            navigate_to_page(2)
            st.rerun()
    
    with col2:
        # Check if user can proceed to next stage
        can_proceed = basic_fields_filled and char_limit_ok
        
        # Optional: Also require evaluation to be completed
        # can_proceed = can_proceed and getattr(st.session_state, 'market_research_evaluated', False)
        
        if can_proceed:
            if st.button("Proceed to Survey Questions →", type="primary"):
                navigate_to_page(4)
                st.rerun()
        else:
            help_text = "Complete all fields first"
            # if not getattr(st.session_state, 'market_research_evaluated', False):
            #     help_text += " and evaluate market research"
            st.button("Proceed to Survey Questions →", disabled=True, help=help_text)
            

elif st.session_state.page == 4:
    # PAGE 4: Survey Questions
    st.title("❓ Survey Questions")
    st.header("Step 4: Generate Yes/No Questions")
    
    # Check required fields
    required_fields = [
        st.session_state.selected_idea,
        st.session_state.problem_statement,
        st.session_state.selected_sdgs
    ]
    
    if all(required_fields):
        st.info(f"**Selected Idea:** {st.session_state.selected_idea}")
        st.info(f"**Problem Statement:** {st.session_state.problem_statement}")
        st.info(f"**Selected SDGs:** {', '.join(st.session_state.selected_sdgs)}")
        
        # Generate Questions Button
        if st.button("📋 Generate Survey Questions", type="primary"):
            with st.spinner("Generating survey questions..."):
                try:
                    result = generate_survey_questions(
                        idea=st.session_state.selected_idea,
                        problem_statement=st.session_state.problem_statement,
                        sdgs=st.session_state.selected_sdgs
                    )
                    
                    # Store the generated questions
                    st.session_state.survey_questions = result
                    
                    # Initialize responses dictionary
                    if 'survey_responses' not in st.session_state:
                        st.session_state.survey_responses = {}
                    
                    st.success("✅ Survey questions generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating questions: {str(e)}")
        
        # Display questions if they exist
        if hasattr(st.session_state, 'survey_questions') and st.session_state.survey_questions:
            st.subheader("📝 Generated Survey Questions")
            
            # Initialize responses if not exists
            if 'survey_responses' not in st.session_state:
                st.session_state.survey_responses = {}
            
            # Display each question with radio buttons
            for i, question in enumerate(st.session_state.survey_questions, 1):
                st.write(f"**{i}.** {question}")
                
                # Add radio buttons for responses
                response = st.radio(
                    f"Response for Question {i}", 
                    ["Yes", "No"], 
                    key=f"response_{i}",
                    index=None,  # No default selection
                    help="Select your response to this question"
                )
                
                # Store individual responses
                if response:
                    st.session_state.survey_responses[f"question_{i}"] = response
            
            # Show response summary if any responses are given
            if st.session_state.survey_responses:
                with st.expander("📊 Response Summary", expanded=False):
                    for key, value in st.session_state.survey_responses.items():
                        question_num = key.split('_')[1]
                        st.write(f"**Question {question_num}:** {value}")
                    
                    # Show completion status
                    completed_responses = len(st.session_state.survey_responses)
                    total_questions = len(st.session_state.survey_questions)
                    st.progress(completed_responses / total_questions)
                    st.write(f"Completed: {completed_responses}/{total_questions} questions")
        
        # Navigation buttons
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("← Back to Market Research", type="secondary"):
                navigate_to_page(3)
                st.rerun()
        
        with col2:
            # Enable next button only if questions are generated
            if hasattr(st.session_state, 'survey_questions') and st.session_state.survey_questions:
                if st.button("Proceed to Next Step →", type="primary"):
                    navigate_to_page(5)
                    st.rerun()
            else:
                st.button("Proceed to Next Step →", disabled=True, help="Generate survey questions first")
    
    else:
        st.error("Please ensure you have selected an idea, defined a problem statement, and chosen SDGs before proceeding.")
        
        # Debug info
        st.subheader("🔍 Debug Information")
        st.write(f"Selected Idea: {st.session_state.get('selected_idea', 'Not set')}")
        st.write(f"Problem Statement: {st.session_state.get('problem_statement', 'Not set')}")
        st.write(f"Selected SDGs: {st.session_state.get('selected_sdgs', 'Not set')}")
        
        # Navigation back button
        if st.button("← Back to Market Research", type="secondary"):
            navigate_to_page(3)
            st.rerun()
            
elif st.session_state.page == 5:
    # PAGE 5: Market Fit Analysis
    st.title("📈 Market Fit Analysis")
    st.header("Step 5: Market Fit and Feedback")
    
    if st.session_state.selected_idea:
        # Display project context in organized columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Selected SDG:** {', '.join(st.session_state.selected_sdgs)}")
            st.info(f"**Selected Idea:** {st.session_state.selected_idea}")
        
        with col2:
            st.info(f"**Problem Statement:** {st.session_state.problem_statement}")
        
        # Expandable sections for detailed information
        with st.expander("📋 View Survey Questions", expanded=False):
            if st.session_state.get('survey_questions'):
                st.write(st.session_state.survey_questions)
            else:
                st.write("No survey questions available yet.")
        
        with st.expander("📊 View Survey Responses", expanded=False):
            if st.session_state.get('survey_responses'):
                st.write(st.session_state.survey_responses)
            else:
                st.write("No survey responses available yet.")
        
        with st.expander("📊 View Market Research Results", expanded=False):
            if st.session_state.get('market_research_results'):
                st.write(st.session_state.market_research_results)
            else:
                st.write("No market research results available yet.")
    
    # Market fit analysis input
    st.subheader("Market Fit Analysis")
    st.write("Analyze how well your solution fits the target market based on research and feedback:")
    
    marketfit = st.text_area(
        "Write your market fit analysis here:",
        value=st.session_state.get('marketfit', ''),
        height=200,
        placeholder="Consider: Target market size, user feedback, competitive landscape, value proposition alignment, pricing strategy, distribution channels, etc.",
        help="Provide a comprehensive analysis of market fit based on your research and user feedback."
    )
    
    # Update session state with current input
    st.session_state.marketfit = marketfit
    
    # Generate Analysis Button
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("📊 Generate Market Fit Analysis", type="primary", use_container_width=True):
            if not marketfit.strip():
                st.error("Please enter your market fit analysis before generating feedback.")
            else:
                with st.spinner("Generating market fit analysis..."):
                    try:
                        # Call the market fit analysis function
                        analysis_result = get_market_fit_feedback(marketfit)
                        st.session_state.market_fit_feedback = analysis_result
                        st.success("Market fit analysis generated successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating analysis: {str(e)}")
    
    # Display generated feedback if available
    if st.session_state.get('market_fit_feedback'):
        st.subheader("📋 Market Fit Feedback")
        with st.container():
            st.write(st.session_state.market_fit_feedback)
        
        if st.button("evaluate market fit", type="secondary", use_container_width=True):
            with st.spinner("Evaluating market fit..."):
                try:
                    logging.info("Evaluating market fit feedback...")
                except Exception as e:
                    st.error(f"Error generating market fit feedback: {str(e)}")
    
    # Navigation buttons
    st.divider()
    
    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
    
    with nav_col1:
        if st.button("← Previous Step", use_container_width=True):
            st.session_state.page = 4
            st.rerun()
    
    with nav_col3:
        # Check if analysis is complete before allowing next step
        if st.session_state.get('market_fit_feedback'):
            if st.button("Next: Prototype →", type="primary", use_container_width=True):
                st.session_state.page = 6
                st.rerun()
        else:
            st.button("Next: Prototype →", disabled=True, use_container_width=True, 
                    help="Complete market fit analysis first")

    # Progress indicator
    st.sidebar.progress(5/6, "Step 5 of 6: Market Fit Analysis")
elif st.session_state.page == 6:
    # PAGE 6: Prototype Generation
    import logging
    import os
    import base64
    
    st.title("🚀 Prototype Image Generator")
    st.header("Step 6: Prototype Generation")

    if st.session_state.get('selected_idea'):
        # Display project context in organized columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Selected SDG:** {', '.join(st.session_state.get('selected_sdgs', []))}")
            st.info(f"**Selected Idea:** {st.session_state.selected_idea}")
        
        with col2:
            st.info(f"**Problem Statement:** {st.session_state.get('problem_statement', 'Not provided')}")
        
       
        
        # Prototype description input
        prototype_description = st.text_area(
            "Describe your prototype visualization:",
            value=st.session_state.get('prototype_description', ''),
            height=150,
            placeholder="Enter a detailed description of your prototype including key features, user interface, materials, technology used, and how it solves the problem...",
            key="prototype_description_input"
        )
        
        # Update session state with prototype description
        if prototype_description:
            st.session_state.prototype_description = prototype_description
        
        # Style and image generation options
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("🎨 Visualization Style")
            
            # Style options with descriptions
            style_options = {
                "Photorealistic Concept": "High-resolution, photorealistic visualization in real-world context",
                "3D Mockup": "Professional 3D render showcasing key features and materials",
                "Whiteboard Sketch": "Innovation sketch with annotations and business model elements",
                "User Interface (UI) Mockup": "Digital product mockup showing user journey and functionality",
                "Infographic Style": "Business-focused infographic with product visualization"
            }
            
            selected_style = st.selectbox(
                "Choose visualization style:",
                options=list(style_options.keys()),
                index=0,
                help="Select the style that best fits your prototype presentation needs"
            )
            
            # Show style description
            st.write(f"**{selected_style}:** {style_options[selected_style]}")
            
           
        
        with col4:
            st.subheader("🔢 Generation Options")
            
            num_images = st.slider(
                "Number of images to generate:",
                min_value=1,
                max_value=4,
                value=2,
                help="Generate multiple variations of your prototype visualization"
            )
            
            st.write(f"**Selected:** {num_images} image{'s' if num_images > 1 else ''}")
        
        # Validation before generation
        can_generate = True
        validation_messages = []
        validation_result = None
        
        # Check basic requirements first
        if not prototype_description or len(prototype_description.strip()) < 50:
            validation_messages.append("⚠️ Please provide a more detailed prototype description (at least 50 characters)")
            can_generate = False
        
        if not st.session_state.get('selected_idea'):
            validation_messages.append("⚠️ Please select an idea first")
            can_generate = False
        
        if not st.session_state.get('problem_statement'):
            validation_messages.append("⚠️ Please provide a problem statement first")
            can_generate = False
        
        # If basic validation passes, run detailed validation
        if can_generate:
            try:
                from validate_student_description import validate_student_input
                
                validation_result = validate_student_input(
                    st.session_state.selected_idea,
                    st.session_state.problem_statement,
                    prototype_description
                )
                
                if not validation_result.get('valid', False):
                    validation_messages.append(f"⚠️ Validation failed: {validation_result.get('message', 'Unknown validation error')}")
                    can_generate = False
                    
            except ImportError:
                st.info("ℹ️ Validation module not found. Basic validation will be used.")
            except Exception as e:
                st.warning(f"⚠️ Validation error: {str(e)}. Proceeding with basic validation.")
        
        # Show validation messages
        if validation_messages:
            for message in validation_messages:
                st.warning(message)
        
        # Show validation success if all checks pass
        if can_generate and validation_result and validation_result.get('valid', False):
            st.success("✅ All validations passed! Ready to generate prototype images.")
        
        # Generation button and logic
        if st.button("🎨 Generate Prototype Images", type="primary", disabled=not can_generate):
            if can_generate:
                with st.spinner("🎨 Generating your prototype visualization... This may take a few moments."):
                    try:
                        # Import prototype generation function
                        try:
                            from protype_image_gen import generate_prototype_images, StyleOption
                            
                            # Convert style string to StyleOption enum
                            style_mapping = {
                                "Photorealistic Concept": StyleOption.PHOTOREALISTIC,
                                "3D Mockup": StyleOption.MOCKUP_3D,
                                "Whiteboard Sketch": StyleOption.WHITEBOARD,
                                "User Interface (UI) Mockup": StyleOption.UI_MOCKUP,
                                "Infographic Style": StyleOption.INFOGRAPHIC
                            }
                            
                            selected_style_enum = style_mapping.get(selected_style, StyleOption.PHOTOREALISTIC)
                            
                            # Get API key from environment variables or Streamlit secrets
                            import os
                            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
                            
                            # Fallback to Streamlit secrets if environment variables not found
                            if not api_key:
                                try:
                                    api_key = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("API_KEY")
                                except:
                                    pass
                            
                            if not api_key:
                                st.error("❌ API key not found. Please set OPENAI_API_KEY in your .env file or environment variables.")
                                
                            
                            # Call the generation function
                            result = generate_prototype_images(
                                idea=st.session_state.selected_idea,
                                problem=st.session_state.problem_statement,
                                prototype_description=prototype_description,
                                style=selected_style_enum,
                                num_images=num_images,
                                api_key=api_key
                            )
                            
                            if result.get("success", False):
                                # Store results in session state
                                st.session_state.prototype_images = result["images"]
                                st.session_state.prototype_prompt = result.get("prompt_used", "")
                                st.session_state.prototype_style = selected_style
                                st.session_state.prototype_generated = True
                                
                                st.success(f"✅ Successfully generated {result.get('num_generated', 0)} prototype image{'s' if result.get('num_generated', 0) > 1 else ''}!")
                                
                            else:
                                st.error(f"❌ Generation failed: {result.get('error', 'Unknown error')}")
                                
                        except ImportError as e:
                            st.error(f"❌ Import error: Could not import prototype generation functions.")
                            st.error(f"Please ensure 'protype_image_gen.py' exists and is accessible.")
                            st.error(f"Error details: {str(e)}")
                            logging.error(f"Import error in prototype generation: {str(e)}")
                            
                            # Fallback: Create mock data for testing
                            st.warning("🧪 Using mock data for testing purposes...")
                            st.session_state.prototype_images = [
                                {
                                    "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
                                    "variation_number": 1
                                }
                            ]
                            st.session_state.prototype_prompt = f"Mock prompt for {selected_style} style prototype"
                            st.session_state.prototype_style = selected_style
                            st.session_state.prototype_generated = True
                            
                    except Exception as e:
                        st.error(f"❌ An unexpected error occurred: {str(e)}")
                        logging.error(f"Unexpected error in prototype generation: {str(e)}")
                
                # Rerun to show the generated images
                st.rerun()
    
        # Display generated images if they exist
        if st.session_state.get('prototype_generated') and st.session_state.get('prototype_images'):
            st.divider()
            st.subheader("🎨 Generated Prototype Visualizations")
            
            # Display generation info
            col5, col6 = st.columns(2)
            with col5:
                st.info(f"**Style Used:** {st.session_state.get('prototype_style', 'Unknown')}")
            with col6:
                st.info(f"**Images Generated:** {len(st.session_state.prototype_images)}")
            
            # Display images in a grid
            if len(st.session_state.prototype_images) == 1:
                # Single image - full width
                img_data = st.session_state.prototype_images[0]
                try:
                    image_bytes = base64.b64decode(img_data["image_base64"])
                    st.image(image_bytes, caption=f"Prototype Visualization", use_container_width=True)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Image",
                        data=image_bytes,
                        file_name=f"prototype_visualization.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"Error displaying image: {str(e)}")
                    
            else:
                # Multiple images - use columns
                cols = st.columns(min(len(st.session_state.prototype_images), 2))
                
                for idx, img_data in enumerate(st.session_state.prototype_images):
                    try:
                        image_bytes = base64.b64decode(img_data["image_base64"])
                        
                        with cols[idx % 2]:
                            st.image(
                                image_bytes, 
                                caption=f"Variation {img_data.get('variation_number', idx + 1)}", 
                                use_container_width=True
                            )
                            
                            # Download button for each image
                            st.download_button(
                                label=f"📥 Download Variation {img_data.get('variation_number', idx + 1)}",
                                data=image_bytes,
                                file_name=f"prototype_variation_{img_data.get('variation_number', idx + 1)}.png",
                                mime="image/png",
                                key=f"download_btn_{idx}"
                            )
                    except Exception as e:
                        st.error(f"Error displaying image {idx + 1}: {str(e)}")
            
            # Show the prompt used (expandable)
            # with st.expander("🔍 View Generation Prompt", expanded=False):
            #     if st.session_state.get('prototype_prompt'):
            #         st.text_area(
            #             "Prompt used for generation:",
            #             value=st.session_state.prototype_prompt,
            #             height=200,
            #             disabled=True
            #         )
            
            # NEW: Evaluation Section
            st.divider()
            st.subheader("📊 Prototype Evaluation")
            
            eval_col1, eval_col2 = st.columns([1, 1])
            
            with eval_col1:
                # Evaluate Description Button
                if st.button("🔍 Evaluate Prototype Description", type="secondary", use_container_width=True):
                    with st.spinner("🔍 Evaluating prototype description..."):
                        try:
                            # Try to import and use evaluation function
                            try:
                                from validate_student_description import evaluate_prototype_description
                                
                                evaluation_result = evaluate_prototype_description(
                                    idea=st.session_state.selected_idea,
                                    problem_statement=st.session_state.get('problem_statement', ''),
                                    prototype_description=st.session_state.get('prototype_description', ''),
                                    selected_sdgs=st.session_state.get('selected_sdgs', [])
                                )
                                
                                logging.info("Prototype description evaluation completed.")
                                logging.info(f"Evaluation result: {evaluation_result}")
                                # Store evaluation results
                                st.session_state.prototype_evaluation = evaluation_result
                                st.session_state.evaluation_completed = True
                                
                                st.success("✅ Prototype evaluation completed!")
                                
                            except ImportError:
                                # Fallback: Create mock evaluation data
                                st.warning("⚠️ Evaluation module not found. Using mock evaluation...")
                                
                                
                        except Exception as e:
                            st.error(f"❌ Evaluation error: {str(e)}")
                            logging.error(f"Prototype evaluation error: {str(e)}")
                    
                    # Rerun to show evaluation results
                    st.rerun()
            
            
            
            # Action buttons
            st.divider()
            col7, col8, col9 = st.columns(3)
            
           
            
            
            
            with col9:
                # Since this is the last page, show completion or summary
                if st.button("✅ Project Complete", type="primary"):
                    st.balloons()
                    st.success("🎉 Congratulations! Your SDG Innovation project is complete!")
                    
                    # Show project summary
                    with st.expander("📋 Project Summary", expanded=True):
                        st.write("**Selected SDGs:**", ', '.join(st.session_state.get('selected_sdgs', [])))
                        st.write("**Idea:**", st.session_state.selected_idea)
                        st.write("**Problem Statement:**", st.session_state.get('problem_statement', 'Not provided'))
                        if st.session_state.get('target_market'):
                            st.write("**Target Market:**", st.session_state.target_market)
                        st.write("**Prototype Style:**", st.session_state.get('prototype_style', 'Not specified'))
                        st.write("**Images Generated:**", len(st.session_state.get('prototype_images', [])))
                        
                        # Include evaluation summary if available
                        if st.session_state.get('prototype_evaluation'):
                            eval_data = st.session_state.prototype_evaluation
                            st.write("**Overall Evaluation Score:**", f"{eval_data.get('overall_score', 0)}/10")
                            st.write("**Evaluation Completed:** ✅")
    
    else:
        # No idea selected - show guidance
        st.warning("⚠️ Please select an idea from the previous steps before generating prototypes.")
        st.write("**To generate prototypes, you need to:**")
        st.write("1. Select SDG goals")
        st.write("2. Choose an idea")
        st.write("3. Define the problem statement")
        st.write("4. Complete the survey and market fit analysis")
        
        if st.button("🔙 Go Back to Start"):
            st.session_state.page = 1  # Go back to the beginning
            st.rerun()

    # Navigation buttons at bottom
    st.divider()
    
    nav_col1, nav_col2 = st.columns([1, 1])
    
    with nav_col1:
        if st.button("← Previous Step", use_container_width=True):
            st.session_state.page = 5
            st.rerun()
    
    with nav_col2:
        # Show reset/restart option
        if st.button("🔄 Start New Project", use_container_width=True):
            # Clear all session state except page
            keys_to_keep = ['page']
            keys_to_delete = [key for key in st.session_state.keys() if key not in keys_to_keep]
            for key in keys_to_delete:
                del st.session_state[key]
            st.session_state.page = 1
            st.rerun()

    # Progress indicator
    st.sidebar.progress(1.0, "Step 6 of 6: Prototype Generation - Complete!")
    
    
# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Current Step:**")
st.sidebar.write(f"Page {st.session_state.page} of 6")

# Progress bar
progress = st.session_state.page / 6
st.sidebar.progress(progress)