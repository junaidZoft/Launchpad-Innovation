# streamlit_app.py

import asyncio
import streamlit as st

from utils_groq_PT import OptimizedPrototypeClassifier

st.set_page_config(page_title="Prototype Classifier Description")
st.title(" Prototype-Description Classifier")

# Initialize classifier once (cached across reruns)
if "classifier" not in st.session_state:
    st.session_state.classifier = OptimizedPrototypeClassifier()

description = st.text_area(
    "Enter prototype description:",
    placeholder="Type your prototype details here...",
    height=150,
)

# Button to start async classification
if st.button("Classify"):
    # Disable reruns while processing
    with st.spinner("Classifying..."):
        # Run the async classify() inside Streamlit
        x_cat, y_cat = asyncio.run(
            st.session_state.classifier.classify(description)
        )
    st.success("Done!")

    # Display results
    st.subheader("Classification Results")
    st.write(f"**X-Axis Category:** {x_cat}")
    st.write(f"**Y-Axis Category:** {y_cat}")
