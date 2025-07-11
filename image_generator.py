# Save this file as image_generator.py
import streamlit as st
from openai import OpenAI
import base64
from PIL import Image
import io

# A dictionary to provide detailed style guidance to the AI model.
STYLE_GUIDANCE = {
    "Photorealistic Concept": "A high-resolution, photorealistic concept art showing the product in a real-world setting. Emphasize clean lighting and a professional, commercial look.",
    "3D Mockup": "A clean, professional 3D product render on a neutral, light-colored studio background. Focus on the product's form, materials, and details.",
    "Whiteboard Sketch": "A clear, detailed digital sketch as if drawn on a whiteboard or in a notebook with a black marker. Use simple lines, shading, and annotations to explain features.",
    "User Interface (UI) Mockup": "A high-fidelity UI/UX mockup for a modern mobile app or website. Show a specific screen with clear, readable elements. Display it on a generic smartphone or laptop screen.",
    "Infographic Style": "An infographic-style visualization explaining how the product or service works. Use icons, arrows, and simple text to show a process or flow."
}

def add_prototype_generation_step(idea: str, problem: str, target_market: str, api_key: str) -> bool:
    """
    Manages the UI for generating prototype images, with options for up to 4 images and a 2x2 grid display.
    """
    st.markdown("Based on your project details, let's create some visual prototypes to bring your idea to life.")

    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}"); return False

    st.subheader("1. Describe Your Prototype")
    prototype_description = st.text_area(
        "Describe the visual appearance and key features of your prototype. Be specific!",
        placeholder="e.g., A solar-powered water purification device that is portable and has a blue, durable casing with a visible filter compartment.",
        height=150
    )

    st.subheader("2. Choose Your Visualization Style")
    selected_style = st.selectbox("Select the style for your prototype image:", options=list(STYLE_GUIDANCE.keys()))

    st.subheader("3. Select Number of Images")
    num_images = st.slider(
        "How many image variations would you like to generate?",
        min_value=1,
        max_value=4,
        value=2 # Default to 2 images
    )
    
    if st.button("🎨 Generate Prototypes", type="primary", key="generate_prototypes_btn", disabled=not prototype_description):
        with st.spinner(f"Generating {num_images} visualization(s)... This can take a minute."):
            style_guide = STYLE_GUIDANCE[selected_style]
            final_prompt = f"""
            **Project Context:** Idea: {idea}, Problem: {problem}, Target Audience: {target_market}.
            **Visualization Task:**
            - **Core Description of Prototype:** {prototype_description}
            - **Required Style:** {selected_style}. {style_guide}
            Create a visually appealing, high-quality image based on the core description and the required style.
            """
            
            try:
                response = client.images.generate(
                    model="gpt-image-1",
                    prompt=final_prompt.strip(),
                    n=num_images,
                    size="1024x1024"
                )
                st.session_state.prototype_generation_result = {
                    "success": True,
                    "images": [img.b64_json for img in response.data],
                    "prompt": final_prompt.strip()
                }
                st.rerun()

            except Exception as e:
                st.error(f"❌ An error occurred during image generation: {e}")
                st.session_state.prototype_generation_result = {"success": False, "error": str(e)}

    if 'prototype_generation_result' in st.session_state and st.session_state.prototype_generation_result.get("success"):
        result = st.session_state.prototype_generation_result
        st.success("✅ Prototype(s) generated successfully!")
        
        st.subheader(f"🖼️ Your Generated Prototype(s)")
        with st.expander("Show prompt used for generation"):
            st.markdown(f"```\n{result['prompt']}\n```")

        # Create a 2x2 grid structure. This is robust and works for 1, 2, 3, or 4 images.
        grid_rows = [st.columns(2) for _ in range(2)]
        
        # Iterate through the generated images and place them in the grid
        for i, b64_json_string in enumerate(result["images"]):
            row_index = i // 2  # Determines the row (0 or 1)
            col_index = i % 2   # Determines the column (0 or 1)
            
            with grid_rows[row_index][col_index]:
                try:
                    image_bytes = base64.b64decode(b64_json_string)
                    image = Image.open(io.BytesIO(image_bytes))
                    st.image(image, caption=f"Prototype Variation {i+1}", use_column_width=True)
                except Exception as e:
                    st.error(f"Error displaying image {i+1}: {e}")
            
    return st.session_state.get('prototype_generation_result', {}).get('success', False)