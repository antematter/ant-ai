import streamlit as st
import base64
from pathlib import Path
import os

st.set_page_config(
    page_title="Welcome to Ant-AI",
    page_icon="🤖",
)

SVG_LOGO = "https://antematter.io/images/logo.svg"  
st.logo(SVG_LOGO, size="Large", link="https://antematter.io/",icon_image="https://antematter.io/images/logo.svg")


st.markdown(
    """
    <div style="padding:15px; border-radius:10px; display: flex; align-items: center; justify-content: center;">
        <img src="https://media.beehiiv.com/cdn-cgi/image/fit=scale-down,format=auto,onerror=redirect,quality=80/uploads/publication/logo/474e213b-3d7a-49ae-9260-c00e2294271f/Frame_35100.png" height="120" alt="Logo" style="border-radius: 10%; margin-right: 20px;">
        <div style="display: flex; flex-direction: column; align-items: center;">
            <span style="font-weight:bold; font-size: 40px; margin-top:-10px; margin-bottom:-20px;">Subscribe to our Antedote</span>
            <span style="color:gray; font-size: 23px; margin-bottom:-10px;">Sharp takes on making AI agents work (and be reliable).</span>
            <a href="https://antedote.antematter.io/" target="_blank" style="text-decoration:none;">
                <button style="background-color: #C5F873; color: black; border: none; padding: 10px 200px; font-size: 18px; font-weight: bold; border-radius: 5px; cursor: pointer; margin-top: 10px;">
                    Subscribe Now!
                </button>
            </a>
        </div>
        <br>
    </div>
    """,
    unsafe_allow_html=True
)


st.session_state.layman_prompt = ""
st.session_state.persona = ""
st.session_state.constraints = ""


st.write("# Welcome to Ant-AI! 🤖")

# Introduction Section
st.header("🌟 Introduction to Ant-AI")
st.write("""
Ant-AI uses cutting-edge technology to optimize task prompts for Large Language Models, improving clarity, relevance, and structure.
Follow the sections below to get acquainted with the platform and enhance your prompt-building experience.
""")

# Interactive Examples Section
st.header("📝 What is a Layman Prompt?")
st.write("A clear, concise description of the task or problem you want the AI to address.")

def display_example(title, problem, prompt, persona, constraints):
    with st.expander(f"{title}"):
        st.subheader("Problem")
        st.markdown(problem)
        st.subheader("Layman Prompt")
        st.markdown(prompt)
        st.subheader("Persona")
        st.markdown(persona)
        st.subheader("Constraints")
        st.markdown(constraints)

# Display Examples
display_example(
    "Example 1: Customer Identification Challenges",
    "**Problem**: Companies struggle to identify and focus on their best-fit customers, causing wasted resources and lower conversion rates.",
    "Antematter is an AI consultancy firm that specializes in Agentic AI and Blockchain. You will be required to create a comprehensive list of ICPs for Antematter. You will identify their traits which must be sharp and not generic. You must also identify their objectives and pain-points which Antematter can potentially address.",
    "Fortune 500 Growth Specialist with over 20 years of experience in identifying sharp ICPs for SMBs and enterprises.",
    "Ensure that the ICPs are sharp and not generic. I need at least 3 ICPs with proper descriptions. Make sure they’re not generic."
)

display_example(
    "Example 2: Organizational Efficiency in Startups",
    "**Problem**: Startups face operational inefficiencies and misaligned teams due to a lack of clear and scalable Standard Operating Procedures (SOPs).",
    "I want you to write a thorough yet sharp SOP on QA for Antematter which is a Fortune 500 AI consultancy firm that primarily deals with Agentic AI and blockchain. The SOP should include instructions for developers before pushing code on GitHub and then instructions for the QA Engineer once the code has been pushed. The SOP should also include a section for production release, i.e., what protocols to follow when you’re about to deliver the product.",
    "CTO of multiple Fortune 500 companies with 30 years of experience in building teams from scratch.",
    "The SOP should be sharp with no fluff in it. The sections must be coherent within each other. I need a complete document at the end."
)

# Persona Section
st.header("🎭 What is a Persona?")
st.write("""
A Persona gives a unique perspective or role for the AI to adopt, ensuring responses align with specific viewpoints or areas of expertise.
""")
st.info("""
**Choosing the Best Persona for Your Problem:**
- **Audience**: Identify who will use the output.
- **Expertise**: Determine required knowledge level.
- **Tone**: Decide on the style or formality.
""")

# Constraints Section
st.header("🚧 What are Constraints?")
st.write("""
Constraints are rules or parameters that set boundaries for the AI's output, ensuring it meets your quality standards.
""")
st.info("""
**Using Constraints Optimally:**
- Be specific to reduce ambiguity.
- Define word counts, format, and critical points.
- Allow for creativity within defined boundaries.
""")

# Modes Section
st.header("⚙️ Modes in Ant-AI")
st.write("""
Select a mode that fits your needs to customize the experience:
- **Verbose Mode**: Get detailed logs and outputs for better insight.
- **Fast Mode**: Prioritize speed to receive quicker feedback.
- **Dynamic Agent Mode**: Create tailored agent roles for unique prompt optimization.
""")

# Closing Statement
st.markdown("""
We hope these guidelines enhance your experience with Ant-AI. Feel free to experiment with prompts, personas, and constraints to reach your optimal interaction with AI.
""")