import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
import streamlit as st
import yaml
from dotenv import load_dotenv

base_dir = os.path.dirname(os.path.abspath("__file__"))
load_dotenv(os.path.join(base_dir, ".env"))


# Set page configuration
st.set_page_config(
    page_title="Ant-AI by Antematter",
    page_icon="https://antematter.io/images/logo.svg",
)


st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stAppDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)


# Add Beehiiv API configuration
BEEHIIV_PUBLICATION_ID = os.getenv('BEEHIIV_PUBLICATION_ID')
BEEHIIV_API_KEY = os.getenv('BEEHIIV_API_KEY')


def subscribe_to_beehiiv(email,name):
    # Define the endpoint URL using the publication ID
    def is_valid_email(email):
    # Define a regex pattern for validating an email
        pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        
        # Use re.match to check if the pattern matches the provided email
        if re.match(pattern, email):
            return True
        else:
            return False
    
    if not is_valid_email(email):
        return False, "Invalid Email Address"
    
    url = f'https://api.beehiiv.com/v2/publications/{BEEHIIV_PUBLICATION_ID}/subscriptions'

    # Create the payload with only the email of the user
    payload = {
        "email": email,
        "name": name,
        "reactivate_existing": True,
        "send_welcome_email": True,
        "utm_source": "website_newsletter",
    }

    # Set the headers, including the API key for authorization
    headers = {
        'Authorization': f'Bearer {BEEHIIV_API_KEY}',
        'Content-Type': 'application/json'
    }

    try:
        # Make a POST request to the Beehiiv API to create a subscription
        response = requests.post(url, json=payload, headers=headers)

        # Check if the request was successful
        if response.status_code == 201:  
            return True, "Successfully Subscribed!"
        else:
            print(f"Error: Email Subscription Request failed with status code {response.status_code}: {response.text}")
            return False, "Something went wrong. Please try again later."

    except requests.RequestException as e:
        # Handle any exceptions during the request
        return False, f"Error: {str(e)}"
    




if 'subscribed' not in st.session_state:
    st.session_state.subscribed = False

# Newsletter banner with improved styling
st.markdown(
"""
<div style="padding:15px 15px 0px 15px; border-radius:10px; display: flex; align-items: center; justify-content: center;">
<img src="https://media.beehiiv.com/cdn-cgi/image/fit=scale-down,format=auto,onerror=redirect,quality=80/uploads/publication/logo/474e213b-3d7a-49ae-9260-c00e2294271f/Frame_35100.png" height="120" alt="Logo" style="border-radius: 10%; margin-right: 20px;">
<div style="display: flex; flex-direction: column; align-items: center;">
<span style="font-weight:bold; font-size: 40px; margin-top:-10px; margin-bottom:-10px;">Subscribe to The Antedote</span>
<span style="color:gray; font-size: 23px; margin-bottom:0px;">Sharp takes on making AI agents work (and be reliable).</span>
</div>
</div>
""",
unsafe_allow_html=True
)

# Add custom styling for the button
st.markdown("""
<style>
    /* Target both the button and its container */
    .stButton button {
        background-color: #C5F873 !important;
        color: black !important;
        border: none !important;
        font-weight: bold !important;
        font-size: 18px !important;
        width: 100% !important;
        margin: 0 auto !important;
        display: block !important;
        border-radius: 5px !important;
    }
    
    /* Hover state */
    .stButton button:hover {
        background-color: #b3e066 !important;
        color: black !important;
    }
    
    /* Remove default Streamlit button styling */
    .stButton button:active, .stButton button:focus {
        background-color: #C5F873 !important;
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)

# Add the button as a Streamlit component
col1, col2, col3, col4 = st.columns([1,6,3,1])

if not st.session_state.subscribed:
    with col2:
        email = st.text_input("Email", placeholder="Enter your email address", label_visibility="collapsed")
    with col3:
        if st.button("**Dose me up!**", 
                    key="subscribe_button",
                    use_container_width=True):
            if email:
                success, message = subscribe_to_beehiiv(email, "Ant-AI User")
                if success:
                    with st.spinner("Subscribing..."):
                        time.sleep(1)
                    st.success("Successfully Subscribed!")
                    time.sleep(2)
                    st.session_state.subscribed = True
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.error("Please enter your email address")
else:
    # Create three equal columns when subscribed
    sub_col1, sub_col2, sub_col3 = st.columns([1,9,1])
    with sub_col2:
        st.markdown(
            f"""
            <div style="background-color: #C5F873; padding: 5px; border-radius: 5px; text-align: center;">
                <p style="margin: 0; font-weight: bold; color: black;">Subscribed</p>
            </div>
            """,
            unsafe_allow_html=True
        )

st.write("") # Add a blank line

SVG_LOGO = "https://antematter.io/images/logo.svg"
st.logo(SVG_LOGO, size="Large", link="https://antematter.io/",icon_image="https://antematter.io/images/logo.svg")


# Combined intro header
st.header("Welcome to Ant-AI! 🤖")
st.write("""
Ant-AI utilizes a multi-agent framework to optimize task prompts for Large Language Models, improving clarity, relevance, and structure. 
Explore how our platform can enhance your prompt-building experience.
""")

# Layman Prompt section
st.subheader("Layman Prompt")
st.write("""
A Layman Prompt is a simple and clear description of the task or problem you want the AI to address.
""")
st.info("""
**A good layman prompt is that which:**
- Describe the task or challenge clearly and succinctly.
- Focus on the core details that define the problem or objective.
""")

# Persona Section
st.subheader("Persona 🎭")
st.write("""
A Persona provides a specific role or perspective for the AI, ensuring responses align with desired expertise and style. 
This helps tailor the AI's output to your specific needs.
""")
st.info("""
**Nailing the right persona:**
- **Expertise**: Decide the level of knowledge required.
- **Tone**: Choose the style or formality for the output.
""")

# Constraints Section
st.subheader("Constraints 🚧")
st.write("""
Constraints set boundaries for the AI's output, ensuring it aligns with your quality standards.
""")
st.info("""
**Using contraints properly:**
- Be specific to reduce ambiguity.
- Define word counts, format requirements, and critical points.
""")

# Modes Section
st.subheader("Dynamic Agents Mode ⚙️")
st.write("""
Have Ant-AI automatically select the best agents for your task based on the provided prompt, persona, and constraints rather than using our fixed Outline, Research, and Tone Expert Agents.
""")

# Closing Statement
st.markdown("""
Experiment with prompts, personas, and constraints to achieve optimal interactions with AI using Ant-AI.
""")

# Examples Section
st.subheader("Examples 📚")
def display_example(title, prompt, persona, constraints):
    with st.expander(f"{title}"):
        st.subheader("Layman Prompt")
        st.markdown(prompt)
        st.subheader("Persona")
        st.markdown(persona)
        st.subheader("Constraints")
        st.markdown(constraints)



data = yaml.safe_load(open(os.path.join(base_dir,"src","ant_ai","usecases.yaml"), 'r'))
# Iterate over each prompt and call the display_example function
for prompt in data['Prompts']:
    title = prompt['Title']
    formatted_prompt = prompt['Prompt']
    persona = prompt['Persona']
    constraints = prompt['Constraints']

    display_example(title, formatted_prompt, persona, constraints)