import asyncio
import os
import time

import streamlit as st
import yaml

from ant_ai import StreamlitDef as std
from ant_ai import Agents



SVG_LOGO = "https://antematter.io/images/logo.svg"  # Replace with your actual SVG URL
st.logo(SVG_LOGO, size="Large", link="https://antematter.io/",icon_image="https://antematter.io/images/logo.svg")

os.environ['OTEL_SDK_DISABLED'] = 'true'



if 'layman_prompt' not in st.session_state:
    st.session_state.layman_prompt = ""
if 'persona' not in st.session_state:
    st.session_state.persona = ""
if 'constraints' not in st.session_state:
    st.session_state.constraints = ""
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""   
if 'final_prompt' not in st.session_state:
    st.session_state.final_prompt = None
if 'time_taken' not in st.session_state:
    st.session_state.time_taken = None
if 'system_logs' not in st.session_state:
    st.session_state.system_logs = ""

def update_layman_prompt():
    st.session_state["layman_prompt"] = st.session_state["layman_text_area"]

def update_persona():
    st.session_state["persona"] = st.session_state["persona_text_area"]

def update_constraints():
    st.session_state["constraints"] = st.session_state["constraints_text_area"]

def update_api_key():
    st.session_state["api_key"] = st.session_state["api_key_text"]



def main():


    st.set_page_config(
    page_title="Ant-AI by Antematter",
    page_icon="https://antematter.io/images/logo.svg"
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




    st.title("Prompt Optimization using Multi-Agentic Framework")

    # Sidebar buttons to fill in example data
    with st.sidebar:
        base_dir = os.path.dirname(os.path.abspath("__file__"))
        data = yaml.safe_load(open(os.path.join(base_dir,"src","ant_ai","usecases.yaml"), 'r'))
        
        for prompt in data['Prompts']:
            title = prompt['Title']
            if st.button(f'Use {title}'):
                st.session_state.layman_prompt = prompt['Prompt']
                st.session_state.persona = prompt['Persona']
                st.session_state.constraints = prompt['Constraints']
                st.session_state.final_prompt = None
                st.session_state.time_taken = None
                st.session_state.system_logs = ""
                st.rerun()
                st.success(f"{title} data loaded!")
        
        st.divider()

        if st.button('Clear All'):
            st.session_state.layman_prompt = ""
            st.session_state.persona = ""
            st.session_state.constraints = ""
            st.session_state.final_prompt = None
            st.session_state.time_taken = None
            st.session_state.system_logs = ""
            st.session_state.api_key = ""
            st.rerun()
            st.toast("All fields cleared.")

    # Input fields using session state values
    LaymanPrompt = st.text_area(
        "Layman Prompt", 
        height=200, 
        placeholder="Enter your prompt",
        value=st.session_state.get("layman_prompt", ""),
        key="layman_text_area",
        on_change=update_layman_prompt
    )
    
    Persona = st.text_area(
        "Persona", 
        height=68, 
        placeholder="Enter your prompt persona",
        value=st.session_state.get("persona", ""),
        key="persona_text_area",
        on_change=update_persona
    )
    
    Constraints = st.text_area(
        "Constraints", 
        height=100, 
        placeholder="Enter your prompt constraints",
        value=st.session_state.get("constraints", ""),
        key="constraints_text_area",
        on_change=update_constraints
    )

    # Simple API Key Input below Constraints
    api_key = st.text_input(
        "OpenAI API Key", 
        type="password", 
        placeholder="Enter your API Key", 
        value=st.session_state.get("api_key", ""),
        key="api_key_text" ,
        on_change=update_api_key
    )

    async def process_optimization(dynamic_agent_mode):
        st.session_state.layman_prompt = LaymanPrompt
        st.session_state.persona = Persona
        st.session_state.constraints = Constraints
        if LaymanPrompt and Persona and Constraints:
            api_key = st.session_state.get('api_key', '')  # Get API key from session state
            if not api_key:
                st.warning("Please enter your OpenAI API Key before optimizing.")
                return
            if not Agents.check_openai_api_key(api_key):
                st.warning("Invalid OpenAI API Key.")
                return
            
            

            # Start measuring time
            start_time = time.time()

            info_message = st.info("⏳ The optimization process typically takes **1-2 minutes**. Please wait while we craft your perfect prompt.", icon="ℹ️")
            st.balloons()

            # Show loading spinner
            with st.spinner('Optimizing...'):
                FinalPrompt_raw, Status, SystemLogs = await std.OptimizePrompt_async_fast(
                    LaymanPrompt, 
                    Persona, 
                    Constraints, 
                    api_key,
                    dynamic_agent_mode
                )
            if Status:
                # Calculate time taken
                end_time = time.time()
                time_taken = end_time - start_time

                
                # Store results in session state
                st.session_state.final_prompt = FinalPrompt_raw
                st.session_state.time_taken = time_taken
                st.session_state.system_logs = SystemLogs

                # Display results
                display_optimization_results()
            else:
                st.warning("Something went wrong with the optimization process.")
        else:
            st.warning("Please fill in all the fields to proceed with optimization.")
    
        # Remove the info message after optimization is complete
        info_message.empty()

    # Add this new function to display results
    def display_optimization_results_with_logs():
        if st.session_state.final_prompt is not None:
            if st.session_state.system_logs is not None:
                st.expander("System Logs", expanded=False).text(st.session_state.system_logs)
            st.success("Optimization Complete")
            if st.session_state.time_taken is not None:
                seconds = st.session_state.time_taken % 60
                minutes = st.session_state.time_taken // 60
                st.write(f"Time taken: {int(minutes)} minutes and {int(seconds)} seconds")
            st.subheader("Final Prompt:")
            st.code(st.session_state.final_prompt, language="text", wrap_lines=True)
    
    def display_optimization_results():
        if st.session_state.final_prompt is not None:
            st.success("Optimization Complete")
            if st.session_state.time_taken is not None:
                seconds = st.session_state.time_taken % 60
                minutes = st.session_state.time_taken // 60
                st.write(f"Time taken: {int(minutes)} minutes and {int(seconds)} seconds")
            st.subheader("Final Prompt:")
            st.code(st.session_state.final_prompt, language="text", wrap_lines=True)

    # Create a container for the optimize button with custom styling
    optimize_button_container = st.container()
    with optimize_button_container:
        optimize_button = st.button(
            "Optimize Prompt",
            use_container_width=True,
            type="primary"
        )

    # Dynamic agent mode in a separate row, right-aligned
    col1, col2 = st.columns([7, 3])
    with col1:
        st.write("")
    with col2:
        dynamic_agent_mode = st.checkbox(
            "Dynamic Agent Mode",
            help="Enable advanced optimization using dynamic agent selection"
        )

    if optimize_button:
        # Run optimization with current values
        asyncio.run(process_optimization(dynamic_agent_mode))
    else:
        # Display previous results if they exist
        display_optimization_results_with_logs()

if __name__ == "__main__":
    main()