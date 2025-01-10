import asyncio
import os
import time

import streamlit as st

from ant_ai import StreamlitDef as std
from ant_ai import Agents

SVG_LOGO = "https://antematter.io/images/logo.svg"  # Replace with your actual SVG URL
st.logo(SVG_LOGO, size="Large", link="https://antematter.io/",icon_image="https://antematter.io/images/logo.svg")

os.environ['OTEL_SDK_DISABLED'] = 'true'



# Examples Data
example_1 = {
    "layman_prompt": "Antematter is an AI consultancy firm that specializes in Agentic AI and Blockchain. You will be required to create a comprehensive list of ICPs for Antematter. You will identify their traits which must be sharp and not generic. You must also identify their objectives and pain-points which Antematter can potentially address.",
    "persona": "Fortune 500 Growth Specialist with over 20 years of experience in identifying sharp ICPs for SMBs and enterprises.",
    "constraints": "Ensure that the ICPs are sharp and not generic. I need at least 3 ICPs with proper descriptions. Make sure they’re not generic."
}

example_2 = {
    "layman_prompt": "I want you to write a thorough yet sharp SOP on QA for Antematter which is a fortune 500 AI consultancy firm which primarily deals with Agentic AI and blockchain. The SOP should include instructions for the developers before pushing the code on GitHub and then instructions for the QA Engineer once the code has been pushed. The SOP should also include a section for production release i.e; what protocols to follow when you’re about to deliver the product.",
    "persona": "CTO of multiple fortune 500 companies with 30 years of experience in building teams from scratch.",
    "constraints": "The SOP should be sharp with no fluff in it. The sections must be coherent within each other. I need a complete document at the end."
}




def main():

    st.set_page_config(
    page_title="AntAI: Prompt Optimization",
    page_icon="🤖",
    )


    st.title("AntAI: Genetic Prompt Optimization Multi-Agent Framework")

    # Prompt Input Fields
    LaymanPrompt = st.text_area("Layman Prompt", height=200, placeholder="Enter your prompt",value=st.session_state.get("layman_prompt", ""))
    Personna = st.text_area("Persona", height=68, placeholder="Enter your prompt persona",value=st.session_state.get("persona", ""))
    Constraints = st.text_area("Constraints", height=100, placeholder="Enter your prompt constraints",value=st.session_state.get("constraints", ""))


        # Sidebar buttons to fill in example data
    with st.sidebar:
        if st.button('Use Example 1'):
            st.session_state.layman_prompt = example_1['layman_prompt']
            st.session_state.persona = example_1['persona']
            st.session_state.constraints = example_1['constraints']
            st.rerun()
            st.success("Example 1 data loaded!")

        if st.button('Use Example 2'):
            st.session_state.layman_prompt = example_2['layman_prompt']
            st.session_state.persona = example_2['persona']
            st.session_state.constraints = example_2['constraints']
            st.rerun()
            st.success("Example 2 data loaded!")

        if st.button('Clear All'):
            st.session_state.layman_prompt = ""
            st.session_state.persona = ""
            st.session_state.constraints = ""
            st.rerun()
            st.success("All fields cleared.")


    # Simple API Key Input below Constraints
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="Enter your API Key")

    async def process_optimization(verbose_mode, fast_mode,dynamic_agent_mode):
        if LaymanPrompt and Personna and Constraints:
            if not api_key:
                st.warning("Please enter your API Key before optimizing.")
                return
            if Agents.check_openai_api_key(api_key) == False:
                st.warning("Invalid API Key.")
                return
            # Start measuring time
            start_time = time.time()

            # Show loading spinner
            with st.spinner('Optimizing...'):
                # Replace std.OptimizePrompt_async with your actual asynchronous method
                if fast_mode:
                    FinalPrompt_raw, Status = await std.OptimizePrompt_async_fast(
                        LaymanPrompt, 
                        Personna, 
                        Constraints, 
                        api_key,
                        verbose_mode,
                        dynamic_agent_mode
                    )
                else:
                    FinalPrompt_raw, Status = await std.OptimizePrompt_async(
                        LaymanPrompt, 
                        Personna, 
                        Constraints, 
                        api_key,
                        verbose_mode,
                        dynamic_agent_mode
                    )
            if Status:
                # Calculate time taken
                end_time = time.time()
                time_taken = end_time - start_time

                st.success("Optimization Complete")
                seconds = time_taken % 60
                minutes = time_taken // 60
                st.write(f"Time taken: {int(minutes)} minutes and {int(seconds)} seconds")

                # Display the Final Prompt in an enclosure
                st.subheader("Final Prompt:")
                st.code(FinalPrompt_raw, language="text")
            else:
                st.warning("Something went wrong with the optimization process.")
        else:
            st.warning("Please fill in all the fields to proceed with optimization.")

    # Add a horizontal container with button and checkbox
    col1, col2, col3, col4= st.columns([4, 2, 2, 3])  # Define column layout (button takes more space than the checkbox)

    with col1:
        optimize_button = st.button("Optimize Prompt")  # Button to trigger optimization

    with col2:
        verbose_mode = st.checkbox("Verbose")  # Checkbox for verbose output

    with col3:
        fast_mode = st.checkbox("Fast Mode") # Checkbox for fast mode

    with col4:
        dynamic_agent_mode = st.checkbox("Dynamic Agent Mode") # Checkbox for dynamic agent mode


    # Trigger when the "Optimize Prompt" button is clicked
    if optimize_button:
        asyncio.run(process_optimization(verbose_mode, fast_mode,dynamic_agent_mode))


if __name__ == "__main__":
    main()