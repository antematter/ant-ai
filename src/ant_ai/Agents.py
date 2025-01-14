import sys
import locale

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

import openai
from openai import OpenAI, AsyncOpenAI
import yaml

from crewai import LLM, Agent, Crew, Task
from crewai.flow.flow import Flow, listen, start, and_




# Initialize a new Crew with specified Agent and Task settings from YAML files.
def GetCrew(
    AgentYamlFile,
    TaskYamlFile,
    LLM,
    verbose,
    allow_delegation,
    Pydantic=None,
    Async=False,
):
    with open(AgentYamlFile, "r", encoding="utf-8") as yaml_file:
        agent_yaml = yaml.safe_load(yaml_file)
    with open(TaskYamlFile, "r", encoding="utf-8") as yaml_file:
        task_yaml = yaml.safe_load(yaml_file)

    NewAgent = Agent(
        role=agent_yaml["role"],
        goal=agent_yaml["goal"],
        backstory=agent_yaml["backstory"],
        verbose=verbose,
        allow_delegation=allow_delegation,
        llm=LLM,
        max_execution_time=60
    )
    if Pydantic is None:
        NewTask = Task(
            description=task_yaml["description"],
            expected_output=task_yaml["expected_output"],
            agent=NewAgent,
            async_execution=Async,
        )
    else:
        NewTask = Task(
            description=task_yaml["description"],
            expected_output=task_yaml["expected_output"],
            agent=NewAgent,
            output_pydantic=Pydantic,
            async_execution=Async,
        )
    NewCrew = Crew(agents=[NewAgent], tasks=[NewTask],share_crew=False)
    return NewCrew

# Initialize a new Crew with specified Agent and Task settings from YAML files.
def GetCrewWithAgent(
    Agent, TaskYamlFile, OutputFile, Pydantic=None, Async=False
):
    with open(TaskYamlFile, "r", encoding="utf-8") as yaml_file:
        task_yaml = yaml.safe_load(yaml_file)

    if Pydantic is None:
        NewTask = Task(
            description=task_yaml["description"],
            expected_output=task_yaml["expected_output"],
            agent=Agent,
            async_execution=Async,
        )
    else:
        NewTask = Task(
            description=task_yaml["description"],
            expected_output=task_yaml["expected_output"],
            agent=Agent,
            output_pydantic=Pydantic,
            async_execution=Async,
        )
    NewCrew = Crew(agents=[Agent], tasks=[NewTask], output_log_file=OutputFile,cache=False)
    return NewCrew

# Send a synchronous chat request to the GPT-4o-mini model and return the generated response.
def InvokeGpt4oMini(prompt):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    messages = [{"role": "user", "content": prompt}]
    chat = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.3)
    reply = chat.choices[0].message.content
    return reply

# Asynchronously send a chat request to the GPT-4o-mini model using the provided OpenAI client.
async def invoke_gpt4o_mini(prompt, client):
    messages = [{"role": "user", "content": prompt}]
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
    )
    reply = response.choices[0].message.content
    return reply

# Collect asynchronous responses for a batch of prompts using the GPT-4o-mini model.
async def gather_responses(prompts):
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    tasks = [invoke_gpt4o_mini(prompt, client) for prompt in prompts]
    responses = await asyncio.gather(*tasks)
    return responses

# Verify the validity of input parameters by generating and evaluating a guardrail prompt with a model response.
def ValidityCheck(prompt, persona, constraints, TaskYamlFile, BaseModel):
    with open(TaskYamlFile, "r", encoding="utf-8") as yaml_file:
        task_yaml = yaml.safe_load(yaml_file)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    GuardrailTemplate=task_yaml["GuardrailTemplate"]
    GuuardrailPrompt=GuardrailTemplate.format(LaymanPrompt=prompt, Persona=persona, Constraints=constraints)
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",  # Ensure you're using the correct GPT-4o model
        messages=[
            {"role": "system", "content": GuuardrailPrompt},  # Pass the formatted template as a system message
        ],
        response_format=BaseModel
    )

    response=completion.choices[0].message
    return response.parsed.Validity,response.parsed.Reason

# Verify the validity of input parameters by generating and evaluating a guardrail prompt with a model response.
def check_openai_api_key(api_key):
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
    except openai.AuthenticationError:
        return False
    else:
        return True