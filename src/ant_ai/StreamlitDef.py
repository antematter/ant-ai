import os
import random
import asyncio
import copy

from pydantic import BaseModel
import streamlit as st
import yaml

import Agents
import Logger

#Default GA Configuration
default_config = {
    "MaxGenerations": 3,  # Integer
    "SelectionSize": 2,   # Integer
    "TopCarry": 2,        # Integer
    "TopCarryRate": 0.5   # Float
}

#Pydantic Models

class Agent(BaseModel):
    agent_name: str
    agent_goal: str
    agent_backstory: str
    agent_task: str

class AgentCreation(BaseModel):
    agents: list[Agent]

class CritiqueReport(BaseModel):
    SpecializedGuidance : str
    SpecializedCritique : str

class GeneralEvaluation(BaseModel):
    ScoreOutof90 : int

class ReasoningEvaluation(BaseModel):
    ScoreOutof70 : int

class CritiqueEvaluation(BaseModel):
    RefinedPromptCritique : str
    ActionableSteps : str

class ValidityCheck(BaseModel):
    Validity : bool
    Reason : str


base_dir = os.path.dirname(os.path.abspath("__file__"))

# Prepare and configure various agents for prompt refinement tasks.
def prepare_agents(verbose_mode):

    global InitialPromptCrew
    InitialPromptCrew = Agents.GetCrew(
        AgentYamlFile = os.path.join(base_dir, "definitions", "AgentDef", "MasterAgent.yaml"),
        TaskYamlFile = os.path.join(base_dir, "definitions", "TaskDef", "PromptGeneration.yaml"),
        OutputFile = os.path.join(base_dir, "logs", "InitialMasterLogs.txt"),
        LLM=Agents.LLM(
        model="gpt-4o-mini",api_key=os.environ["OPENAI_API_KEY"],temperature=0.4
        ),
        verbose=verbose_mode,
        allow_delegation=True,
        Async=True
    )

    global PromptMutationCrew
    PromptMutationCrew = Agents.GetCrew(
        AgentYamlFile = os.path.join(base_dir, "definitions", "AgentDef", "MutationAgent.yaml"),
        TaskYamlFile = os.path.join(base_dir, "definitions", "TaskDef", "MutationTask.yaml"),
        OutputFile = os.path.join(base_dir, "logs", "PromptMutationLogs.txt"),
        LLM=Agents.LLM(
        model="gpt-4o-mini",api_key=os.environ["OPENAI_API_KEY"],temperature=0.8
        ),
        verbose=False,
        allow_delegation=True,
        Async=True
    )

    global PromptCrossoverCrew
    PromptCrossoverCrew = Agents.GetCrew(
        AgentYamlFile = os.path.join(base_dir, "definitions", "AgentDef", "CrossoverAgent.yaml"),
        TaskYamlFile = os.path.join(base_dir, "definitions", "TaskDef", "CrossoverTask.yaml"),
        OutputFile = os.path.join(base_dir, "logs", "CrossoverLogs.txt"),
        LLM=Agents.LLM(
        model="gpt-4o-mini",api_key=os.environ["OPENAI_API_KEY"],temperature=0.7
        ),
        verbose=False,
        allow_delegation=True,
        Async=True

    )

    global PromptGeneralEvaluationCrew
    PromptGeneralEvaluationCrew = Agents.GetCrew(
        AgentYamlFile = os.path.join(base_dir, "definitions", "AgentDef", "EvaluationAgent.yaml"),
        TaskYamlFile = os.path.join(base_dir, "definitions", "TaskDef", "GeneralEvaluationTask.yaml"),
        OutputFile = os.path.join(base_dir, "logs", "EvaluationLogs.txt"),
        LLM=Agents.LLM(
        model="gpt-4o-mini",api_key=os.environ["OPENAI_API_KEY"],temperature=0.2
        ),
        verbose=False,
        allow_delegation=True,
        Pydantic=GeneralEvaluation,
        Async=True
    )

    global PromptReasoningEvaluationCrew
    PromptReasoningEvaluationCrew = Agents.GetCrew(
        AgentYamlFile = os.path.join(base_dir, "definitions", "AgentDef", "EvaluationAgent.yaml"),
        TaskYamlFile = os.path.join(base_dir, "definitions", "TaskDef", "ReasoningEvaluationTask.yaml"),
        OutputFile = os.path.join(base_dir, "logs", "EvaluationLogs.txt"),
        LLM=Agents.LLM(
        model="gpt-4o-mini",api_key=os.environ["OPENAI_API_KEY"],temperature=0.2
        ),
        verbose=False,
        allow_delegation=True,
        Pydantic=ReasoningEvaluation,
        Async=True
    )

    global PromptCritiqueEvaluationCrew
    PromptCritiqueEvaluationCrew = Agents.GetCrew(
        AgentYamlFile = os.path.join(base_dir, "definitions", "AgentDef", "EvaluationAgent.yaml"),
        TaskYamlFile = os.path.join(base_dir, "definitions", "TaskDef", "EvaluationCritiqueTask.yaml"),
        OutputFile = os.path.join(base_dir, "logs", "EvaluationLogs.txt"),
        LLM=Agents.LLM(
        model="gpt-4o-mini",api_key=os.environ["OPENAI_API_KEY"],temperature=0.6
        ),
        verbose=False,
        allow_delegation=True,
        Pydantic=CritiqueEvaluation,
        Async=True
    )

    global DynamicAgentCreationCrew
    DynamicAgentCreationCrew = Agents.GetCrew(
        AgentYamlFile = os.path.join(base_dir, "definitions", "AgentDef", "DynamicAgentManager.yaml"),
        TaskYamlFile = os.path.join(base_dir, "definitions", "TaskDef", "AgentCreation.yaml"),
        OutputFile = os.path.join(base_dir, "logs", "DynamicAgentCreationLogs.txt"),
        LLM=Agents.LLM(
        model="gpt-4o-mini",api_key=os.environ["OPENAI_API_KEY"],temperature=0.3
        ),
        verbose=verbose_mode,
        allow_delegation=True,
        Pydantic=AgentCreation,
        Async=True
    )

    global InitialCritiqueCrew
    InitialCritiqueCrew = Agents.GetCrew(
        AgentYamlFile = os.path.join(base_dir, "definitions", "AgentDef", "CritiquePromptAgent.yaml"),
        TaskYamlFile = os.path.join(base_dir, "definitions", "TaskDef", "CritiquePromptGeneration.yaml"),
        OutputFile = os.path.join(base_dir, "logs", "InitialMetaMasterLogs.txt"),
        LLM=Agents.LLM(
        model="gpt-4o-mini",api_key=os.environ["OPENAI_API_KEY"],temperature=0.6
        ),
        verbose=False,
        allow_delegation=False,
        Pydantic=CritiqueReport,
        Async=True
    )

    global FinalUnificationCrew
    FinalUnificationCrew = Agents.GetCrew(
        AgentYamlFile = os.path.join(base_dir, "definitions", "AgentDef", "CombinationAgent.yaml"),
        TaskYamlFile = os.path.join(base_dir, "definitions", "TaskDef", "CombinationTask.yaml"),
        OutputFile = os.path.join(base_dir, "logs", "CombinationLogs.txt"),
        LLM=Agents.LLM(
        model="gpt-4o",api_key=os.environ["OPENAI_API_KEY"],temperature=0.9
        ),
        verbose=verbose_mode,
        allow_delegation=True,
        Async=True
    )

# Clean up any pending asynchronous tasks and ensure graceful termination.
async def cleanup_asyncio():
    try:
        current_task = asyncio.current_task()
        tasks = [t for t in asyncio.all_tasks() if t is not current_task]
        if tasks:
            for task in tasks:
                task.cancel()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for task, result in zip(tasks, results):
                if isinstance(result, Exception):
                    print(f"Task {task.get_name()} raised during cleanup: {result}")
    except Exception as e:
        print(f"Error during cleanup: {e}")

# Validate and update configuration values for the GA using defaults if needed.
def validate_config(config, defaults):
    validated_config = defaults.copy()
    faulty = False

    for key, default_value in defaults.items():
        if key not in config:
            update_log(f"Missing key '{key}' in config. Using default value: {default_value}.")
            faulty = True
            continue
        if isinstance(default_value, int) and isinstance(config[key], int):
            validated_config[key] = config[key]
        elif isinstance(default_value, float) and isinstance(config[key], (float, int)):
            validated_config[key] = float(config[key])
        else:
            update_log(f"Invalid type for key '{key}'. Expected {type(default_value).__name__}. Using default value: {default_value}.")
            faulty = True

    if faulty:
        update_log("Faulty configuration detected. Using default values where necessary.")

    return validated_config

# Create and configure a new agent based on input parameters.
def get_agent(agent_inputs):
    agent = Agents.Agent(
        role = agent_inputs['AgentName'],
        goal = agent_inputs['AgentGoal'],
        backstory = agent_inputs['AgentBackstory'],
        verbose = False,
        llm = Agents.LLM(
        model="gpt-4o-mini",api_key=os.environ["OPENAI_API_KEY"]
        ),
        temperature = 0.4,
        max_execution_time = 60
    )
    return agent

# Define a genetic model encompassing multiple elements related to the prompt.
class PromptGene:
    
    def __init__(self,initial_prompt="", agent_guidance="", agent_critique="", refined_prompt="", result="", action_steps="", general_score=0, reasoning_score=0, refined_prompt_critique=""):
        self.InitialPrompt = initial_prompt
        self.AgentGuidance = agent_guidance
        self.AgentCritique = agent_critique
        self.RefinedPrompt = refined_prompt
        self.Result = result
        self.ActionSteps = action_steps
        self.GeneralScore = general_score
        self.ReasoningScore = reasoning_score
        self.RefinedPromptCritique = refined_prompt_critique
    

    def display(self):
        print(f"\n\n\n\n\nInitial Prompt: {self.InitialPrompt}")
        print(f"\n\n\n\n\nAgent Guidance: {self.AgentGuidance}")
        print(f"\n\n\n\n\nAgent Critique: {self.AgentCritique}")
        print(f"\n\n\n\n\nRefined Prompt: {self.RefinedPrompt}")
        print(f"\n\n\n\n\nResult: {self.Result}")
        print(f"\n\n\n\n\nAction Steps: {self.ActionSteps}")
        print(f"\n\n\n\n\nGeneral Score: {self.GeneralScore}")
        print(f"\nReasoning Score: {self.ReasoningScore}")
        print(f"\nTotal Score: {self.GeneralScore+self.ReasoningScore}")    
        print(f"\n\n\n\n\nRefined Prompt Critique: {self.RefinedPromptCritique}")

    def getdisplaytext(self):
        return "\n\n\n\n\nInitial Prompt: "+self.InitialPrompt+"\n\n\n\n\nAgent Guidance: "+self.AgentGuidance+"\n\n\n\n\nAgent Critique: "+self.AgentCritique+"\n\n\n\n\nRefined Prompt: "+self.RefinedPrompt+"\n\n\n\n\nResult: "+self.Result+"\n\n\n\n\nAction Steps: "+self.ActionSteps+"\n\n\n\n\nGeneral Score: "+str(self.GeneralScore)+"\nReasoning Score: "+str(self.ReasoningScore)+"\nTotal Score: "+str(self.GeneralScore+self.ReasoningScore)+"\n\n\n\n\nRefined Prompt Critique: "+self.RefinedPromptCritique

    def get(self):
        inputArr={
            'InitialTask': self.InitialPrompt, 
            'SpecializedGuidance': self.AgentGuidance,
            'RefinedTask': self.RefinedPrompt, 
            'RefinedTaskOutput': self.Result
        }
        return inputArr

# Evaluate and update the gene's scores if they've not been assigned.
def check_evaluation(Gene):
    if Gene.GeneralScore==0:
        GeneralScore=PromptGeneralEvaluationCrew.kickoff(inputs=Gene.get())
        Gene.GeneralScore=GeneralScore.pydantic.ScoreOutof90
    if Gene.ReasoningScore==0:
        ReasoningScore=PromptReasoningEvaluationCrew.kickoff(inputs=Gene.get())
        Gene.ReasoningScore=ReasoningScore.pydantic.ScoreOutof70
    return Gene
    

# Crossover and mutate prompts by generating new genes from input populations.
async def CrossoverMutatePrompts(PrimaryPopulation,SecondaryPopulation,MutationPopulation,AgentInputs):
    await cleanup_asyncio()
    if len(PrimaryPopulation)!=len(SecondaryPopulation) or len(PrimaryPopulation)!=len(MutationPopulation):
        return None
    for i in range(len(AgentInputs)):
        PrimaryPopulation[i]=check_evaluation(PrimaryPopulation[i])
        SecondaryPopulation[i]=check_evaluation(SecondaryPopulation[i])
        MutationPopulation[i]=check_evaluation(MutationPopulation[i])

    update_log("- Performing Crossover and Mutation.")

    tasks=[]
    PromptCrossoverCrewCopies=[PromptCrossoverCrew.copy() for _ in range(len(AgentInputs))]
    PromptMutationCrewCopies=[PromptMutationCrew.copy() for _ in range(len(AgentInputs))]
    update_log("- Generating New Prompts (Running Crossover and Mutation).")
    for i in range(len(AgentInputs)):
        task=PromptCrossoverCrewCopies[i].kickoff_async(inputs={
            'InitialTask1':PrimaryPopulation[i].RefinedPrompt,
            'PromptCritique1':PrimaryPopulation[i].RefinedPromptCritique,
            'InitialTask2':SecondaryPopulation[i].RefinedPrompt,
            'PromptCritique2':SecondaryPopulation[i].RefinedPromptCritique,
        })
        tasks.append(task)
    for i in range(len(AgentInputs)):
        task=PromptMutationCrewCopies[i].kickoff_async(inputs={
            'InitialTask':MutationPopulation[i].RefinedPrompt,
            'PromptCritique':MutationPopulation[i].RefinedPromptCritique,
            'ActionSteps':MutationPopulation[i].ActionSteps
        })
        tasks.append(task)
    NewPrompts=await asyncio.gather(*tasks)
    CrossoverPrompts=NewPrompts[:len(AgentInputs)] 
    MutationPrompts=NewPrompts[len(AgentInputs):]    
    update_log("- Generating New Prompt Critiques.")
    tasks=[]
    InitialCritiqueCrewCopies1=[InitialCritiqueCrew.copy() for _ in range(len(AgentInputs))]
    InitialCritiqueCrewCopies2=[InitialCritiqueCrew.copy() for _ in range(len(AgentInputs))]
    for i in range(len(AgentInputs)):
        task=InitialCritiqueCrewCopies1[i].kickoff_async(inputs={
            'InitialTask':CrossoverPrompts[i].raw,
            'AgentName':AgentInputs[i]['AgentName'],
            'AgentBackstory':AgentInputs[i]['AgentBackstory'],
            'AgentGoal':AgentInputs[i]['AgentGoal'],
            'AgentTask':AgentInputs[i]['AgentTask']
        })
        tasks.append(task)
    for i in range(len(AgentInputs)):
        task=InitialCritiqueCrewCopies2[i].kickoff_async(inputs={
            'InitialTask':MutationPrompts[i].raw,
            'AgentName':AgentInputs[i]['AgentName'],
            'AgentBackstory':AgentInputs[i]['AgentBackstory'],
            'AgentGoal':AgentInputs[i]['AgentGoal'],
            'AgentTask':AgentInputs[i]['AgentTask']
        })
        tasks.append(task)
    UpdatedCritiques = await asyncio.gather(*tasks)
    CrossoverCritiques=UpdatedCritiques[:len(AgentInputs)]
    MutationCritiques=UpdatedCritiques[len(AgentInputs):]
    
    PromptGenerator=[]
    for i in range(len(AgentInputs)):
        PromptGenerator.append(
            Agents.GetCrewWithAgent(
                Agent=get_agent(AgentInputs[i]),
                TaskYamlFile= os.path.join(base_dir, "definitions", "TaskDef", "SpecializedPromptGeneration.yaml"),
                OutputFile= os.path.join(base_dir, "logs", "SpecializedPromptLogs.txt"),
            )
        )
    update_log("- Generating New Refined Prompts.")
    PromptGeneratorCopies1=[PromptGenerator[i].copy() for i in range(len(AgentInputs))]
    PromptGeneratorCopies2=[PromptGenerator[i].copy() for i in range(len(AgentInputs))]

    tasks=[]
    for i in range(len(AgentInputs)):
        task=PromptGeneratorCopies1[i].kickoff_async(inputs={
            'InitialTask':CrossoverPrompts[i].raw,
            'SpecializedGuidance':CrossoverCritiques[i].pydantic.SpecializedGuidance,
            'SpecializedCritique':CrossoverCritiques[i].pydantic.SpecializedCritique
        })
        tasks.append(task)
    for i in range(len(AgentInputs)):
        task=PromptGeneratorCopies2[i].kickoff_async(inputs={
            'InitialTask':MutationPrompts[i].raw,
            'SpecializedGuidance':MutationCritiques[i].pydantic.SpecializedGuidance,
            'SpecializedCritique':MutationCritiques[i].pydantic.SpecializedCritique
        })
        tasks.append(task)
    UpdatedPrompts=await asyncio.gather(*tasks)
    CrossoverRefinedPrompts=UpdatedPrompts[:len(AgentInputs)]
    MutationRefinedPrompts=UpdatedPrompts[len(AgentInputs):]
    UpdatedPromptsRaw=[UpdatedPrompts[i].raw for i in range(len(UpdatedPrompts))]
    update_log("- Generating New Gene Outputs.")
    GeneOutputs=await Agents.gather_responses(UpdatedPromptsRaw)
    CrossoverGeneOutputs=GeneOutputs[:len(AgentInputs)]
    MutationGeneOutputs=GeneOutputs[len(AgentInputs):]

    CrossoverGenes=[]
    MutationGenes=[]
    for i in range(len(AgentInputs)):
        NewGene=PromptGene(CrossoverPrompts[i].raw,CrossoverCritiques[i].pydantic.SpecializedGuidance,CrossoverCritiques[i].pydantic.SpecializedCritique,CrossoverRefinedPrompts[i].raw,CrossoverGeneOutputs[i])
        CrossoverGenes.append(NewGene)
    for i in range(len(AgentInputs)):
        NewGene=PromptGene(MutationPrompts[i].raw,MutationCritiques[i].pydantic.SpecializedGuidance,MutationCritiques[i].pydantic.SpecializedCritique,MutationRefinedPrompts[i].raw,MutationGeneOutputs[i])
        MutationGenes.append(NewGene)
    PromptGeneralEvaluationCrewCopies1=[PromptGeneralEvaluationCrew.copy() for _ in range(len(AgentInputs))]
    PromptGeneralEvaluationCrewCopies2=[PromptGeneralEvaluationCrew.copy() for _ in range(len(AgentInputs))]
    PromptReasoningEvaluationCrewCopies1=[PromptReasoningEvaluationCrew.copy() for _ in range(len(AgentInputs))]
    PromptReasoningEvaluationCrewCopies2=[PromptReasoningEvaluationCrew.copy() for _ in range(len(AgentInputs))]
    update_log("- Evaluating Numerical parameters of All Genes.")
    await cleanup_asyncio()

    gen_eval_tasks=[]
    res_eval_tasks=[]
    for i in range(len(AgentInputs)):
        task=PromptGeneralEvaluationCrewCopies1[i].kickoff_async(inputs=CrossoverGenes[i].get())
        gen_eval_tasks.append(task)
        task=PromptReasoningEvaluationCrewCopies1[i].kickoff_async(inputs=CrossoverGenes[i].get())
        res_eval_tasks.append(task)
        
    AllTaskCross=gen_eval_tasks+res_eval_tasks

    gen_eval_tasks=[]
    res_eval_tasks=[]
    for i in range(len(AgentInputs)):
        task=PromptGeneralEvaluationCrewCopies2[i].kickoff_async(inputs=MutationGenes[i].get())
        gen_eval_tasks.append(task)
        task=PromptReasoningEvaluationCrewCopies2[i].kickoff_async(inputs=MutationGenes[i].get())
        res_eval_tasks.append(task)

    AllTaskMut=gen_eval_tasks+res_eval_tasks
    AllTasks=AllTaskCross+AllTaskMut
    Results=await asyncio.gather(*AllTasks)
    CrossResults=Results[:len(AllTaskCross)]
    MutResults=Results[len(AllTaskCross):]


    CrossoverGeneralScores=CrossResults[:len(AgentInputs)]
    CrossoverReasoningScores=CrossResults[len(AgentInputs):]

    MutationGeneralScores=MutResults[:len(AgentInputs)]
    MutationReasoningScores=MutResults[len(AgentInputs):]
    update_log("- Evaluating Critique of New Gene.")

    PromptCritiqueEvaluationCrewCopies1=[PromptCritiqueEvaluationCrew.copy() for _ in range(len(AgentInputs))]
    PromptCritiqueEvaluationCrewCopies2=[PromptCritiqueEvaluationCrew.copy() for _ in range(len(AgentInputs))]

    CritiqueEvalTasks=[]
    for i in range(len(AgentInputs)):
        inputs=CrossoverGenes[i].get()
        task=PromptCritiqueEvaluationCrewCopies1[i].kickoff_async(inputs={
            'InitialTask': inputs['InitialTask'],
            'SpecializedGuidance': inputs['SpecializedGuidance'],
            'RefinedTask': inputs['RefinedTask'],
            'RefinedTaskOutput': inputs['RefinedTaskOutput'],
            'GeneralEvaluation': CrossoverGeneralScores[i].raw,
            'ReasoningEvaluation': CrossoverReasoningScores[i].raw
        })
        CritiqueEvalTasks.append(task)
    for i in range(len(AgentInputs)):
        inputs=MutationGenes[i].get()
        task=PromptCritiqueEvaluationCrewCopies2[i].kickoff_async(inputs={
            'InitialTask': inputs['InitialTask'],
            'SpecializedGuidance': inputs['SpecializedGuidance'],
            'RefinedTask': inputs['RefinedTask'],
            'RefinedTaskOutput': inputs['RefinedTaskOutput'],
            'GeneralEvaluation': MutationGeneralScores[i].raw,
            'ReasoningEvaluation': MutationReasoningScores[i].raw
        })
        CritiqueEvalTasks.append(task)

    CritiqueEvals=await asyncio.gather(*CritiqueEvalTasks)
    update_log("- New Gene Generation Complete.")

    CrossoverCritiqueEvals=CritiqueEvals[:len(AgentInputs)]
    MutationCritiqueEvals=CritiqueEvals[len(AgentInputs):]


    for i in range(len(AgentInputs)):
        CrossoverGenes[i].GeneralScore=CrossoverGeneralScores[i].pydantic.ScoreOutof90
        CrossoverGenes[i].ReasoningScore=CrossoverReasoningScores[i].pydantic.ScoreOutof70
        CrossoverGenes[i].RefinedPromptCritique=CrossoverCritiqueEvals[i].pydantic.RefinedPromptCritique
        CrossoverGenes[i].ActionSteps=CrossoverCritiqueEvals[i].pydantic.ActionableSteps
        MutationGenes[i].GeneralScore=MutationGeneralScores[i].pydantic.ScoreOutof90
        MutationGenes[i].ReasoningScore=MutationReasoningScores[i].pydantic.ScoreOutof70
        MutationGenes[i].RefinedPromptCritique=MutationCritiqueEvals[i].pydantic.RefinedPromptCritique
        MutationGenes[i].ActionSteps=MutationCritiqueEvals[i].pydantic.ActionableSteps
        
    return CrossoverGenes,MutationGenes





# Run a genetic algorithm for a given number of generations, refining prompt populations.
async def run_genetic_algorithm_async(
    InitialPopList, MaxGenerations, SelectionSize, TopCarry, TopCarryRate, LayerAgentInputs
):
    await cleanup_asyncio()
    update_log("- Initializing Genetic Algorithm.")

    update_log("--- Initial Population ---")
    for i, population in enumerate(InitialPopList):
        update_log(f"- Agent {LayerAgentInputs[i]['AgentName']} Population: ")
        update_log(f"  - Population Size: {len(population)}")
        InitialPopList[i] = ArrangePopulation(population)
        update_log(f"  - Population scores (Sorted): {[gene.GeneralScore+gene.ReasoningScore for gene in InitialPopList[i]]}")
        PopList = copy.deepcopy(InitialPopList)
    for generation in range(1, MaxGenerations + 1):
        await cleanup_asyncio()
        update_log(f"############################")
        update_log(f"Generation {generation}")
        update_log(f"############################")

        new_population_list = [[] for _ in range(len(LayerAgentInputs))]

        update_log(f"- Performing crossover and mutation.")
        while True:
            parent_1 = []
            parent_2 = []
            parent_3 = []
            for i in range(len(PopList)):
                p1, p2 = random.sample(PopList[i], 2)
                parent_1.append(p1)
                parent_2.append(p2)
                parent_3.append(PopList[i][0])

            p1_scores=[gene.GeneralScore+gene.ReasoningScore for gene in parent_1]
            p2_scores=[gene.GeneralScore+gene.ReasoningScore for gene in parent_2]
            p3_scores=[gene.GeneralScore+gene.ReasoningScore for gene in parent_3]

            update_log(f"  - Selected Parent scores for Crossover: Parent_1 = {p1_scores}, Parent_2 = {p2_scores}")
            update_log(f"  - Selected Parent scores for Mutation: Parent = {p3_scores}")

            crossed_genes, mutated_genes = await CrossoverMutatePrompts(
                parent_1, parent_2, parent_3, LayerAgentInputs
            )
            crossover_scores=[gene.GeneralScore+gene.ReasoningScore for gene in crossed_genes]
            mutation_scores=[gene.GeneralScore+gene.ReasoningScore for gene in mutated_genes]
            update_log(f"  - New Crossover Gene Scores: {crossover_scores}")
            update_log(f"  - New Mutated Gene Scores: {mutation_scores}")

            for i in range(len(PopList)):
                new_population_list[i].append(crossed_genes[i])
                new_population_list[i].append(mutated_genes[i])

            if len(new_population_list[0]) >= SelectionSize:
                break

        update_log(f"- Trimming the Top {TopCarry} individuals.")
        for i in range(len(PopList)):
            top_carries = PopList[i][:TopCarry]
            top_carries_scores=[gene.GeneralScore+gene.ReasoningScore for gene in top_carries]
            update_log(f"  - Top {TopCarry} Carry Individual Scores: {top_carries_scores}")
            for gene in top_carries:
                if random.random() < TopCarryRate:
                    new_population_list[i].append(gene)
                    update_log(f"  - Adding Top Carry Individual: {gene.GeneralScore+gene.ReasoningScore}")
                else:
                    update_log(f"  - Skipping Top Carry Individual: {gene.GeneralScore+gene.ReasoningScore}")
        new_population_sizes=[len(pop) for pop in new_population_list]
        update_log(f"- New Population sizes after adding Top Carry: {new_population_sizes}")
        PopList = new_population_list

        update_log(f"- Sorting and trimming the population.")
        for i in range(len(PopList)):
            update_log(f"- Agent {LayerAgentInputs[i]['AgentName']} Populations: ")
            update_log(f"  - Population Size: {len(PopList[i])}")
            PopList[i] = ArrangePopulation(PopList[i])
            update_log(f"  - Population scores (Sorted): {[gene.GeneralScore+gene.ReasoningScore for gene in PopList[i]]}")
            if len(PopList[i]) > SelectionSize:
                PopList[i] = PopList[i][:SelectionSize]
            pop_scores=[gene.GeneralScore+gene.ReasoningScore for gene in PopList[i]]
            update_log(f"  - Population scores at end of generation {generation}: {pop_scores}")
            update_log(f"  - Population Size: {len(PopList[i])}")

    update_log("############################")
    update_log("# Final Generation Complete")
    update_log("############################")
    for i in range(len(PopList)):
        update_log(f"- Agent {LayerAgentInputs[i]['AgentName']} Final Populations: ")
        PopList[i] = ArrangePopulation(PopList[i])
        update_log(f"  - Final Population Size: {len(PopList[i])}")
        update_log(f"  - Final Population scores (Sorted): {[gene.GeneralScore+gene.ReasoningScore for gene in PopList[i]]}")
    update_log("############################")
    update_log("- Genetic Algorithm Complete")
    return PopList



# Sort a population based on combined general and reasoning scores.
def ArrangePopulation(Population):
    Population.sort(key=lambda obj: obj.GeneralScore+obj.ReasoningScore, reverse=True)
    return Population    
    
# Generate an initial population of prompts asynchronously for later refinement.
async def generate_population_async(initial_prompt, agent_inputs , population_size):    
    await cleanup_asyncio()
    update_log("- Initializing Population Generation.")
    InputData=[]
    PromptGenerators=[]
    for agent_input in agent_inputs:
        for i in range(population_size):
            InputData.append({
                'InitialTask':initial_prompt,
                'AgentName':agent_input['AgentName'],
                'AgentBackstory':agent_input['AgentBackstory'],
                'AgentGoal':agent_input['AgentGoal'],
                'AgentTask':agent_input['AgentTask']
            })
        PromptGenerator=Agents.GetCrewWithAgent(
            Agent=get_agent(agent_input),
            TaskYamlFile= os.path.join(base_dir, "definitions", "TaskDef", "SpecializedPromptGeneration.yaml"),
            OutputFile= os.path.join(base_dir, "logs", "SpecializedPromptLogs.txt"),
            Async=True,
        )
        PromptGenerators.append(PromptGenerator)
    update_log("- Generating Initial Population Critiques.")
    UpdatedCritiques = await InitialCritiqueCrew.kickoff_for_each_async(inputs=InputData)


    InputData=[]
    for i in range(len(UpdatedCritiques)):
        InputData.append({
            'InitialTask':initial_prompt,
            'SpecializedGuidance':UpdatedCritiques[i].pydantic.SpecializedGuidance,
            'SpecializedCritique':UpdatedCritiques[i].pydantic.SpecializedCritique
        })
    update_log("- Generating Initial Population Refined Prompts.")
    PromptGeneratorsCopies=[]
    for i in range(len(agent_inputs)):
        for j in range(population_size):
            PromptGeneratorsCopies.append(PromptGenerators[i].copy())
    
    prompt_gen_tasks=[]
    for i in range(len(InputData)):
        task=PromptGeneratorsCopies[i].kickoff_async(inputs=InputData[i])
        prompt_gen_tasks.append(task)

    UpdatedPrompts=await asyncio.gather(*prompt_gen_tasks)       

    UpdatedRawPrompts=[UpdatedPrompts[i].raw for i in range(len(UpdatedCritiques))]
    GeneOutputs=await Agents.gather_responses(UpdatedRawPrompts)

    NewGenes=[]
    for i in range(len(UpdatedCritiques)):
        NewGene=PromptGene(initial_prompt,UpdatedCritiques[i].pydantic.SpecializedGuidance,UpdatedCritiques[i].pydantic.SpecializedCritique,UpdatedPrompts[i].raw,GeneOutputs[i])
        NewGenes.append(NewGene)
    
    InputData=[]
    for i in range(len(NewGenes)):
        InputData.append(NewGenes[i].get())
    
    update_log("- Initializing Initial Population Numerical Evaluation.")
    PromptGeneralEvaluationCrewCopies=[PromptGeneralEvaluationCrew.copy() for _ in range(len(InputData))]
    PromptReasoningEvaluationCrewCopies=[PromptReasoningEvaluationCrew.copy() for _ in range(len(InputData))]
    tasks=[]
    for i in range(len(InputData)):
        task=PromptGeneralEvaluationCrewCopies[i].kickoff_async(inputs=InputData[i])
        tasks.append(task)
    for i in range(len(InputData)):
        task=PromptReasoningEvaluationCrewCopies[i].kickoff_async(inputs=InputData[i])
        tasks.append(task)
    Results=await asyncio.gather(*tasks)
    GeneralScores=Results[:len(InputData)]
    ReasoningScores=Results[len(InputData):]

    InputData=[]
    for i in range(len(NewGenes)):
        InputData.append({
            'InitialTask': NewGenes[i].InitialPrompt,
            'SpecializedGuidance': NewGenes[i].AgentGuidance,
            'RefinedTask': NewGenes[i].RefinedPrompt,
            'RefinedTaskOutput': NewGenes[i].Result,
            'GeneralEvaluation': GeneralScores[i].raw,
            'ReasoningEvaluation': ReasoningScores[i].raw
        })

    update_log("- Initializing Initial Population Critique Evaluation.")
    CritiqueEvals=  await PromptCritiqueEvaluationCrew.kickoff_for_each_async(inputs=InputData)

    for i in range(len(NewGenes)):
        NewGenes[i].GeneralScore=GeneralScores[i].pydantic.ScoreOutof90
        NewGenes[i].ReasoningScore=ReasoningScores[i].pydantic.ScoreOutof70
        NewGenes[i].RefinedPromptCritique=CritiqueEvals[i].pydantic.RefinedPromptCritique
        NewGenes[i].ActionSteps=CritiqueEvals[i].pydantic.ActionableSteps

    Genes=[]
    for i in range(len(agent_inputs)):
        AgentGenes=[]
        for y in range(population_size):
            AgentGenes.append(NewGenes[i*population_size+y])
        Genes.append(AgentGenes)

    update_log("- Population Generation Complete.")
    return Genes


# Update the log with new information about processing steps.
def update_log(new_entry):
    log_list.append(new_entry)
    log_area.text("\n".join(log_list))

# Asynchronously optimize a provided prompt using verbose, fast, and dynamic modes.
async def OptimizePrompt_async(LaymanPrompt, Persona, Constraints, api_key, verbose,dynamic_mode):
    os.environ["OPENAI_API_KEY"]=api_key
    prepare_agents(verbose)
    global log_list
    log_list = []
    global log_section
    log_section = st.expander("System Logs", expanded=False)
    global log_area 
    log_area = log_section.empty()
    Validity,Reason= Agents.ValidityCheck(
        LaymanPrompt, 
        Persona, 
        Constraints,    
        os.path.join(base_dir, "definitions", "TaskDef", "SystemGuardrail.yaml") , 
        ValidityCheck,
        )
    if Validity==False:
        update_log("Prompt Rejected By ANT_AI")
        st.error("Prompt Rejected")
        return "",Validity
    
    try:
        with open(os.path.join(base_dir, "src", "ant_ai", "StreamlitConfig.yaml"), "r", encoding="utf-8") as yaml_file:
            agent_yaml = yaml.safe_load(yaml_file)

        validated_config = validate_config(agent_yaml, default_config)

        update_log("Gentic Algorithm Configuration: " + str(validated_config) + "\n")

    except FileNotFoundError:
        update_log("Configuration file not found. Using default configuration.")
        validated_config = default_config
    except yaml.YAMLError as e:
        update_log("Error while parsing the YAML configuration file:"+ e)
        update_log("Using default configuration.")
        validated_config = default_config
        
    update_log("*** Prompt Optimization Initiated ***\n\n")
    update_log("- Initializing Initial Prompt Generation")
    PromptCrew = InitialPromptCrew.kickoff(inputs={"LaymanPrompt": LaymanPrompt, "Persona": Persona, "Constraints": Constraints})
    InitialPrompt = PromptCrew.raw
    AgentCount = 3
    if dynamic_mode:
        update_log(f"- Initializing Dynamic Agent Creation with {AgentCount} Agents")
        AgentsListRaw = DynamicAgentCreationCrew.kickoff(inputs={"InitialTask": InitialPrompt, "AgentCount": str(AgentCount)})
        AgentInputs = [
            {
                "AgentName": agent.agent_name,
                'AgentBackstory': agent.agent_backstory,
                'AgentGoal': agent.agent_goal,
                'AgentTask': agent.agent_task,
                "InitialTask": InitialPrompt
            }
            for agent in AgentsListRaw.pydantic.agents
        ]
        update_log("- Agents Generated:")
    else:
        #Outline, Research, Tone Agent
        update_log(f"- Initializing Static Agent Creation with {AgentCount} Agents")
        AgentInputs = [
            {
                "AgentName": "Outline Agent",
                "AgentBackstory": "You are a skilled technical writer with vast experience in creating clear and structured documentation. Your attention to detail ensures that all relevant aspects of a task are captured in an orderly manner, reducing ambiguity and enhancing consistency for any kind of task.",
                "AgentGoal": "As an Outline Agent, your goal is to enhance the structure and clarity of <ant-task> by organizing it into a logical and concise framework, ensuring the intelligent assistant can easily interpret and respond to it.",
                "AgentTask": "Your task is to systematically break down <ant-task> into a clear outline, identifying key components and arranging them logically. Utilize structure to make <ant-task> understandable and actionable, while applying a chain of thought process to ensure completeness and coherence.",
                "InitialTask": InitialPrompt
            },
            {
                "AgentName": "Research Agent",
                "AgentBackstory": "You are a seasoned researcher with expertise in gathering and synthesizing relevant information from various sources. Your skill lies in identifying crucial context and deeper insights that can be integrated into task definitions to make them robust and comprehensive.",
                "AgentGoal": "As a Research Agent, your goal is to provide comprehensive background and context for <ant-task>, enhancing its depth and breadth to ensure a well-informed approach by the intelligent assistant.",
                "AgentTask": "Conduct thorough research to gather relevant information that supports and enriches <ant-task>. Incorporate validated data, relevant context, and accurate references into the task definition, using a chain of thought to link insights and enhance its informative quality.",
                "InitialTask": InitialPrompt
            },
            {
                "AgentName": "Tone Agent",
                "AgentBackstory": "You are a communication specialist with a keen understanding of language nuances and style adaptation. Your expertise allows you to tailor tasks into precise and consistent messages, making them suitable for the intended audience and context.",
                "AgentGoal": "As a Tone Agent, your goal is to ensure that the language and style of <ant-task> are appropriate and consistent with its purposes, facilitating effective communication with the intelligent assistant.",
                "AgentTask": "Review and adjust the language and style used in <ant-task> to ensure it conveys the intended tone and professionalism. Apply a chain of thought that considers purpose, audience, and context to refine the written expression for maximum clarity and impact.",
                "InitialTask": InitialPrompt
            }
        ]
        update_log("- Agents Generated:")
    for agent in AgentInputs:
        update_log(f"  - Agent Name: {agent['AgentName']}")
        


    PopulationList = await generate_population_async(
        initial_prompt=InitialPrompt,
        agent_inputs=AgentInputs,
        population_size=3
    )

    
    SpecializedPopulationList = await run_genetic_algorithm_async(
        InitialPopList=PopulationList,
        MaxGenerations=validated_config["MaxGenerations"],
        SelectionSize=validated_config["SelectionSize"],
        TopCarry=validated_config["TopCarry"],
        TopCarryRate=validated_config["TopCarryRate"],
        LayerAgentInputs=AgentInputs
    )
    
    update_log("- Finalizing Prompt Unification")
    ConcatenatedText = ""
    for i in range(len(SpecializedPopulationList)):
        ConcatenatedText += f"<Employee_{i+1}_Details>\n"
        ConcatenatedText += f"Employee Role : {AgentInputs[i]['AgentName']}\n"
        ConcatenatedText += f"Employee Goal : {AgentInputs[i]['AgentGoal']}\n"
        ConcatenatedText += f"Employee Backstory : {AgentInputs[i]['AgentBackstory']}\n"
        BestGene = SpecializedPopulationList[i][0]
        ConcatenatedText += f"Employee's Task Definition : \n{BestGene.RefinedPrompt}\n"
        ConcatenatedText += f"</Employee_{i+1}_Details>\n"
    
    FinalPrompt = await FinalUnificationCrew.kickoff_async(inputs={"AntTaskList": ConcatenatedText})
    update_log(f"*** Final Prompt Generated ***\n\n")
    return FinalPrompt.raw,Validity




# Quickly generate a population of dynamic prompts asynchronously without running GA.
async def generate_population_async_fast(initial_prompt, agent_inputs):    
    await cleanup_asyncio()
    update_log("- Initializing Dynamic Prompt Generation.")
    InputData=[]
    PromptGenerators=[]
    for agent_input in agent_inputs:
        InputData.append({
            'InitialTask':initial_prompt,
            'AgentName':agent_input['AgentName'],
            'AgentBackstory':agent_input['AgentBackstory'],
            'AgentGoal':agent_input['AgentGoal'],
            'AgentTask':agent_input['AgentTask']
        })
        PromptGenerator=Agents.GetCrewWithAgent(
            Agent=get_agent(agent_input),
            TaskYamlFile= os.path.join(base_dir, "definitions", "TaskDef", "SpecializedPromptGeneration.yaml"),
            OutputFile= os.path.join(base_dir, "logs", "SpecializedPromptLogs.txt"),
            Async=True,
        )
        PromptGenerators.append(PromptGenerator)
    update_log("- Generating Dynamic Prompt Critiques.")
    UpdatedCritiques = await InitialCritiqueCrew.kickoff_for_each_async(inputs=InputData)


    InputData=[]
    for i in range(len(UpdatedCritiques)):
        InputData.append({
            'InitialTask':initial_prompt,
            'SpecializedGuidance':UpdatedCritiques[i].pydantic.SpecializedGuidance,
            'SpecializedCritique':UpdatedCritiques[i].pydantic.SpecializedCritique
        })
    update_log("- Generating Refined Prompt Versions.")
    
    prompt_gen_tasks=[]
    for i in range(len(InputData)):
        task=PromptGenerators[i].kickoff_async(inputs=InputData[i])
        prompt_gen_tasks.append(task)

    UpdatedPrompts=await asyncio.gather(*prompt_gen_tasks)       

    UpdatedRawPrompts=[UpdatedPrompts[i].raw for i in range(len(UpdatedCritiques))]
    GeneOutputs=await Agents.gather_responses(UpdatedRawPrompts)

    NewGenes=[]
    for i in range(len(UpdatedCritiques)):
        NewGene=PromptGene(initial_prompt,UpdatedCritiques[i].pydantic.SpecializedGuidance,UpdatedCritiques[i].pydantic.SpecializedCritique,UpdatedPrompts[i].raw,GeneOutputs[i])
        NewGenes.append(NewGene)
    
    update_log("- Population Generation Complete.")
    return NewGenes






# Execute a fast optimization pass on a provided prompt using available modes.
async def OptimizePrompt_async_fast(LaymanPrompt, Persona, Constraints, api_key, verbose, dynamic_mode):
    os.environ["OPENAI_API_KEY"]=api_key
    prepare_agents(verbose)
    global log_list
    log_list = []
    global log_section
    log_section = st.expander("System Logs", expanded=False)
    global log_area 
    log_area = log_section.empty()
    Validity,Reason= Agents.ValidityCheck(
        LaymanPrompt, 
        Persona, 
        Constraints,    
        os.path.join(base_dir, "definitions", "TaskDef", "SystemGuardrail.yaml") , 
        ValidityCheck,
        )
    if Validity==False:
        update_log("Prompt Rejected By ANT_AI")
        st.error("Prompt Rejected")
        return "",Validity
    
        
    update_log("*** Prompt Optimization Initiated ***\n\n")
    update_log("- Initializing Initial Prompt Generation")
    PromptCrew = InitialPromptCrew.kickoff(inputs={"LaymanPrompt": LaymanPrompt, "Persona": Persona, "Constraints": Constraints})
    InitialPrompt = PromptCrew.raw
    AgentCount = 3
    if dynamic_mode:
        update_log(f"- Initializing Dynamic Agent Creation with {AgentCount} Agents")
        AgentsListRaw = DynamicAgentCreationCrew.kickoff(inputs={"InitialTask": InitialPrompt, "AgentCount": str(AgentCount)})
        AgentInputs = [
            {
                "AgentName": agent.agent_name,
                'AgentBackstory': agent.agent_backstory,
                'AgentGoal': agent.agent_goal,
                'AgentTask': agent.agent_task,
                "InitialTask": InitialPrompt
            }
            for agent in AgentsListRaw.pydantic.agents
        ]
        update_log("- Agents Generated:")
    else:
        #Outline, Research, Tone Agent
        update_log(f"- Initializing Static Agent Creation with {AgentCount} Agents")
        AgentInputs = [
            {
                "AgentName": "Outline Agent",
                "AgentBackstory": "You are a skilled technical writer with vast experience in creating clear and structured documentation. Your attention to detail ensures that all relevant aspects of a task are captured in an orderly manner, reducing ambiguity and enhancing consistency for any kind of task.",
                "AgentGoal": "As an Outline Agent, your goal is to enhance the structure and clarity of <ant-task> by organizing it into a logical and concise framework, ensuring the intelligent assistant can easily interpret and respond to it.",
                "AgentTask": "Your task is to systematically break down <ant-task> into a clear outline, identifying key components and arranging them logically. Utilize structure to make <ant-task> understandable and actionable, while applying a chain of thought process to ensure completeness and coherence.",
                "InitialTask": InitialPrompt
            },
            {
                "AgentName": "Research Agent",
                "AgentBackstory": "You are a seasoned researcher with expertise in gathering and synthesizing relevant information from various sources. Your skill lies in identifying crucial context and deeper insights that can be integrated into task definitions to make them robust and comprehensive.",
                "AgentGoal": "As a Research Agent, your goal is to provide comprehensive background and context for <ant-task>, enhancing its depth and breadth to ensure a well-informed approach by the intelligent assistant.",
                "AgentTask": "Conduct thorough research to gather relevant information that supports and enriches <ant-task>. Incorporate validated data, relevant context, and accurate references into the task definition, using a chain of thought to link insights and enhance its informative quality.",
                "InitialTask": InitialPrompt
            },
            {
                "AgentName": "Tone Agent",
                "AgentBackstory": "You are a communication specialist with a keen understanding of language nuances and style adaptation. Your expertise allows you to tailor tasks into precise and consistent messages, making them suitable for the intended audience and context.",
                "AgentGoal": "As a Tone Agent, your goal is to ensure that the language and style of <ant-task> are appropriate and consistent with its purposes, facilitating effective communication with the intelligent assistant.",
                "AgentTask": "Review and adjust the language and style used in <ant-task> to ensure it conveys the intended tone and professionalism. Apply a chain of thought that considers purpose, audience, and context to refine the written expression for maximum clarity and impact.",
                "InitialTask": InitialPrompt
            }
        ]
        update_log("- Agents Generated:")
    for agent in AgentInputs:
        update_log(f"  - Agent Name: {agent['AgentName']}")
        


    PopulationList = await generate_population_async_fast(
        initial_prompt=InitialPrompt,
        agent_inputs=AgentInputs
    )

    update_log("- Finalizing Prompt Unification")
    ConcatenatedText = ""
    for i in range(len(PopulationList)):
        ConcatenatedText += f"<Employee_{i+1}_Details>\n"
        ConcatenatedText += f"Employee Role : {AgentInputs[i]['AgentName']}\n"
        ConcatenatedText += f"Employee Goal : {AgentInputs[i]['AgentGoal']}\n"
        ConcatenatedText += f"Employee Backstory : {AgentInputs[i]['AgentBackstory']}\n"
        BestGene = PopulationList[i]
        ConcatenatedText += f"Employee's Task Definition : \n{BestGene.RefinedPrompt}\n"
        ConcatenatedText += f"</Employee_{i+1}_Details>\n"
 

    
    FinalPrompt = await FinalUnificationCrew.kickoff_async(inputs={"AntTaskList": ConcatenatedText})
    update_log(f"*** Final Prompt Generated ***\n\n")
    return FinalPrompt.raw,Validity
