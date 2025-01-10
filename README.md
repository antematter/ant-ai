---

# Ant-AI: Advanced Prompt Optimization Using Multi-Agent Systems and Genetic Algorithms

Ant-AI is an innovative project leveraging **Multi-Agent Systems** and **Genetic Algorithms (GAs)** to refine and optimize task prompts for Large Language Models (LLMs). Designed to iteratively enhance prompt relevance, clarity, and structure, Ant-AI dynamically evolves task definitions through intelligent agent collaboration, mutation, crossover, and systematic evaluation.

---

## **Features**
- **Dynamic Multi-Agent Collaboration**: AI agents perform specialized roles such as critique, mutation, crossover, or evaluation to refine task prompts.
- **Genetic Algorithm Integration**: Utilizes evolutionary strategies, including crossover and mutation, to explore optimal solutions iteratively.
- **Exploratory Innovation**: Encourages exploration and exploitation of reasoning pathways for prompt refinement.
- **Structured Prompt Tagging System**: Implements a tag-guided system, ensuring clarity and consistency.
- **Tailored Feedback & Evaluation**: Refines prompts using layered critique loops and reasoning-driven scoring metrics.
- **Seamless Streamlit Integration**: Visualizes the optimization process in an easy-to-use interactive environment.
- **Verbose Mode**: Provides detailed logs and outputs for more transparency during the optimization process.
- **Fast Mode**: Prioritizes efficiency, delivering quicker results by simplifying certain processes.
- **Dynamic Agent Mode**: Dynamically creates and adapts agent roles based on the prompt's needs for bespoke optimization.

---

## **Prerequisites**
Before running this project, ensure the following are installed on your system:
- **Python 3.11+**
- **Poetry** (for dependency and environment management)
- **Git**

---

## **Installation & Setup**
Follow the steps below to clone the repository, set up the environment, and run the Ant-AI application.

### 1. **Clone the Repository**
Clone the project repository to your local machine using Git:
```bash
git clone https://github.com/antematter/ant-ai.git
cd ant-ai
```

---

### 2. **Install Dependencies**
Ant-AI uses **Poetry** to manage dependencies:
- Install all the required dependencies by running:
  ```bash
  poetry install
  ```

---

### 3. **Activate the Environment**
Activate the Python virtual environment created by Poetry:
```bash
poetry shell
```

---

### 4. **Run the Application**
Run the Streamlit application to interact with Ant-AI:
```bash
streamlit run ./src/ant_ai/Get_Started.py
```
> **Note**: Ensure you are in the project's root folder.

---

## **Configuration**
Ant-AI allows you to customize the hyperparameters of the Genetic Algorithm using the `StreamlitConfig.yaml` file located in `src/ant_ai/`.

### `StreamlitConfig.yaml` File:
This file lets you configure essential parameters for the Genetic Algorithm's operation:
- `MaxGenerations`: Number of generations for the GA to run. Default is `3`.
- `SelectionSize`: Number of genes to produce in one generation. Default is `2`.
- `TopCarry`: Number of genes to consider for carry-over from the last generation. Default is `2`.
- `TopCarryRate`: Probability of each top-carry gene being added to the new population. Default is `0.5`.

Adjust these parameters to fine-tune the optimization process according to your needs.

---

## **How to Use Ant-AI**
1. On the Streamlit interface: Enter a **Prompt**, **Persona**, and **Constraints**.
2. Select your desired modes:
   - **Verbose** for detailed logs.
   - **Fast Mode** for quicker processing.
   - **Dynamic Agent Mode** for adaptive agent generation.
3. The system will:
   - Validate the input for safety.
   - Dynamically create agent teams to refine and evaluate the prompt.
   - Use **Genetic Algorithms (GA)** for iterations:
     - **Crossover**: Merge strengths of top-performing prompts.
     - **Mutation**: Introduce randomness for exploration.
   - Provide an optimized task prompt using structured reasoning and refinement cycles.
4. Use the "Get Started" page to navigate through options and explore the additional functionalities.
5. View results, metrics, and further details on the interface.

---

## **System Highlights & Workflow**
1. **Dynamic Multi-Agent System**:
   - Agents dynamically form teams to execute specialized tasks like mutation, crossover, and iterative evaluation.
2. **Genetic Algorithm Core**:
   - Leverages evolutionary strategies like **crossover**, **mutation**, and **elitism** to optimize prompts.
3. **Feedback Loops**:
   - Iterative refinement occurs through continuous feedback from Evaluation Agents.
4. **Structured Prompt Tagging**:
   - Tags like `<ant-task>` and `<ant-constraints>` are used to ensure clarity in task definitions.

---

## **Project Architecture**
The Ant-AI system operates through the following key components:
1. **Dynamic Agent Manager**:
   - Dynamically creates multiple agents (e.g., Mutation Agent, Crossover Agent) for specific tasks.
2. **Prompt Refinement Workflow**:
   - Agents utilize **specialized critiques**, **genetic evolution**, and structured **tagging systems** to optimize initial prompts.
3. **Evaluation Layers**:
   - Prompts are rigorously scored via **General Evaluation** (static metrics) and **Reasoning Evaluation** (adaptive creativity metrics).
4. **Streamlit Web Interface**:
   - Allows users to enter and interact with inputs while visualizing the refinement process.

**Core Files & Structure**:
```
ant-ai/
├── definitions/                           # Definitions for agents and tasks
│   ├── AgentDef/                          # YAML definitions for agents
│   └── TaskDef/                           # YAML definitions for tasks
├── logs/                                  # Logs for agent processes and workflow execution
├── src/                                   # Core Python project codebase
│   ├── ant_ai/                            # Main module for Ant-AI
│   │   ├── Agents.py                      # Agent management and dynamic generation
│   │   ├── Logger.py                      # Logging utilities
│   │   ├── Get_Started.py                 # Streamlit main entry point
│   │   ├── pages/                         # Additional Streamlit pages
│   │   │   └── 🤖_Prompt_Optimization.py  # Prompt optimization page
│   │   ├── StreamlitConfig.yaml           # Configuration for GA parameters
│   │   └── lib.py                         # Generic utility functions
├── tests/                                 # Tests for validating functionality
├── LICENSE                                # Licensing information
├── poetry.lock                            # Locked dependencies for stable environments
├── pyproject.toml                         # Poetry dependency and project configuration
└── README.md                              # Project documentation (current file)
```

---

## **License**
This project is licensed under the **Apache License 2.0**.  
You may use, modify, and distribute this project under the terms defined in the [LICENSE](LICENSE) file.

For more details about the license, visit the official [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) page.

---

## **Badges**
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-brightgreen)
