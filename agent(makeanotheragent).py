import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load API key
load_dotenv("openaikey.env")
os.environ["OPENAI_API_KEY"]

# === 1. Meta-Agent: Understand problem and propose an AI Agent ===

llm = ChatOpenAI(model="gpt-4", temperature=0.7)

analysis_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are an AI Agent Builder.

Your task is to read the following text, detect any inefficiencies or repetitive tasks, and suggest an AI Agent that could help.

Then, generate a LangChain-based plan (code or logic) for that Agent.

TEXT:
{text}

Response Format:
1. Detected problem:
2. Agent name:
3. Agent purpose:
4. Required tools/APIs:
5. LangChain component design (PromptChain, Tools, Memory, etc.):
6. Optional Python code:
"""
)

analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt)

input_text = """
Our team forgets to send meeting summaries and people are confused about next steps after calls.
"""

agent_description = analysis_chain.run(input_text)
print("=== Agent Description ===")
print(agent_description)

# === 2. Code Generator: Generate LangChain code for that agent ===

code_prompt = PromptTemplate(
    input_variables=["agent_description"],
    template="""w
Given this agent description:

{agent_description}

Generate Python code using LangChain that implements this Agent. Include:
- All required imports
- Prompts
- Chains or Tools
- Comments explaining each part
"""
)

code_chain = LLMChain(llm=llm, prompt=code_prompt)
generated_code = code_chain.run(agent_description)

print("\n=== LangChain Code Output ===")
print(generated_code)
