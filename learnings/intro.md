# Learning

## Content

### 1. What are AI Agents?

## What are AI Agents?

In basis, an AI agent is a system that utilizes its **environment** (input text, sensors, etc) to perform **tasks**. These tasks are performed in autonomy requiring no human interaction. 

> Although, having "AI" in the name, an AI agent does not have to use any sort of generative intelligence or artifical intelligence. (e.g., a simple motion sensor light is an AI agent).

The main components of an AI Agent (*PRAO*):

- **P**erception - Must be able to observe the environment (e.g., camera, input text, sensors)
- **R**easoning - Process that information and decides what to do
- **A**ction - Execute a response or tasks (e.g., click a button, speak, move)
- **O**bjective - There exists a desired outcome

### Examples

Let's go through some examples of some AI agents:

- Motion-sensor light
  - The objective is to help light the way for people and save energy when not in use (O). It's ability to perceive is through it's motion sensor (P). When the sensor detects someone, it turns on a light (R+A).
- Chess bot (Minimax)
  - The objective is to win a game of chess using the environment which tracks the moves of the player (O+P). It processes this information and calculates the best moves through the minimax algorithm to make the next move (R+A).

### Generative Intelligence (LLM-Based)

The concept of AI agents isn't new. They've been around for decades in robots, video games, etc. The reason for it's rise in popularity, is the rise of Large Language Models (LLM) and it's utilization inside AI Agents.

#### What can this Achieve?

LLMs allows for AI agents to **comprehend more complex human language** (e.g., before: "if A, then B", now: "write a blog post, summarize an article, then email it to me"). 

LLMs can use tools such as:

- Searching the web
- Read/Write files
- Access DB, APIs and etc.,
- Build and execute code
- Perform multi-step planning (i.e., plan out the steps required to achieve an objective)

Popular tools like **LangChain, OpenAI Functions, AutoGPT, and etc.,** allows these functionalities.

In addition, LLMs can have multiple roles such as: a researcher, planner, executor, reviewer using tools like **AutoGen**.