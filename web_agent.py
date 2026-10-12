import string

from langchain_classic import hub
from langchain_openai import ChatOpenAI
import requests
from dotenv import load_dotenv
from langchain_classic.agents import create_react_agent,AgentExecutor
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun


load_dotenv()

weather_api="https://api.weatherstack.com/current?access_key=e58cb64a6e894e05056d335cd1e7c02b&query="
# city='Pune'
# result=requests.get(f"{weather_api}{city}")
# print(result.json())


# step 1 : create tool
@tool
def get_weather_data(city:string):
    """given city name function return its realtime weather data"""
    response=requests.get(f"{weather_api}{city}")
    return response.json()


internet_search_tool=DuckDuckGoSearchRun()



#step 2 : create agent

llm=ChatOpenAI()

from langchain_core.prompts import PromptTemplate

template = """
Answer the following questions as best you can.

You have access to these tools:
{tools}

Use the following format:

Question: the input question
Thought: your reasoning about what to do
Action: one of [{tool_names}]
Action Input: the input to the tool
Observation: the result of the tool
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)

# prompt = hub.pull(
#     "hwchase17/react",
#     dangerously_pull_public_prompt=True
# )
# print(prompt)

agent=create_react_agent(llm=llm,tools=[internet_search_tool,get_weather_data],prompt=prompt)

agent_executor=AgentExecutor(agent=agent,tools=[internet_search_tool,get_weather_data],verbose=True)

# response=agent_executor.invoke({"input": "Find the capital of Madhya Pradesh, then find it's current weather condition"})
response=agent_executor.invoke({"input": "who is prime minister of india"})

print(response)