from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_community.llms import Ollama
from langchain.agents import Tool
from langchain_community.utilities import PythonREPL


llm = Ollama(model="mistral")

## Define Tools
python_repl = PythonREPL()

tools = load_tools(["python_repl","llm-math"], llm=llm)


agent = initialize_agent(
  tools, 
  llm,
  agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
  verbose=True,
  return_intermediate_steps=True
)

response = agent("hello")
print(response)