from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_core.tools import Tool
from langchain import hub

from tools.tools import get_profile_url_tavily

load_dotenv()

def lookup(name: str) -> str:
    llm = ChatOpenAI(temperature = 0, model_name = "gpt-4o-mini")
    template = """
    given the name {name_of_person} i want you to find a link to their Twitter profile page, and extract from it their username.
    In your final answer include only the person's username
    """

    prompt_template = PromptTemplate(template = template, input_variables=["name_of_person"])

    tools_for_agent = [
        Tool(
            name = "crawl google for twitter handle",
            func = get_profile_url_tavily,
            description = "useful for when you need to get the twitter profile handle"
        )]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm = llm, tools = tools_for_agent, prompt = react_prompt)
    agent_executor = AgentExecutor(agent = agent, tools = tools_for_agent, verbose = True)

    result = agent_executor.invoke(
        input = {"input": prompt_template.format_prompt(name_of_person = name)}
    )

    return result["output"]

if __name__ == "__main__":
    print(lookup(name="Eden Marco"))