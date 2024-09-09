from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

from output_parsers import summary_parser
from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from agents.twitter_lookup_agent import lookup as twitter_lookup_agent
from third_parties.twitter import scrape_user_tweets


def ice_break_with(search_query: str) -> str:
    linkedin_url = linkedin_lookup_agent(name=search_query)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_url)

    twitter_username = twitter_lookup_agent(name = search_query)
    twitter_data = scrape_user_tweets(username = twitter_username)

    summary_template = """
            given the linkedin information {linkedin_information} about a person,
            and twitter posts {twitter_posts}, I want you to create:
            1. a short summary
            2. two interesting facts about them
            
            use information from both, linkedin and twitter
            \n{format_instructions}
        """

    summary_prompt_template = PromptTemplate(
        input_variables = ["linkedin_information", "twitter_posts"],
        template = summary_template,
        partial_variables = {"format_instructions": summary_parser.get_format_instructions()}
    )

    # llm = ChatOllama(model="llama3")
    llm = ChatOpenAI(temperature = 0, model_name = "gpt-4o-mini")
    chain = summary_prompt_template | llm | summary_parser
    res = chain.invoke(input={"linkedin_information": linkedin_data, "twitter_posts": twitter_data})
    print(res)
    return res

if __name__ == "__main__":
    load_dotenv()
    print("Ice Breaker Enter")
    res = ice_break_with(search_query="Eden Marco")