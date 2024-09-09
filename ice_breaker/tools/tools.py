from langchain_community.tools.tavily_search import TavilySearchResults

def get_profile_url_tavily(search_query: str):
    """Searches for Linkedin or Twitter Profile Page"""
    search = TavilySearchResults()
    res = search.run(f"{search_query}")
    print(res)
    return [res[0]["url"], res[1]["url"]]