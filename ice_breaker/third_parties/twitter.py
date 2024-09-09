import os
from dotenv import load_dotenv
import requests

load_dotenv()

def scrape_user_tweets(username, num_tweets=5, mock:bool = True):
    """
    Scrapes a Twitter user's original tweets (i.e., not retweets or replies) and returns them as a list of dictionaries.
    Each dictionary has three fields: "time_posted" (relative to now), "text", and "url".
    """
    tweet_list = []

    if mock:
        EDEN_TWITTER_GIST = "https://gist.github.com/emarco177/9d4fdd52dc432c72937c6e383dd1c7cc/raw"
        tweets = requests.get(EDEN_TWITTER_GIST, timeout=5).json()

        for tweet in tweets:
            tweet_dict = {}
            tweet_dict["text"] = tweet["text"]
            tweet_dict["url"] = f"https://twitter.come/{username}/status/{tweet['id']}"
            tweet_list.append(tweet_dict)

    return tweet_list

if __name__ == "__main__":
    tweets = scrape_user_tweets(username="EdenEmarco117")
    print(tweets)