import voyageai
import os
import pandas as pd
import backoff
from dotenv import load_dotenv
import requests

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
vo = voyageai.Client(api_key=os.environ['VOYAGE_API_KEY'])

@backoff.on_exception(backoff.constant, Exception, max_tries=2, interval=120)
def voyage_embedding_create(input: list, **kwargs) -> list:
    '''Compute an embedding for a list of strings with exponential backoff'''
    embeds = vo.embed(
        input, 
        model="voyage-3-large",
        **kwargs
    ).embeddings
    return embeds

def text_to_embedding(text_list: list, **kwargs) -> pd.DataFrame:
    '''Compute an embedding for a string or list of strings using the OpenAI API'''
    try:    
        embeds = voyage_embedding_create(input=text_list, **kwargs)
        embed_df = pd.DataFrame(list(zip(text_list, embeds)), columns=['input', 'embedding'])
    except Exception as e:
        raise print(f'Error: {e} problem getting embedding. Text: {text_list}')
    return embed_df

def get_hacker_news_stories(limit=500):
    """Fetch top Hacker News stories quickly"""
    # Get top story IDs
    top_ids = requests.get("https://hacker-news.firebaseio.com/v0/topstories.json").json()[:limit]
    
    # Fetch individual stories in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        stories = list(executor.map(
            lambda id: requests.get(f"https://hacker-news.firebaseio.com/v0/item/{id}.json").json(), 
            top_ids
        ))
    
    # Filter out any None values just in case
    return pd.DataFrame([story for story in stories if story])

def compute_similarity(embedded_user, embedded_titles):
    '''Compute cosine similarity between user bio and story titles'''
    # Convert embeddings to numpy arrays
    user_embedding = np.array(embedded_user.embedding.tolist()[0])
    title_embeddings = np.array(embedded_titles.embedding.tolist())
    
    # Compute cosine similarity
    similarities = cosine_similarity(user_embedding.reshape(1, -1), title_embeddings)
    
    # Add similarity scores to the DataFrame
    embedded_titles['similarity'] = similarities[0]
    
    return embedded_titles

