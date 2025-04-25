from utils import get_hacker_news_stories, text_to_embedding

# fetch latest Hacker News stories
print('Fetching latest Hacker News stories...')
story_df = get_hacker_news_stories(limit=500)

# Compute embeddings for the titles
print('Computing embeddings for the titles...')
embedded_titles = text_to_embedding(
    story_df.title.str.replace('"', '').str.replace("'", '').tolist()
)

# save the stories and embeddings to a CSV file
print('Saving stories and embeddings to CSV...')
story_df \
    .merge(embedded_titles, left_on='title', right_on='input', how='left') \
    .to_csv('embedded_hn.csv', index=False)

print('Cache updated successfully!')