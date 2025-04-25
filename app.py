import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils import text_to_embedding, compute_similarity
from typing import List, Dict, Any, Optional

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class UserBioRequest(BaseModel):
    user_bio: str

# Define response model
class RankedStory(BaseModel):
    title: str
    url: Optional[str] = None
    type: str
    similarity: float

class RankingResponse(BaseModel):
    stories: List[RankedStory]

# Load data at startup to avoid reloading for each request
@app.on_event("startup")
async def startup_event():
    global story_df
    try:
        story_df = pd.read_csv('embedded_hn.csv').dropna(axis=0, subset=['embedding'])
        # Convert the string representation of the list back to a list
        story_df['embedding'] = story_df['embedding'].apply(eval)
    except Exception as e:
        print(f"Error loading data: {e}")
        # Initialize with empty DataFrame as fallback
        story_df = pd.DataFrame(columns=['title', 'url', 'type', 'embedding'])

@app.post("/api/rank", response_model=RankingResponse)
async def rank_stories(request: UserBioRequest):
    try:
        # Get user embedding
        embedded_user = text_to_embedding([request.user_bio])
        
        # Compute similarity
        similarity_df = compute_similarity(embedded_user, story_df) \
            .sort_values(by='similarity', ascending=False) \
            [['title', 'url', 'type', 'similarity']] \
            .replace({np.nan: None})
        
        # Convert to records for API response
        ranked_stories = similarity_df.to_dict(orient='records')
        return {"stories": ranked_stories}
    
    except Exception as e:
        print(f"Error in rank_stories: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error ranking stories: {str(e)}")

# Add a simple health check endpoint
@app.get("/")
async def health_check():
    return {"status": "ok", "message": "HN Ranking API is running"}

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
