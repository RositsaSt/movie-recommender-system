"""Start the FastAPI server via uvicorn."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("movie_recommender.api.main:app", reload=True)
