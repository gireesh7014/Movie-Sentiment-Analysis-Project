import joblib
import gensim.downloader as api
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== Preprocessing Setup ==========
logger.info("Downloading NLTK data...")
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)  # For newer NLTK versions
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
logger.info("NLTK data downloaded.")

# Initialize NLTK components after downloading
try:
    stop_words = set(stopwords.words("english"))
    lemm = WordNetLemmatizer()
    # Test the lemmatizer to ensure it's working
    _ = lemm.lemmatize("testing")
    logger.info("NLTK components initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing NLTK components: {e}")
    stop_words = set()
    lemm = None

def simple_preprocess(text: str):
    """
    Simple fallback preprocessing without NLTK dependencies.
    """
    import re
    import string
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and split into words
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    
    # Remove short words and numbers
    tokens = [t for t in tokens if len(t) > 2 and t.isalpha()]
    
    # Simple stopword removal (basic English stopwords)
    basic_stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'within', 'without', 'along', 'following', 'across', 'behind', 'beyond', 'plus', 'except', 'but', 'until', 'unless', 'since', 'while'}
    tokens = [t for t in tokens if t not in basic_stopwords]
    
    return tokens

def preprocess(text: str):
    """
    Cleans and tokenizes the input text.
    """
    try:
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t.isalpha()]
        tokens = [t for t in tokens if t not in stop_words]
        
        # Only lemmatize if lemmatizer is available
        if lemm is not None:
            try:
                tokens = [lemm.lemmatize(t) for t in tokens]
            except Exception as e:
                logger.warning(f"Lemmatization failed: {e}. Skipping lemmatization.")
                # Continue without lemmatization
        
        return tokens
    except Exception as e:
        logger.error(f"NLTK preprocessing failed: {e}. Using simple preprocessing.")
        # Use simple fallback preprocessing
        return simple_preprocess(text)

def review_to_vector(tokens, model, vector_size):
    """
    Converts a list of tokens to a single vector by averaging the word vectors.
    """
    vectors = [model[w] for w in tokens if w in model.key_to_index]
    if len(vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

# --- Global variables to hold the loaded model and embeddings ---
clf = None
keyed_vectors = None

# --- Lifespan event handler to replace the deprecated on_event ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan manager for the FastAPI app. This function runs on startup
    to load the ML model and embeddings.
    """
    global clf, keyed_vectors
    logger.info("Startup event: Loading model and embeddings...")

    model_path = "sentiment_svm_model.pkl"
    meta_path = "model_metadata.pkl"

    try:
        logger.info(f"Loading model from {model_path}")
        clf = joblib.load(model_path)
        
        logger.info(f"Loading metadata from {meta_path}")
        meta = joblib.load(meta_path)

        embedding_name = meta.get("embedding") 
        if not embedding_name:
            raise RuntimeError("Embedding name missing in metadata file.")
        
        logger.info(f"Loading word embeddings: {embedding_name}...")
        keyed_vectors = api.load(embedding_name)
        logger.info("Model and embeddings loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"Error loading model/metadata: {e}. Make sure '{model_path}' and '{meta_path}' exist.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during startup: {e}")

    yield
    
    # Code below yield runs on shutdown
    logger.info("Shutdown event: Cleaning up resources...")
    # Clear the global variables to release memory
    clf = None
    keyed_vectors = None

# ========== FastAPI App Setup ==========
app = FastAPI(title="Sentiment Prediction API", lifespan=lifespan)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# --- Pydantic Models for Request and Response ---
class ReviewRequest(BaseModel):
    text: str

class ReviewResponse(BaseModel):
    review: str
    sentiment: str

@app.post("/predict", response_model=ReviewResponse)
def predict(req: ReviewRequest):
    """
    Prediction endpoint. Takes a review text and returns the predicted sentiment.
    """
    try:
        if clf is None or keyed_vectors is None:
            raise HTTPException(status_code=503, detail="Model is not loaded yet. Please wait.")
        
        tokens = preprocess(req.text)
        
        # Convert tokens to vector
        vector = review_to_vector(tokens, keyed_vectors, keyed_vectors.vector_size)

        # Make prediction
        pred = int(clf.predict([vector])[0])
        
        sentiment = "Positive" if pred == 1 else "Negative"

        return ReviewResponse(review=req.text, sentiment=sentiment)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
def health_check():
    """
    Health check endpoint to verify that the API is running.
    """
    return {"status": "ok", "model_loaded": clf is not None and keyed_vectors is not None}

@app.get("/")
def read_root():
    return {"message": "Welcome! This is a sentiment analysis API. Please send a POST request to /predict to use it."}
