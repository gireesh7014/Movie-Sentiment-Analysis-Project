import joblib
import gensim.downloader as api
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import numpy as np
import sys

# ========== Preprocessing Setup ==========
# Download necessary NLTK data quietly
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

stop_words = set(stopwords.words("english"))
lemm = WordNetLemmatizer()

def preprocess(text: str):
    """
    Cleans and tokenizes the input text.
    - Converts to lowercase
    - Tokenizes
    - Removes non-alphabetic characters
    - Removes stopwords
    - Lemmatizes
    """
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemm.lemmatize(t) for t in tokens]
    return tokens

def review_to_vector(tokens, model, vector_size):
    """
    Converts a list of tokens to a single vector by averaging the word vectors.
    """
    # Get vectors for words that exist in the embedding model
    vectors = [model[w] for w in tokens if w in model.key_to_index]
    
    if len(vectors) == 0:
        # If no words are in the vocabulary, return a zero vector
        return np.zeros(vector_size)
        
    # Average the vectors to get a single representative vector
    return np.mean(vectors, axis=0)

# ========== Load Model & Embeddings ==========
def load_pipeline():
    """
    Loads the trained SVM classifier and the Gensim word embeddings.
    """
    # Load classifier
    clf = joblib.load("sentiment_svm_model.pkl")
    # Load metadata (which contains the embedding name)
    metadata = joblib.load("model_metadata.pkl")
    embedding_name = metadata["embedding"]
    
    print(f"Loading embeddings: {embedding_name} ...")
    embeddings = api.load(embedding_name)
    print("Embeddings loaded.")
    
    return clf, embeddings

# ========== Predict Function ==========
def predict_sentiment(text, clf, embeddings):
    """
    Predicts the sentiment of a given text string.
    """
    tokens = preprocess(text)
    vector = review_to_vector(tokens, embeddings, embeddings.vector_size)
    
    # Scikit-learn models expect a 2D array, so we reshape the single vector
    pred = clf.predict([vector])[0] 
    
    return "Positive" if pred == 1 else "Negative"

# ========== Main Execution Block ==========
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sentiment_pipeline.py \"Your review text here\"")
        sys.exit(1)

    review_text = sys.argv[1]
    
    print("Initializing sentiment analysis pipeline...")
    clf, embeddings = load_pipeline()
    
    sentiment = predict_sentiment(review_text, clf, embeddings)
    
    print("-" * 30)
    print(f"Review: \"{review_text}\"")
    print(f"Predicted Sentiment: {sentiment}")
    print("-" * 30)
