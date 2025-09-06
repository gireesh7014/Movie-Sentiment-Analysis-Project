# Movie Sentiment Analyzer

A web application that analyzes the sentiment of movie reviews using machine learning. The app fetches movie reviews from The Movie Database (TMDb) API and uses a trained Support Vector Machine (SVM) model to classify sentiments as positive or negative.

## Features

- 🎬 Search for movies by title
- 📊 Fetch real movie reviews from TMDb API
- 🤖 AI-powered sentiment analysis using SVM and word embeddings
- 📈 Visual sentiment summary with positive/negative counts
- 🎨 Modern, responsive web interface
- ⚡ FastAPI backend for real-time predictions

## Tech Stack

### Frontend
- HTML5, CSS3, JavaScript
- Tailwind CSS for styling
- TMDb API for movie data

### Backend
- Python 3.8+
- FastAPI for API endpoints
- scikit-learn for machine learning
- NLTK for text preprocessing
- Gensim for word embeddings
- Uvicorn as ASGI server

### Machine Learning
- Support Vector Machine (SVM) classifier
- GloVe word embeddings (100-dimensional)
- NLTK for text preprocessing (tokenization, lemmatization, stopword removal)

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- TMDb API key (free from [themoviedb.org](https://www.themoviedb.org/))

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd sentiment_analyser
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install fastapi uvicorn scikit-learn nltk gensim joblib numpy
```

### 4. Configure API Key
1. Copy the configuration template:
   ```bash
   cp config.template.js config.js
   ```
2. Edit `config.js` and replace `YOUR_API_KEY_HERE` with your actual TMDb API key:
   ```javascript
   const CONFIG = {
       TMDB_API_KEY: 'your_actual_api_key_here'
   };
   ```

### 5. Train the Model (Optional)
If you don't have the pre-trained model files, run the Jupyter notebook:
```bash
jupyter notebook main.ipynb
```
This will:
- Download NLTK data and movie reviews dataset
- Train the SVM model
- Save the model files (`sentiment_svm_model.pkl` and `model_metadata.pkl`)

### 6. Start the Backend Server
```bash
uvicorn app:app --reload --port 8000
```
The API will be available at `http://127.0.0.1:8000`

### 7. Open the Frontend
Open `index.html` in your web browser or serve it with a local server:
```bash
# Using Python's built-in server
python -m http.server 3000
# Then open http://localhost:3000
```

## Usage

1. **Enter a movie title** in the search box (e.g., "The Matrix", "Avengers")
2. **Click "Analyze"** to fetch reviews and analyze sentiments
3. **View results** including:
   - Overall sentiment summary
   - Individual review sentiments
   - Positive/negative counts

## API Endpoints

### GET `/health`
Health check endpoint
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### POST `/predict`
Analyze sentiment of text
```json
{
  "text": "This movie was amazing!"
}
```
Response:
```json
{
  "review": "This movie was amazing!",
  "sentiment": "Positive"
}
```

### GET `/docs`
Interactive API documentation (Swagger UI)

## Model Performance

The SVM model is trained on the NLTK movie reviews corpus and achieves good performance for sentiment classification. The model uses:
- TF-IDF vectorization with word embeddings
- Support Vector Machine with RBF kernel
- NLTK preprocessing pipeline

## File Structure

```
sentiment_analyser/
├── app.py                 # FastAPI backend application
├── sentiment.py           # Command-line sentiment analysis script
├── main.ipynb             # Jupyter notebook for model training
├── index.html             # Frontend web interface
├── config.template.js     # Configuration template
├── config.js              # API configuration (ignored by git)
├── .gitignore             # Git ignore rules
├── README.md              # This file
├── sentiment_svm_model.pkl    # Trained SVM model (generated)
└── model_metadata.pkl         # Model metadata (generated)
```

## Security Notes

- ⚠️ Never commit API keys to version control
- ✅ The `config.js` file is ignored by git
- ✅ Use environment variables in production
- ✅ Model files are excluded from git due to size

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NLTK team for the movie reviews dataset
- The Movie Database (TMDb) for the API
- Stanford NLP Group for GloVe embeddings
- FastAPI team for the excellent web framework
