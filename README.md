# Multilingual News Summarizer with Text-to-Speech

A web application that allows users to search for news articles, generate concise summaries, and listen to them using text-to-speech functionality in multiple languages.

![Screenshot 2025-04-24 215517](https://github.com/user-attachments/assets/6e073eb8-9de8-4115-8312-7a0495759ae5)

## Features

- **News Search**: Fetch news articles from Google News RSS feeds based on user queries
- **Automatic Summarization**: Generate concise summaries using multiple advanced algorithms:
  - ML-based summarization (TF-IDF + cosine similarity)
  - TextRank algorithm
  - KMeans clustering
  - Simple extractive summarization (fallback)
- **Multilingual Support**: Translate summaries into different languages:
  - English
  - Hindi
  - Marathi
- **Text-to-Speech**: Listen to summaries in your preferred language
- **Responsive Design**: User-friendly interface that works on all devices
- **Content Extraction**: Intelligent content extraction from various news sources
- **Error Handling**: Robust error handling and fallback mechanisms

## Technologies Used

- **Backend**: Python Flask
- **Frontend**: HTML, CSS, JavaScript
- **Natural Language Processing**: NLTK, scikit-learn
- **Content Extraction**: Diffbot API, Beautiful Soup, newspaper3k
- **Translation**: deep-translator (Google Translate)
- **Text-to-Speech**: Web Speech API (browser), pyttsx3 (server-side fallback)

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/news-summarizer.git
cd news-summarizer
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the application**

```bash
python app.py
```

5. **Access the application**

Open your browser and navigate to `http://127.0.0.1:5000`

## Usage

### Search for News

1. Enter a search term in the search box
2. Select the number of articles to display
3. Click "Search"

### Generate Summary

1. Click the "Summarize" button on any news card
2. Select your preferred language from the dropdown
3. Wait for the summary to be generated

### Listen to Summary

1. After generating a summary, click the "🔊 Speak Summary" button
2. The summary will be read aloud in the selected language

## Project Structure

```
news-summarizer/
├── app.py               # Main Flask application
├── requirements.txt     # Python dependencies
├── ml_model/            # Machine learning models
│   ├── kmeans_model.pkl
│   └── tfidf_model.pkl
├── static/              # Static assets
│   ├── contract.png
│   └── styles.css
└── templates/           # HTML templates
    ├── 404.html
    └── home.html
```

## API Endpoints

- **/** (GET): Home page with top news articles
- **/search** (POST): Search for news articles by topic
- **/summarize** (POST): Generate summary for a given URL
- **/speak** (POST): Convert text to speech

## How Summarization Works

The application uses a multi-layered approach for summarization:

1. **Content Extraction**: Extract article text using Diffbot API or web scraping
2. **Primary Algorithm**: TF-IDF vectorization and cosine similarity to identify key sentences
3. **Fallback Mechanisms**: TextRank algorithm → K-Means clustering → Simple extractive summarization
4. **Post-processing**: Truncate lengthy summaries and translate if needed

## Customization

### Adding More Languages

To add support for additional languages, update the language dropdown in `home.html`:

```html
<select id="lang-{{ news.news_id }}">
    <option value="en">English</option>
    <option value="hi">Hindi</option>
    <option value="mr">Marathi</option>
    <!-- Add more languages here -->
</select>
```

### Improving TTS for Specific Languages

For better text-to-speech support, update the language fallback patterns in `app.py`:

```python
lang_patterns = {
    'en': ['en', 'english', 'david', 'zira'],
    'hi': ['hi', 'hindi', 'indian', 'hemant', 'kalpana'],
    'mr': ['mr', 'marathi', 'hindi', 'indian'],
    # Add more languages here
}
```

## Future Improvements

- Add user accounts to save favorite articles and summaries
- Implement more sophisticated summarization algorithms (e.g., transformer-based models)
- Expand language support
- Add sentiment analysis for news articles
- Enable summarization of multiple articles together for topic-based overviews

## Credits

- News data: Google News RSS feeds
- Content extraction: Diffbot API
- Translation: Google Translate (via deep-translator)
- Icons: Various open-source libraries

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NLTK community for natural language processing tools
- scikit-learn for machine learning capabilities
- Flask community for the web framework
