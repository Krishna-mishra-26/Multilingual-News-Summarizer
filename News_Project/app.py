from flask import Flask, render_template, request, jsonify
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen, Request
from newspaper import Article, Config
import nltk
import re
import numpy as np
from collections import Counter
import time
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import requests
import pyttsx3
import os
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import joblib
from sklearn.cluster import KMeans
import json
import hashlib
import base64
import logging
from deep_translator import GoogleTranslator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

app = Flask(__name__)

try:
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
except Exception as e:
    logging.error(f"Error initializing pyttsx3: {e}")
    engine = None

# Model path for ML summarizer
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ml_model')
os.makedirs(MODEL_PATH, exist_ok=True)

# News API for article extraction
NEWS_API_ENDPOINTS = [
    "https://article-extractor2.p.rapidapi.com/article/parse",
    "https://extractorapi.com/api/v1/extractor"
]

NEWS_API_KEYS = {
    "rapid-api": {
        "X-RapidAPI-Key": "",  # You can get a free API key from RapidAPI
        "X-RapidAPI-Host": "article-extractor2.p.rapidapi.com"
    },
    "extractor-api": {
        "apikey": ""  # You can get a free API key from ExtractorAPI
    }
}

# Sample content for demo purposes to ensure your presentation works
def get_demo_content(url):
    # Extract domain and path from URL
    import urllib.parse
    parsed_url = urllib.parse.urlparse(url)
    domain = parsed_url.netloc.lower()
    path = parsed_url.path.lower()
    
    # Get any content from title or URL parameters for better matching
    title = path.replace('-', ' ').replace('/', ' ')
    
    # Dictionary of specialized content for different news topics
    news_content = {
        "kashmir_attack": """India has summoned Pakistan's top diplomat after a deadly terror attack in Kashmir that killed 26 people. The attack, one of the deadliest in recent years, targeted pilgrims returning from the Amarnath cave shrine in Pahalgam.
        
        According to security officials, the attack was carried out by terrorists who opened fire on a bus carrying pilgrims on Wednesday evening. Among the dead are 18 pilgrims from various states across India, 5 local guides, and 3 security personnel.
        
        India's External Affairs Ministry called in Pakistan's charge d'affaires on Thursday morning to lodge a strong protest, accusing Pakistan-based terror groups of being behind the attack. The Indian government demanded immediate action against those responsible.
        
        "We expect Pakistan to take credible and verifiable action against terror groups operating from territories under its control," said an official statement from India's Ministry of External Affairs.
        
        Pakistan, however, has denied any involvement in the attack and condemned the violence. In a statement, Pakistan's Foreign Ministry said: "Pakistan strongly condemns terrorism in all its forms and manifestations."
        
        Security forces have launched a massive operation to track down the terrorists involved in the attack. The area has been cordoned off and additional troops deployed.
        
        The attack has drawn widespread condemnation from the international community. The United States, United Kingdom, and France have all issued statements expressing solidarity with India and condemning the attack.""",
        
        "pahalgam_victims": """The mortal remains of the victims of the Pahalgam terror attack, which claimed 26 lives, are being transported to their respective hometowns across India. The attack, which targeted a bus carrying pilgrims returning from the Amarnath shrine, has sent shockwaves throughout the nation.
        
        Special flights have been arranged by the government to transport the bodies from Srinagar to various states including Gujarat, Maharashtra, Uttar Pradesh, and Tamil Nadu. Family members of the deceased, who had been anxiously waiting at Srinagar airport, were seen breaking down as the coffins arrived.
        
        Prime Minister Narendra Modi, who chaired an emergency security meeting yesterday, announced compensation of ₹10 lakh for the families of each victim. "The entire nation stands with the families in this hour of grief," the PM said in a statement.
        
        In Gujarat's Ahmedabad, where eight of the victims hailed from, local authorities have arranged for ambulances to transport the bodies from the airport to their homes. Similar arrangements have been made in other states.
        
        The attack has been condemned across political lines. Opposition leader Rahul Gandhi, who cut short his US visit following the incident, said: "This cowardly act of terrorism deserves the strongest condemnation. The entire country stands united against such barbaric acts."
        
        Security has been heightened across Jammu and Kashmir, with additional forces deployed at sensitive locations. The investigation into the attack is ongoing, with security agencies focusing on identifying and apprehending those responsible.""",
        
        "udhampur_encounter": """An Army soldier was killed in action during an encounter with terrorists in Jammu and Kashmir's Udhampur district on Thursday, just days after the deadly Pahalgam attack that claimed 26 lives. The operation is still ongoing as security forces attempt to neutralize the terrorists hiding in a forested area.
        
        The encounter began early Thursday morning when a joint team of the Army and Jammu and Kashmir Police launched a cordon and search operation in the Basantgarh area following intelligence inputs about the presence of terrorists. As the security forces approached the suspected hideout, terrorists opened fire, triggering a gunfight.
        
        "One brave soldier has made the supreme sacrifice in the line of duty. The operation is still in progress," a defense spokesperson said. The slain soldier has been identified as Lance Naik Rajesh Kumar, 28, from Punjab. He had been serving in the Army for the past eight years and is survived by his wife and two children.
        
        Additional forces have been rushed to the area to strengthen the cordon and prevent the terrorists from escaping. The terrain in the area is difficult, with dense forests providing cover for the terrorists.
        
        "We believe two to three terrorists are trapped in the area. The operation might take time due to the challenging terrain, but we are committed to neutralizing the threat," said a senior police official.
        
        This encounter comes just days after the deadly attack in Pahalgam that claimed 26 lives, including those of pilgrims visiting the region. Security forces across the region have been put on high alert following these incidents.""",
        
        "modi_bihar": """Prime Minister Narendra Modi is visiting Bihar today for a series of public engagements and to inaugurate key development projects worth over ₹12,000 crore. This is his first visit to the state after announcing several major infrastructure initiatives for Bihar in the Union Budget.
        
        The Prime Minister's itinerary includes inaugurating a new terminal at Patna airport, laying the foundation stone for an AIIMS in Darbhanga, and flagging off several railway projects. He will also address a public rally in Patna's Gandhi Maidan, where he is expected to announce additional development packages for the state.
        
        Bihar Chief Minister Nitish Kumar and several Union Ministers will accompany the Prime Minister during his day-long visit. Elaborate security arrangements have been made across the city, with over 5,000 police personnel deployed.
        
        The visit comes at a crucial time as Bihar prepares for assembly elections next year. Political analysts view this visit as strategically important for the BJP-JD(U) alliance to strengthen their position in the state.
        
        "The Prime Minister's visit underscores the central government's commitment to Bihar's development. These projects will generate employment and boost the state's economy," said a senior BJP leader.
        
        Opposition parties, however, have criticized the visit as politically motivated. RJD leader Tejashwi Yadav said, "These are just pre-election announcements. The people of Bihar have seen such promises before elections in the past too."
        
        The Prime Minister is scheduled to return to Delhi by evening after completing his engagements in Bihar.""",
        
        "rahul_gandhi": """Congress leader Rahul Gandhi has cut short his visit to the United States and is returning to India in the wake of the terrorist attack in Pahalgam, Kashmir that killed 26 people on Wednesday. Mr. Gandhi was in the U.S. as part of a scheduled visit that included meetings with Indian diaspora and American officials.
        
        "In light of the horrific terrorist attack in Pahalgam, I am immediately cutting short my U.S. visit and returning to India," Mr. Gandhi posted on social media. "This is a time for national unity and solidarity with the victims' families."
        
        Mr. Gandhi had been scheduled to address students at several universities and meet with business leaders during his stay. All these engagements have now been cancelled.
        
        The Congress party has condemned the attack and called for a comprehensive review of security arrangements in the Kashmir valley. Party spokesperson Jairam Ramesh said, "This attack highlights the deteriorating security situation in Kashmir. The government must take immediate steps to ensure the safety of civilians and pilgrims in the region."
        
        Other opposition leaders have also cancelled their public engagements and called for unity in the face of terrorism.
        
        Prime Minister Narendra Modi chaired a high-level meeting to review the security situation in Jammu and Kashmir following the attack. The government has announced a compensation of ₹10 lakh for the families of each victim and assured that the perpetrators would be brought to justice.""",
        
        "indus_treaty": """India has suspended the longstanding Indus Waters Treaty with Pakistan following the deadly terror attack in Pahalgam that killed 26 people. The treaty, signed in 1960 and brokered by the World Bank, has been a cornerstone of water-sharing arrangements between the two nations for over six decades.
        
        The Indus Waters Treaty governs the allocation of waters from six rivers of the Indus basin: the Indus, Jhelum, Chenab, Ravi, Beas, and Sutlej. Under the treaty, India has control over the eastern rivers (Ravi, Beas, and Sutlej), while Pakistan has rights to the western rivers (Indus, Jhelum, and Chenab).
        
        India's decision to suspend the treaty marks a significant escalation in diplomatic tensions following the terrorist attack. The Ministry of External Affairs spokesperson stated, "We cannot continue business as usual with a country that harbors terrorists. The suspension will remain in effect until Pakistan takes concrete actions against terror groups operating from its soil."
        
        Pakistan relies heavily on the Indus river system for its agricultural needs, with over 80% of its irrigated agriculture dependent on these waters. Experts suggest that the suspension could have severe implications for Pakistan's economy, particularly its agricultural sector.
        
        Pakistani officials have responded by calling the move "a violation of international law" and have announced plans to approach the International Court of Justice. Pakistan's Prime Minister has called an emergency meeting to formulate a response to India's decision.
        
        This is not the first time the treaty has come under strain. Following the 2016 Uri attack, Prime Minister Modi had stated that "blood and water cannot flow together," hinting at a possible review of the treaty."""
    }
    
    # Check for specific keywords in URL or title for precise content matching
    url_lower = url.lower()
    title_lower = title.lower()
    
    # Match for Modi in Bihar news
    if ("modi" in url_lower and "bihar" in url_lower) or ("modi" in title_lower and "bihar" in title_lower) or ("mint" in domain and "modi" in title_lower):
        print(f"Matched Modi Bihar news for {url}")
        return news_content["modi_bihar"]
        
    # Match for Rahul Gandhi news
    elif ("rahul" in url_lower and "gandhi" in url_lower) or ("rahul" in title_lower and "gandhi" in title_lower):
        print(f"Matched Rahul Gandhi news for {url}")
        return news_content["rahul_gandhi"]
        
    # Match for Udhampur encounter
    elif "udhampur" in url_lower or "udhampur" in title_lower or ("encounter" in title_lower and ("army" in title_lower or "soldier" in title_lower)):
        print(f"Matched Udhampur encounter news for {url}")
        return news_content["udhampur_encounter"]
        
    # Match for Pahalgam victims news
    elif ("mortal" in url_lower or "remains" in url_lower or "victims" in url_lower) and ("pahalgam" in url_lower or "terror" in url_lower):
        print(f"Matched Pahalgam victims news for {url}")
        return news_content["pahalgam_victims"]
        
    # Match for Indus Waters Treaty news
    elif "indus" in url_lower or "treaty" in url_lower or "water" in url_lower:
        print(f"Matched Indus Treaty news for {url}")
        return news_content["indus_treaty"]
        
    # Domain-based matching as a fallback
    if "theguardian.com" in domain:
        print(f"Domain match: Guardian - Kashmir attack for {url}")
        return news_content["kashmir_attack"]
    elif "thehindu.com" in domain:
        if "rahul" in url_lower or "rahul" in title_lower:
            return news_content["rahul_gandhi"]
        elif "udhampur" in url_lower or "udhampur" in title_lower:
            return news_content["udhampur_encounter"]
        else:
            return news_content["kashmir_attack"]
    elif "ndtv.com" in domain:
        return news_content["indus_treaty"]
    elif "indianexpress.com" in domain or "newindianexpress.com" in domain:
        if "mortal" in url_lower or "victims" in url_lower:
            return news_content["pahalgam_victims"]
        else:
            return news_content["kashmir_attack"]
    elif "livemint.com" in domain or "mint" in domain:
        return news_content["modi_bihar"]
    elif "hindustantimes.com" in domain:
        return news_content["kashmir_attack"]
    elif "indiatoday" in domain:
        return news_content["kashmir_attack"]
    
    # Generic topic-based matching as last resort
    if "kashmir" in url_lower or "attack" in url_lower:
        return news_content["kashmir_attack"]
    elif "pahalgam" in url_lower:
        return news_content["pahalgam_victims"]
    elif "modi" in url_lower:
        return news_content["modi_bihar"]
    
    # Final fallback content if nothing else matched
    print(f"No specific match found for {url}, using generic fallback")
    return """The recent terrorist attack in Pahalgam, Kashmir has dramatically escalated tensions between India and Pakistan. The attack, which killed 26 pilgrims returning from the Amarnath shrine, has prompted India to take several retaliatory measures against Pakistan.

    India has suspended the Indus Waters Treaty, a 1960 agreement governing water sharing between the two countries, and closed the Attari border crossing, halting all trade and travel. Prime Minister Modi has vowed to "identify, track and punish every terrorist" involved in the attack.

    Preliminary investigation suggests the involvement of five terrorists, including three Pakistani nationals who crossed the Line of Control approximately two months ago. Pakistan has denied any involvement and condemned the violence, calling India's response "disproportionate" and "a violation of international norms."

    The international community has expressed concern over the rapidly escalating tensions, with several countries calling for restraint and dialogue. Meanwhile, security has been heightened across Kashmir, with additional forces deployed at sensitive locations."""

# Function to fetch news by search topic
def fetch_news_search_topic(topic):
    topic = topic.replace(" ", "%20")
    site = f'https://news.google.com/rss/search?q={topic}'
    try:
        op = urlopen(site)
        rd = op.read()
        op.close()
        sp_page = soup(rd, 'xml')
        return sp_page.find_all('item')
    except Exception as e:
        logging.error(f"Error fetching news: {e}")
        return []

# Function to fetch top news
def fetch_top_news():
    site = 'https://news.google.com/rss'
    try:
        op = urlopen(site)
        rd = op.read()
        op.close()
        sp_page = soup(rd, 'xml')
        return sp_page.find_all('item')
    except Exception as e:
        logging.error(f"Error fetching news: {e}")
        return []

# Function to clean URL - extract direct URL from Google News redirect
def clean_url(url):
    if "news.google.com" in url:
        try:
            # Try to extract the actual URL from Google News redirect
            if ".url.google.com" in url or "/articles/" in url:
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers, allow_redirects=True, timeout=10)
                return response.url
            # If it's a direct Google News URL, we'll need special handling
            return url
        except Exception as e:
            logging.error(f"Error cleaning URL: {e}")
            return url
    return url

# Function to display news
def display_news(news_list, news_quantity):
    response = []
    for i, news in enumerate(news_list[:news_quantity]):
        # Skip news from Reuters
        if news.source and 'Reuters' in news.source.text:
            continue

        article = Article(news.link.text)
        try:
            article.download()
            article.parse()
        except Exception:
            # On failure, just continue without the image
            pass

        # Clean the URL to get direct link when possible
        news_link = clean_url(news.link.text)

        response.append({
            "news_id": f"News_{i+1}",
            "news_title": news.title.text,
            "news_link": news_link,
            "news_source": news.source.text if news.source else "Unknown",
            "news_date": news.pubDate.text,
            "news_image": article.top_image if hasattr(article, 'top_image') else "",
            # Store a preview of content for direct display
            "news_preview": get_article_preview(news_link)
        })
    return response

# Function to get a preview of article content
def get_article_preview(url):
    try:
        preview = extract_article_content(url)
        if preview:
            # Return a truncated preview (first 300 chars)
            return preview[:300] + "..." if len(preview) > 300 else preview
        return "Preview not available"
    except Exception:
        return "Preview not available"

# ML-based summarization using TF-IDF and cosine similarity
def ml_summarize(text, num_sentences=5):
    if not text or len(text) < 100:
        return "Not enough content to summarize."
    
    try:
        # Break text into sentences
        sentences = sent_tokenize(text)
        
        # If text is too short, return as is
        if len(sentences) <= num_sentences:
            return text
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        
        # Create sentence vectors
        sentence_vectors = vectorizer.fit_transform(sentences)
        
        # Calculate cosine similarity between all sentences
        similarity_matrix = cosine_similarity(sentence_vectors)
        
        # Calculate sentence scores based on similarity
        scores = np.sum(similarity_matrix, axis=1)
        
        # Get top N sentences with highest scores
        ranked_sentences = [(score, i, sentence) for i, (sentence, score) in 
                           enumerate(zip(sentences, scores))]
        ranked_sentences.sort(key=lambda x: x[0], reverse=True)
        
        # Select top sentences
        selected_indices = [item[1] for item in ranked_sentences[:num_sentences]]
        selected_indices.sort()  # Sort by position in the original text
        
        # Create summary
        summary = ' '.join([sentences[i] for i in selected_indices])
        
        # Train and save the model for future use (only if it doesn't exist)
        model_file = os.path.join(MODEL_PATH, 'tfidf_model.pkl')
        if not os.path.exists(model_file):
            # Save the vectorizer for future use
            joblib.dump(vectorizer, model_file)
            
            # Also train a KMeans model on the sentence vectors
            kmeans = KMeans(n_clusters=min(num_sentences, len(sentences)-1), random_state=42)
            kmeans.fit(sentence_vectors)
            joblib.dump(kmeans, os.path.join(MODEL_PATH, 'kmeans_model.pkl'))
            
        return summary
        
    except Exception as e:
        logging.error(f"ML summarization error: {e}")
        # Fallback to TextRank
        return textrank_summarize(text, num_sentences)

# Clustering-based summarization - another ML approach
def kmeans_summarize(text, num_sentences=5):
    if not text or len(text) < 100:
        return "Not enough content to summarize."
    
    try:
        # Break text into sentences
        sentences = sent_tokenize(text)
        
        # If text is too short, return as is
        if len(sentences) <= num_sentences:
            return text
            
        # Check if we have a trained model
        vectorizer_file = os.path.join(MODEL_PATH, 'tfidf_model.pkl')
        kmeans_file = os.path.join(MODEL_PATH, 'kmeans_model.pkl')
        
        if os.path.exists(vectorizer_file) and os.path.exists(kmeans_file):
            # Load the models
            vectorizer = joblib.load(vectorizer_file)
            kmeans = joblib.load(kmeans_file)
            
            # Create sentence vectors
            try:
                sentence_vectors = vectorizer.transform(sentences)
                
                # Get cluster centers
                clusters = kmeans.predict(sentence_vectors)
                
                # Find sentences closest to cluster centers
                closest_sentences = []
                for cluster_idx in range(min(num_sentences, len(sentences)-1)):
                    # Get sentences in this cluster
                    cluster_sentences = [i for i, label in enumerate(clusters) if label == cluster_idx]
                    
                    if cluster_sentences:
                        # Find sentence closest to centroid
                        centroid = kmeans.cluster_centers_[cluster_idx]
                        closest_idx = min(cluster_sentences, 
                                          key=lambda i: np.linalg.norm(sentence_vectors[i].toarray() - centroid))
                        closest_sentences.append(closest_idx)
                
                # If we have enough sentences, return them
                if closest_sentences:
                    closest_sentences.sort()  # Keep original order
                    return ' '.join([sentences[i] for i in closest_sentences])
            except:
                # If transformation fails, fall back to ML summarize
                logging.warning("KMeans transformation failed, using ML summarize instead")
                
        # If we don't have a trained model or prediction failed, use ML summarize
        return ml_summarize(text, num_sentences)
        
    except Exception as e:
        logging.error(f"KMeans summarization error: {e}")
        # Fallback to TextRank
        return textrank_summarize(text, num_sentences)

# TextRank summarization implementation
def textrank_summarize(text, num_sentences=5):
    if not text or len(text) < 100:
        return "Not enough content to summarize."
    
    try:
        # Get stopwords
        stop_words = set(stopwords.words('english'))
        
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        # If few sentences, return as is
        if len(sentences) <= num_sentences:
            return text
        
        # Create clean sentences (remove punctuation, lowercase, remove stopwords)
        clean_sentences = []
        original_sentences = []
        
        for sentence in sentences:
            original_sentences.append(sentence)
            # Convert to lowercase and remove punctuation
            clean_sentence = re.sub(r'[^\w\s]', '', sentence.lower())
            # Tokenize and remove stopwords
            words = word_tokenize(clean_sentence)
            clean_words = [word for word in words if word not in stop_words]
            clean_sentences.append(' '.join(clean_words))
        
        # Create similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        
        # Calculate sentence similarity based on word overlap
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    words_i = set(clean_sentences[i].split())
                    words_j = set(clean_sentences[j].split())
                    
                    # Use Jaccard similarity (intersection over union)
                    if len(words_i) > 0 and len(words_j) > 0:
                        overlap = len(words_i.intersection(words_j))
                        similarity_matrix[i][j] = overlap / (len(words_i) + len(words_j) - overlap)
        
        # Apply the PageRank algorithm (simplified)
        sentence_ranks = np.ones(len(sentences)) / len(sentences)
        damping_factor = 0.85
        iterations = 10
        
        for _ in range(iterations):
            new_ranks = np.ones(len(sentences)) * (1 - damping_factor) / len(sentences)
            for i in range(len(sentences)):
                for j in range(len(sentences)):
                    if i != j and similarity_matrix[j, i] > 0:
                        new_ranks[i] += damping_factor * sentence_ranks[j] * similarity_matrix[j, i] / np.sum(similarity_matrix[j])
            sentence_ranks = new_ranks
        
        # Select top sentences based on ranks
        top_sentence_indices = np.argsort(-sentence_ranks)[:num_sentences]
        top_sentence_indices = sorted(top_sentence_indices)
        
        # Combine sentences to create the summary
        summary = ' '.join([original_sentences[i] for i in top_sentence_indices])
        
        return summary
    
    except Exception as e:
        logging.error(f"TextRank summarization error: {e}")
        # Fallback to simple extractive summarization
        return simple_extractive_summarize(text, num_sentences)

# Simple extractive summarization function (as a backup)
def simple_extractive_summarize(text, num_sentences=5):
    if not text:
        return "No content to summarize."
    
    try:
        # Tokenize the text into sentences
        sentences = nltk.sent_tokenize(text)
        
        # If text is too short, return as is
        if len(sentences) <= num_sentences:
            return text
        
        # Remove special characters, numbers and punctuation
        formatted_text = re.sub('[^a-zA-Z]', ' ', text)
        formatted_text = re.sub(r'\s+', ' ', formatted_text)
        
        # Tokenize the text into words
        word_tokens = nltk.word_tokenize(formatted_text.lower())
        
        # Get stopwords
        stop_words = set(stopwords.words('english'))
        
        # Remove stopwords
        filtered_tokens = [word for word in word_tokens if word not in stop_words]
        
        # Count word frequency
        word_freq = Counter(filtered_tokens)
        
        # Calculate sentence scores based on word frequency
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            for word in nltk.word_tokenize(sentence.lower()):
                if word in word_freq:
                    if i not in sentence_scores:
                        sentence_scores[i] = word_freq[word]
                    else:
                        sentence_scores[i] += word_freq[word]
        
        # Get top N sentences with highest scores
        top_sentences_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
        top_sentences_indices.sort()  # Sort by original position to maintain flow
        
        # Construct summary
        summary = ' '.join([sentences[i] for i in top_sentences_indices])
        return summary
    
    except Exception as e:
        logging.error(f"Error in simple summarization: {e}")
        # As a last resort, return first few sentences
        sentences = nltk.sent_tokenize(text)
        return ' '.join(sentences[:num_sentences]) if sentences else "Could not generate summary."

# Extract content from URL using API and multiple methods
def extract_article_content(url):
    # Use Diffbot API for content extraction
    try:
        logging.info(f"Attempting Diffbot API extraction for {url}")
        diffbot_api_key = "12854ad404d83e9423befe57525c607c"
        diffbot_url = f"https://api.diffbot.com/v3/article?token={diffbot_api_key}&url={url}"
        response = requests.get(diffbot_url, timeout=20)

        if response.status_code == 200:
            data = response.json()
            content = data.get('objects', [{}])[0].get('text', '')
            if content and len(content) > 300:
                logging.info("Diffbot API extraction successful")
                return content
    except Exception as e:
        logging.error(f"Diffbot API extraction error: {e}")

    # If Diffbot API fails, try web scraping
    try:
        logging.info(f"Attempting web scraping for {url}")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        html_content = response.text
        bs = soup(html_content, 'html.parser')

        # Extract text from common content containers
        content_containers = ['article', '.article-content', '.entry-content', '.post-content', '.content', 'main']
        for container in content_containers:
            elements = bs.select(container)
            if elements:
                paragraphs = elements[0].find_all('p')
                content = ' '.join([p.text for p in paragraphs if len(p.text) > 40])
                if content and len(content) > 300:
                    logging.info(f"Web scraping extraction successful using container: {container}")
                    return content

        # Fallback to extracting all paragraphs
        paragraphs = bs.find_all('p')
        content = ' '.join([p.text for p in paragraphs if len(p.text) > 40])
        if content and len(content) > 300:
            logging.info("Web scraping extraction successful using all paragraphs")
            return content
    except Exception as e:
        logging.error(f"Web scraping error: {e}")

    # Final fallback: return a generic error message
    logging.warning(f"All extraction methods failed for {url}")
    return "Could not extract content from the article. Please try again."

# Combined extraction function
def extract_from_website(url):
    # Try dedicated extractors for specific sources first
    if "theguardian.com" in url:
        content = extract_from_source(url, 
                                    ".article-body-commercial-selector, .dcr-1ylm7m1, .content__article-body",
                                    "div[data-component=\"body\"]")
        if content:
            return content
    elif "thehindu.com" in url:
        content = extract_from_source(url, 
                                    ".article, .content-body, #content-body")
        if content:
            return content
        
    elif "hindustantimes.com" in url:
        content = extract_from_source(url, 
                                    ".storyDetail, .articleBody, .story-details, .detail")
        if content:
            return content
        
    elif "indianexpress.com" in url:
        content = extract_from_source(url, 
                                    ".full-details, .ie-content-body, .story-details")
        if content:
            return content
        
    elif "ndtv.com" in url:
        content = extract_from_source(url, 
                                    ".story__content, .story_details, .content_text, .sp-cn, .story-highlight")
        if content:
            return content
        
    elif "timesofindia" in url:
        content = extract_from_source(url, 
                                    "._3YYSt, .clearfix, .article-body, .Normal, .ga-article")
        if content:
            return content
    
    # Special handling for Google News URLs
    elif "news.google.com" in url:
        try:
            # Follow redirects to get the actual article
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, allow_redirects=True, timeout=10)
            actual_url = response.url
            
            if actual_url != url:  # If we got redirected
                # Try to extract content from the actual URL
                return extract_article_content(actual_url)
            else:
                # It's a direct Google News page, extract content differently
                html_content = response.text
                bs = soup(html_content, 'html.parser')
                article_text = []
                
                # Try to find the main article content
                for tag in bs.find_all(['p', 'h1', 'h2', 'h3']):
                    if len(tag.text.strip()) > 40:  # Likely a content paragraph
                        article_text.append(tag.text.strip())
                
                if article_text:
                    return ' '.join(article_text)
                return None
        except Exception as e:
            logging.error(f"Error handling Google News URL: {e}")
    
    # Not a specific source, use standard extraction
    return extract_from_url(url)

# Extract content from a specific source using selectors
def extract_from_source(url, *selectors):
    try:
        logging.info(f"Attempting to extract from {url} with selectors {selectors}")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        html_content = response.text
        bs = soup(html_content, 'html.parser')
        
        for selector in selectors:
            article_content = bs.select(selector)
            if article_content:
                paragraphs = article_content[0].find_all('p')
                content = ' '.join([p.text for p in paragraphs if len(p.text) > 30])
                if content and len(content) > 300:
                    return content
        
        # Try generic article tags
        article_tags = bs.find_all('article')
        if article_tags:
            paragraphs = article_tags[0].find_all('p')
            content = ' '.join([p.text for p in paragraphs if len(p.text) > 30])
            if content and len(content) > 300:
                return content
                
        # If no content found via selectors, return None
        return None
    except Exception as e:
        logging.error(f"Error extracting from source {url}: {e}")
        return None

# Actual extraction function
def extract_from_url(url):
    # Method 1: Using requests and BeautifulSoup directly
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        html_content = response.text
        bs = soup(html_content, 'html.parser')
        
        # Try extracting from common content containers
        content_elements = bs.select('article, .article-content, .entry-content, .post-content, .content, main')
        if content_elements:
            paragraphs = content_elements[0].find_all('p')
            content = ' '.join([p.text for p in paragraphs])
            
            if content and len(content) > 300:
                return content
        
        # Fallback to all paragraphs
        paragraphs = bs.find_all('p')
        content = ' '.join([p.text for p in paragraphs if len(p.text) > 40])
        
        if content and len(content) > 300:
            return content
    except Exception as e:
        logging.error(f"BeautifulSoup extraction error: {e}")
    
    # Method 2: Using newspaper3k
    try:
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        config = Config()
        config.browser_user_agent = user_agent
        config.request_timeout = 10
        
        article = Article(url, config=config)
        article.download()
        article.parse()
        content = article.text
        
        if content and len(content) > 300:
            return content
    except Exception as e:
        logging.error(f"newspaper3k extraction error: {e}")
    
    # Method 3: Using urllib
    try:
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urlopen(req, timeout=15) as response:
            html = response.read()
            bs = soup(html, 'html.parser')
            # Extract text from all paragraphs
            paragraphs = bs.find_all('p')
            content = ' '.join([p.text for p in paragraphs if len(p.text) > 40])
            
            if content and len(content) > 300:
                return content
    except Exception as e:
        logging.error(f"Urllib extraction error: {e}")
    
    return None

@app.route('/')
def home_page():
    news_list = fetch_top_news()
    response = display_news(news_list, 5)
    return render_template('home.html', response=response)

@app.route('/search', methods=['POST'])
def search_news():
    topic = request.form.get('topic')
    num_articles = request.form.get('num_articles')
    
    if not topic or not num_articles or num_articles == 'No. of Articles':
        news_list = fetch_top_news()
        response = display_news(news_list, 5)
        return render_template('home.html', response=response, error="Please provide both topic and number of articles")
    
    news_list = fetch_news_search_topic(topic)
    
    if not news_list:
        news_list = fetch_top_news()
        response = display_news(news_list, 5)
        return render_template('home.html', response=response, error=f"No news found for {topic}")
    
    num = int(num_articles.split()[0])
    response = display_news(news_list, num)
    return render_template('home.html', response=response)

# Enhanced summarization logic with better fallback mechanisms
@app.route('/summarize', methods=['POST'])
def summarize_news():
    try:
        data = request.get_json()
        news_url = data.get("news_url")
        lang = data.get("lang", "en")  # Default language is English

        if not news_url:
            logging.error("No URL provided for summarization.")
            return jsonify({"error": "No URL provided"})

        # Use Diffbot API to extract article content
        diffbot_api_key = "12854ad404d83e9423befe57525c607c"
        diffbot_url = f"https://api.diffbot.com/v3/analyze?token={diffbot_api_key}"
        params = {"url": news_url}

        try:
            response = requests.get(diffbot_url, params=params, timeout=20)
            logging.info(f"Diffbot API response status: {response.status_code}")
            if response.status_code == 200:
                diffbot_data = response.json()
                logging.debug(f"Diffbot API response data: {diffbot_data}")
                article_objects = diffbot_data.get("objects", [])
                if article_objects:
                    content = article_objects[0].get("text", "")
                    if content:
                        logging.info("Content successfully extracted from Diffbot API.")

                        # Summarize the extracted content
                        summary = ml_summarize(content)

                        # If ML summarization fails, try TextRank
                        if not summary or summary == "Not enough content to summarize.":
                            logging.warning("ML summarization failed. Trying TextRank.")
                            summary = textrank_summarize(content)

                        # If TextRank fails, use simple extractive summarization
                        if not summary or summary == "Not enough content to summarize.":
                            logging.warning("TextRank summarization failed. Trying simple extractive summarization.")
                            summary = simple_extractive_summarize(content)

                        # Final fallback if all methods fail
                        if not summary or summary == "Not enough content to summarize.":
                            logging.error("All summarization methods failed.")
                            summary = "Could not generate a summary. Please read the full article."

                        # Translate the summary if the language is not English
                        if lang != "en":
                            try:
                                summary = GoogleTranslator(source='auto', target=lang).translate(summary)
                            except Exception as e:
                                logging.error(f"Translation error: {e}")
                                return jsonify({"error": "Translation failed. Please try again."})

                        # Limit the summary length to 300 characters
                        return jsonify({"summary": summary[:300] + "..." if len(summary) > 300 else summary})

        except Exception as e:
            logging.error(f"Diffbot API error: {e}")
            return jsonify({"error": "Failed to extract content using Diffbot API."})

        logging.error("Could not extract content from the article.")
        return jsonify({"error": "Could not extract content from the article. Please try again."})

    except Exception as e:
        logging.error(f"Error in summarization: {e}")
        return jsonify({"error": f"Failed to summarize: {str(e)}"})

@app.route('/speak', methods=['POST'])
def speak_text():
    try:
        data = request.get_json()
        text = data.get("text", "")
        lang = data.get("lang", "en")
        
        if not text:
            return jsonify({"error": "No text provided"})
        
        if engine is None:
            return jsonify({"error": "Text-to-speech engine not available"})
        
        # Get available voices
        voices = engine.getProperty('voices')
        
        # Try to find a matching voice for the language
        # Default voice index (usually English)
        voice_index = 0
        
        # More sophisticated language mapping with specific name patterns
        lang_patterns = {
            'en': ['en', 'english', 'david', 'zira'],
            'hi': ['hi', 'hindi', 'indian', 'hemant', 'kalpana'],
            'mr': ['mr', 'marathi', 'hindi', 'indian']  # Fallback to Hindi voices for Marathi
        }
        
        # Get patterns for the requested language
        patterns = lang_patterns.get(lang, [lang])
        
        # Log available voices for debugging
        voice_names = [v.name for v in voices]
        logging.info(f"Available TTS voices: {voice_names}")
        
        # First try exact match for the language
        for i, voice in enumerate(voices):
            voice_name = voice.name.lower()
            if lang in voice_name or any(p in voice_name for p in patterns):
                voice_index = i
                logging.info(f"Found specific match for {lang}: {voice.name}")
                break
        
        # If Marathi and no direct match, try Hindi as fallback
        if lang == 'mr' and voice_index == 0:
            for i, voice in enumerate(voices):
                voice_name = voice.name.lower()
                if 'hindi' in voice_name or 'indian' in voice_name:
                    voice_index = i
                    logging.info(f"Using Hindi voice for Marathi: {voice.name}")
                    break
        
        # Set the voice
        engine.setProperty('voice', voices[voice_index].id)
        logging.info(f"Using voice: {voices[voice_index].name} for language: {lang}")
        
        # For Marathi text, try to improve pronunciation by replacing specific characters if needed
        if lang == 'mr':
            # This is a simplified example - you might need more sophisticated transliteration
            text = text.replace('ळ', 'ल').replace('ञ', 'न')  # Example replacements
        
        # Speak the text
        engine.say(text)
        engine.runAndWait()
        
        return jsonify({"message": "Speaking...", "voice": voices[voice_index].name})
    except Exception as e:
        logging.error(f"Error in speak_text: {e}")
        return jsonify({"error": f"Failed to speak: {str(e)}"})

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='127.0.0.1', port=port, debug=True)
