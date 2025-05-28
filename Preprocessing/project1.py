import nltk
import json
import numpy as np
import heapq
import pandas as pd
import os
import streamlit as st
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from gensim.models import Word2Vec
from numpy.linalg import norm

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Set page title
st.set_page_config(page_title="Top 5 Similar Sentences Finder")
st.title("Project: Top 5 Similar Sentences Finder")

# Helper functions
def get_lemm_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_sentence(sentence,lemmatiser):
    words = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)
    lemmatised_words = []
    for word, tag in pos_tags:
        lemm_tag = get_lemm_pos(tag)
        lemmatised_words.append(lemmatiser.lemmatize(word.lower(), lemm_tag))
    return lemmatised_words

def cosine_similarity(lst1, lst2):
    return np.dot(lst1, lst2) / (norm(lst1) * norm(lst2))

def avgWord2Vec(sen,model):
    vectors = []
    for word in sen:
        if word in model.wv:
            vectors.append(model.wv[word])
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors,axis=0)

def closest_sentences(sen,model,pre_headlines,news,lemmatiser):
    p = preprocess_sentence(sen,lemmatiser)
    v_in = avgWord2Vec(p,model)
    pq = []
    heapq.heapify(pq)
    for i in range(len(pre_headlines)):
        v = avgWord2Vec(pre_headlines[i],model)
        dist = cosine_similarity(v_in,v)
        heapq.heappush(pq,(dist,i))
        if len(pq) > 5:
            heapq.heappop(pq)
    res = []
    for _,idx in pq:
        res.append((news[idx]['headline'],news[idx]['short_description']))
    return res

# Caching preprocessed data
lemmatiser = WordNetLemmatizer()
cache_file = 'pre_headlines.parquet'
news_cache_file = 'news.parquet'

@st.cache_data
def load_data(cache_file, news_cache_file):
    if os.path.exists(cache_file):
        df = pd.read_parquet(cache_file)
        pre_headlines = [list(x) if not isinstance(x, list) else x for x in df['tokens']]
        with open('news.json', 'r') as file:
            news = [json.loads(line) for line in file]
    else:
        news = []
        with open('news.json', 'r') as file:
            for line in file:
                news.append(json.loads(line))

        headlines = [
            new['headline'] + ". " + new['short_description']
            for new in news
        ]

        pre_headlines = [preprocess_sentence(h, lemmatiser) for h in headlines]
        df = pd.DataFrame({'tokens': pre_headlines})
        df.to_parquet(cache_file)

    if os.path.exists(news_cache_file):
        news_df = pd.read_parquet(news_cache_file)
        news = news_df.to_dict(orient='records')
    else:
        news_df = pd.DataFrame(news)
        news_df.to_parquet(news_cache_file)

    return pre_headlines, news

with st.spinner("🔄 Loading data..."):
    pre_headlines, news = load_data(cache_file, news_cache_file)


@st.cache_resource
def train_model(pre_headlines):
    return Word2Vec(pre_headlines, vector_size=100, min_count=1, window=5, workers=4)

if 'model' not in st.session_state:
    with st.spinner("⚙️ Training Word2Vec model..."):
        st.session_state.model = train_model(pre_headlines)
    st.success("✅ Training complete!")

model = st.session_state.model

# Input area
user_input = st.text_area("📥 Enter your query :")
search_button = st.button("Find Similar Sentences")

if user_input.strip() and search_button:
    with st.spinner("🔍 Finding closest matches..."):
        st.session_state.matches = closest_sentences(user_input, model, pre_headlines, news,lemmatiser)

if 'matches' in st.session_state:
    st.subheader("🔝 Top 5 Closest Matches")
    for i, (headline, desc) in enumerate(st.session_state.matches, start=1):
        st.markdown(f"**{i}. 🗞️ Headline:** {headline}")
        st.markdown(f"> 💬 {desc}")
        st.markdown("---")