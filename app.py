from flask import Flask, request, render_template, redirect, url_for 
import pandas as pd 
import nltk 
import string
import math
from nltk.tokenize import word_tokenize 
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory 
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 
from flask_mysqldb import MySQL 
from flask_caching import Cache
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.models import FastText
import numpy as np

app = Flask(__name__, template_folder='templates')
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Configure MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'stki'

mysql = MySQL(app)

# Load LDA final model, fasttext model, vektor dokumen, dan dictionary
lda_model = LdaModel.load("model/lda/model_lda_terbaik.model")
dictionary = Dictionary.load("model/lda/dictionary.dict")
fasttext_model = FastText.load("model/fasttext/fasttext_model.model")
df_doc_vectors = pd.read_csv('dokumen_vektor.csv')

# Identifikasi indikator/parameter vektor (judul)
doc_titles = df_doc_vectors['judul'].values
doc_vectors = df_doc_vectors.drop(columns=['judul']).values

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Stemmer and stopwords
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

# Initialize custom stopwords and combine all stopwords
custom_stopwords = set(open('custom_stoplist.txt').read().split())
combined_stopwords = stop_words.union(custom_stopwords)

@cache.memoize(timeout=250)  # Cache results for 5 minutes
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word.isalpha() and word not in combined_stopwords]
    return ' '.join(tokens)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search_ta():
    return render_template('search.html')

@app.route('/hasil_search', methods=['GET', 'POST'])
def hasil_search_ta():
    if request.method == 'POST':
        title = request.form['judul']
        return redirect(url_for('hasil_search_ta', judul=title))
    
    title = request.args.get('judul')
    if not title:
        return redirect(url_for('search_ta'))

    # Load dataset
    df = pd.read_csv('dataset_ta.csv')

    # Check if the documents table is empty
    with mysql.connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]

        if doc_count == 0:
            # Store documents in the database only if the table is empty
            try:
                for index, row in df.iterrows():
                    cursor.execute("INSERT INTO documents (judul, penulis, tahun, deskripsi, tautan, kata_kunci) VALUES (%s, %s, %s, %s, %s, %s)",
                                   (row['judul'], row['penulis'], row['tahun'], row['deskripsi'], row['tautan'], row['kata_kunci']))
                mysql.connection.commit()
            except Exception as e:
                mysql.connection.rollback()
                print(f"Error inserting documents: {e}")

    # Preprocess the search title
    preprocessed_title = preprocess_text(title)

    @cache.memoize(timeout=300)  # Cache results for 5 minutes
    def calculate_similarity(df, preprocessed_title):
        # TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(max_features=1000)
        tfidf = vectorizer.fit_transform(df['deskripsi'].apply(preprocess_text))

        try:
            with mysql.connection.cursor() as cursor:
                cursor.execute("DELETE FROM word_document")
                cursor.execute("DELETE FROM words")
                
                feature_names = vectorizer.get_feature_names_out()
                for idx, word in enumerate(feature_names):
                    cursor.execute("INSERT INTO words (word) VALUES (%s)", (word,))
                    word_id = cursor.lastrowid
                    for doc_idx, score in enumerate(tfidf[:, idx].toarray()):
                        if score > 0:
                            cursor.execute("INSERT INTO word_document (word_id, document_id, tfidf_score) VALUES (%s, %s, %s)",
                                           (word_id, doc_idx + 1, score[0]))
                mysql.connection.commit()
        except Exception as e:
            mysql.connection.rollback()
            print(f"Error indexing words: {e}")

        title_vector = vectorizer.transform([preprocessed_title])

        # Calculate similarity scores
        scores = cosine_similarity(title_vector, tfidf)[0]

        # Tambahkan bobot ekstra berdasarkan umpan balik relevansi
        try:
            with mysql.connection.cursor() as cursor:
                cursor.execute("SELECT document_id FROM relevance_feedback WHERE query = %s", (preprocessed_title,))
                relevant_docs = cursor.fetchall()
                relevant_docs = [doc[0] for doc in relevant_docs]
                for doc_id in relevant_docs:
                    scores[doc_id - 1] *= 1.5  
        except Exception as e:
            print(f"Error processing relevance feedback: {e}")

        indices = scores.argsort()[::-1]

        similar_titles = pd.DataFrame({
            'Index': df.index[indices].tolist(),
            'Judul': df['judul'].iloc[indices].tolist(),
            'Penulis': df['penulis'].iloc[indices].tolist(),
            'Tahun': df['tahun'].iloc[indices].tolist(),
            'Deskripsi': df['deskripsi'].iloc[indices].tolist(),
            'Tautan': df['tautan'].iloc[indices].tolist(),
            'Kata_kunci': df['kata_kunci'].iloc[indices].tolist()
        })

        similar_titles = similar_titles[scores[indices] > 0]  # Remove the 0.1 threshold
        return similar_titles

    similar_titles = calculate_similarity(df, preprocessed_title)

    # === Sistem Rekomendasi 
    # Identifikasi topik dokumen dari query
    def get_topic(text):
        bow_vector = dictionary.doc2bow(preprocess_text(text)) # Menentukan jumlah kemunculan setiap kata dari query
        topics = lda_model.get_document_topics(bow_vector) # Menentukan distribusi topik query
        if not topics:
            return -1
        topics = sorted(topics, key=lambda x: -x[1]) # Mengurutkan distribusi topiknya (probabilitas)
        dominant_topic = topics[0][0]
        return dominant_topic
    
    def get_fasttext_vector(text):
        tokens = preprocess_text(text)
        vectors = [fasttext_model.mv[word] for word in tokens if word in fasttext_model.mv]
        if not vectors:
            return np.zeros(fasttext_model.vector_size)
        return np.mean(vectors, axis=0)
    
    # Identifikasi topik query
    dominant_topic = get_topic(title)
    if dominant_topic == -1:
        return render_template("output.html")
    
    # Ambil dokumen dengan topik sama
    cursor.execute("SELECT * FROM documents WHERE topik_dominan = %s", (dominant_topic,))
    dokumen_filter = cursor.fetchball()

    # vektorisasi query
    query_vector = get_fasttext_vector(title)

    # Similarity antara query dengan dokumen terpilih
    similarities = []

    for dok in dokumen_filter:
        # Mencari indeks dokumen
        index = df_doc_vectors[df_doc_vectors['judul'] == dok['judul']].index
        if not index.empty:
            doc_vec = doc_vectors[index[0]]
            sim = cosine_similarity([query_vector], [doc_vectors])[0][0]
            similarities.append({'judul': dok['judul'], 'similarity': sim})

    # Mengurutkan similarity tertinggi
    top_results = sorted(similarities, key=lambda x: -x['similarity'])[:5]
    # End sistem Rekomendasi ===

    # Pagination
    page = request.args.get('page', 1, type=int)
    per_page = 8
    total_pages = math.ceil(len(similar_titles) / per_page)
    start = (page - 1) * per_page
    end = start + per_page

    data = []
    for index, row in similar_titles[start:end].iterrows():
        data.append({
            'index': row['Index'],
            'judul': row['Judul'],
            'penulis': row['Penulis'],
            'tahun': row['Tahun'],
            'deskripsi': row['Deskripsi'],
            'tautan': row['Tautan'],
            'kata_kunci': row['Kata_kunci']
        })

    jumlah_baris = len(similar_titles)

    return render_template('output.html', 
                           result_searching=data, 
                           jumlah_baris=jumlah_baris, 
                           page=page, 
                           total_pages=total_pages,
                           title=title,
                           hasil=top_results)

@app.route('/detail/<int:index>', methods=['GET'])
def detail_ta(index):
    df = pd.read_csv('dataset_ta.csv')
    if index >= len(df) or index < 0:
        return "Index out of range", 404

    detail_data = {
        'judul': df.loc[index, 'judul'],
        'penulis': df.loc[index, 'penulis'],
        'tahun': df.loc[index, 'tahun'],
        'deskripsi': df.loc[index, 'deskripsi'],
        'tautan': df.loc[index, 'tautan'],
        'kata_kunci': df.loc[index, 'kata_kunci']
    }

    return render_template('detail.html', detail_data=detail_data)

@app.route('/relevance_feedback', methods=['POST'])
def relevance_feedback():
    relevant_docs = request.form.getlist('relevant_docs')
    irrelevant_docs = request.form.getlist('irrelevant_docs')
    query = request.form['query']

    if not relevant_docs and not irrelevant_docs:
        return redirect(url_for('hasil_search_ta', judul=query))

    try:
        with mysql.connection.cursor() as cursor:
            for doc_id in relevant_docs:
                cursor.execute("INSERT INTO relevance_feedback (query, document_id, relevance) VALUES (%s, %s, %s)", (query, doc_id, 1))
            for doc_id in irrelevant_docs:
                cursor.execute("INSERT INTO relevance_feedback (query, document_id, relevance) VALUES (%s, %s, %s)", (query, doc_id, 0))
            mysql.connection.commit()
    except Exception as e:
        mysql.connection.rollback()
        print(f"Error saving relevance feedback: {e}")

    return redirect(url_for('hasil_search_ta', judul=query))

if __name__ == '__main__':
    app.run(debug=True)
