from flask import Flask, request, render_template, redirect, url_for, make_response, session, g
import pandas as pd, nltk, string, math, pymysql, fasttext
import numpy as np, json, uuid, secrets, traceback, os, gdown
from nltk.tokenize import word_tokenize 
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory 
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 
from flask_caching import Cache
from gensim.models import LdaModel
from gensim.corpora import Dictionary

app = Flask(__name__, template_folder='templates')
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
app.secret_key = secrets.token_hex(32) # Security enkrip session

# Configure MySQL
def get_connection_dict():
    return pymysql.connect(
        host=os.getenv("MYSQLHOST"),
        port=int(os.getenv("MYSQLPORT", 10724)),
        user=os.getenv("MYSQLUSER"),
        password=os.getenv("MYSQLPASSWORD"),
        database=os.getenv("MYSQLDATABASE"),
        cursorclass=pymysql.cursors.DictCursor
    )
def get_connection_tuple():
    return pymysql.connect(
        host=os.getenv("MYSQLHOST"),
        port=int(os.getenv("MYSQLPORT", 10724)),
        user=os.getenv("MYSQLUSER"),
        password=os.getenv("MYSQLPASSWORD"),
        database=os.getenv("MYSQLDATABASE"),
        cursorclass=pymysql.cursors.Cursor
    )
conn_tuple = get_connection_tuple()
conn_dict = get_connection_dict()

model_path = "model/fasttext/fasttext_model.bin"

# Jika file model fasttext belum ada maka download lewat GDrive
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1TncKz_Jh_SkYuUIs6GlLRxLHCndFKL5i"  # path model fasttext di Gdrive
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print("Downloading FastText model...")
    gdown.download(url, model_path, quiet=False)

if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"[INFO] FastText model ditemukan, ukuran: {size_mb:.2f} MB")
else:
    print("[ERROR] FastText model tidak ditemukan!")

# Load LDA final model, fasttext model, vektor dokumen, dan dictionary
lda_model = LdaModel.load("model/lda/model_lda_terbaik.model")
dictionary = Dictionary.load("model/lda/dictionary.dict")
fasttext_model = fasttext.load_model(model_path)
df_doc_vectors = pd.read_csv("model/fasttext/dokumen_vektor.csv")

# Load dataset
df = pd.read_csv('dataset_ta.csv')

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

def safe_stem(word, stemmer):
    try:
        return stemmer.stem(word)
    except SystemExit:
        return word
    except Exception as e:
        print(f"Stemming error: {e}")
        return word

# Buat koneksi baru jika terputus untuk koneksi tuple dan dict
def ensure_connection_tuple():
    global conn_tuple
    try:
        conn_tuple.ping(reconnect=True)
    except:
        conn_tuple = get_connection_tuple()
    return conn_tuple

def ensure_connection_dict():
    global conn_dict
    try:
        conn_dict.ping(reconnect=True)
    except:
        conn_dict = get_connection_dict()
    return conn_dict

# Initialize custom stopwords and combine all stopwords
custom_stopwords = set(open('custom_stoplist.txt').read().split())
combined_stopwords = stop_words.union(custom_stopwords)

@cache.memoize(timeout=300)  # Cache results for 5 minutes
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [safe_stem(word, stemmer) for word in tokens if word.isalpha() and word not in combined_stopwords]
    return ' '.join(tokens)

@cache.memoize(timeout=300)
def preprocess_to_tokens(text):
    return preprocess_text(text).split()

@app.route('/')
def index():
    return render_template('index.html')

@app.before_request
def identify_user():
    # Deteksi cookie consent
    consent = request.cookies.get('cookie_consent')
    if consent == 'true':
        session['consent_given'] = True # Sesi disimpan

        # Request user token untuk yang menyetujui cookie
        user_token = request.cookies.get('user_token')
        if not user_token:
            user_token = str(uuid.uuid4())
            g.set_user_cookie = user_token
        else:
            g.user_token = user_token

        # Buat sesi id baru tiap kunjungan
        g.session_id = str(uuid.uuid4())
    else:
        session['consent_given'] = False   

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
    
    # Simpan search query ke session
    session['last_query'] = title

    # Check if the documents table is empty
    conn = ensure_connection_tuple()
    with conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]

        if doc_count == 0:
            # Store documents in the database only if the table is empty
            try:
                for index, row in df.iterrows():
                    cursor.execute("INSERT INTO documents (judul, penulis, tahun, deskripsi, tautan, kata_kunci) VALUES (%s, %s, %s, %s, %s, %s)",
                                   (row['judul'], row['penulis'], row['tahun'], row['deskripsi'], row['tautan'], row['kata_kunci']))
                conn.connection.commit()
            except Exception as e:
                conn.connection.rollback()
                print(f"Error inserting documents: {e}")

    # Preprocess the search title
    preprocessed_title = preprocess_text(title)

    @cache.memoize(timeout=300)  # Cache results for 5 minutes
    def calculate_similarity(df, preprocessed_title):
        # TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(max_features=1000)
        tfidf = vectorizer.fit_transform(df['deskripsi'].apply(preprocess_text))

        conn = ensure_connection_tuple()
        try:
            with conn.cursor() as cursor:
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
                conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"Error indexing words: {e}")

        title_vector = vectorizer.transform([preprocessed_title])

        # Calculate similarity scores
        scores = cosine_similarity(title_vector, tfidf)[0]

        # Tambahkan bobot ekstra berdasarkan umpan balik relevansi
        conn = ensure_connection_tuple()
        try:
            with conn.cursor() as cursor:
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
    response = make_response()
    
    # Fallback default
    user_token = None
    session_id = None 

    # Cek atau buat id sesi
    if session.get('consent_given'):
        # Set cookie jika belum ada
        if hasattr(g, 'set_user_cookie'):
            response.set_cookie('user_token', g.set_user_cookie, max_age=60*60*24*7) # 1 minggu
        
        # Simpan sesi
        user_token = g.user_token if hasattr(g, 'user_token') else g.set_user_cookie
        session['session_id'] = g.session_id
        session_id = g.session_id

        # Ambil user_id dari tabel users
        conn = ensure_connection_tuple()
        with conn.cursor() as cursor:
            cursor.execute("SELECT id FROM users WHERE user_token = %s", (user_token,))
            user = cursor.fetchone()
            if user:
                user_id = user[0]
            else:
                # Insert user_token baru
                cursor.execute("INSERT INTO users (user_token) VALUES (%s)", (user_token,))
                conn.commit()
                user_id = cursor.lastrowid

            # Simpan sesi ke DB
            cursor.execute("INSERT INTO user_sessions (session_id, user_id) VALUES (%s, %s)", (session_id, user_id))
            conn.commit()

    # Identifikasi topik dokumen dari query
    @cache.memoize(timeout=300)
    def get_topic(text):
        bow_vector = dictionary.doc2bow(preprocess_to_tokens(text)) # Menentukan jumlah kemunculan setiap kata dari query
        topics = lda_model.get_document_topics(bow_vector) # Menentukan distribusi topik query
        if not topics:
            return -1
        topics = sorted(topics, key=lambda x: -x[1]) # Mengurutkan distribusi topiknya (probabilitas)
        dominant_topic = topics[0][0]
        return dominant_topic
    
    # Representasi vektor query
    @cache.memoize(timeout=300)
    def get_fasttext_vector(text):
        tokens = preprocess_to_tokens(text)
        vectors = [fasttext_model.get_word_vector(word) for word in tokens]
        if not vectors:
            return np.zeros(fasttext_model.get_dimension())
        return np.mean(vectors, axis=0)
    
    # Identifikasi topik query
    dominant_topic = get_topic(preprocessed_title)
    if dominant_topic == -1:
        return render_template("output.html")

    # Ambil preferensi pengguna --- (cookie) ---
    def get_preference_vec(user_token=None):
        conn = ensure_connection_dict()
        try:
            with conn.cursor() as cursor:
                if user_token is not None:
                    cursor.execute(
                        """SELECT user_query, COALESCE(rf.freq, 0) + COALESCE(log.freq, 0) AS total_freq, COALESCE(rf.freq_fb, 0) AS feedback_freq
                        FROM (
							SELECT lr.user_query, COUNT(*) AS freq 
							FROM log_recommendations lr
							JOIN user_sessions us ON lr.session_id = us.id
							JOIN users u ON us.user_id = u.id 
							WHERE u.user_token = %s GROUP BY lr.user_query
						) log
                        LEFT JOIN (
                            SELECT rf.query, COUNT(*) AS freq, SUM(CASE WHEN rf.relevance > 0 THEN 1 ELSE 0 END) AS freq_fb
                            FROM relevance_feedback rf 
                            JOIN user_sessions us ON rf.session_id = us.session_id
							JOIN users u ON us.user_id = u.id 
                            WHERE u.user_token = %s
                            GROUP BY rf.query    
                        ) rf ON log.user_query = rf.query""", (user_token, user_token)
                    )
                    rows = cursor.fetchall()
                    print(f"[DEBUG] Jumlah rows dari query preference: {len(rows)}")
                    if rows:
                        print("[DEBUG] Contoh row pertama:", rows[0])
                    else:
                        print("[DEBUG] Query preference tidak mengembalikan data")
                else: 
                    return None
            
            # Hitung bobot & vektor rata-rata
            bobot_vecs = []
            for row in rows:
                freq_query = float(row['total_freq'] or 0)
                freq_fb = float(row['feedback_freq'] or 0)
                weight = freq_query + (0.5 * freq_fb)
                vec = get_fasttext_vector(row['user_query']) 
                bobot_vecs.append(vec * weight)
                # Debug
                print("[DEBUG] query:", row['user_query'])
                print("        freq_query:", freq_query)
                print("        freq_feedback:", freq_fb)
                print("        bobot:", weight)
                print("        vektor dim:", len(vec))
            if not bobot_vecs:
                return None
            print("[DEBUG] total vektor terkumpul:", len(bobot_vecs))
            print("[DEBUG] contoh vektor pertama:", bobot_vecs[0][:10]) 
            return np.mean(bobot_vecs, axis=0) # Rata-rata
        except Exception as e:
            print("Error: ", e)
            traceback.print_exc()
            raise 
    
    # Ambil topik dominan dari preferensi --- (cookie) ---
    @cache.memoize(timeout=300)
    def get_preference_topics(top_n=5):
        topic_vectors = []
        try:
            for topic_id in range(lda_model.num_topics):
                topic_words = [word for word, _ in lda_model.show_topic(topic_id, top_n)]
                word_vecs = [fasttext_model.get_word_vector(word) for word in topic_words]
                if word_vecs:
                    topic_vectors.append(np.mean(word_vecs, axis=0))
                else:
                    topic_vectors.append(np.zeros(fasttext_model.get_dimension()))
                    print("Topik vektor : ", topic_vectors)
            return topic_vectors
        except Exception as e:
            print("Error: ", e)
            traceback.print_exc()
            raise    
    
    # Similarity antara preferensi dengan dokumen --- (cookie) ---
    def get_preference_similarities():
        print("[DEBUG] >>> Masuk ke get_preference_similarities <<<")
        top_n_docs = int(request.args.get("top_n_docs", 5))

            # Hitung vektor preferensi
        pref_vec = get_preference_vec(user_token)
        if pref_vec is None:
            print("[DEBUG] pref_vec kosong (None) → langsung return []")
            return []
            # Debug
            # print("[DEBUG] pref_vec shape:", pref_vec.shape if pref_vec is not None else None)
            # print("[DEBUG] pref_vec sample:", pref_vec[:10] if pref_vec is not None else None)
        topic_vecs = get_preference_topics() # Menghitung vektor tiap topik
            # Debug
            # print("[DEBUG] topic_vecs count:", len(topic_vecs))
            # print("[DEBUG] topic_vec[0] sample:", topic_vecs[0][:10])

        # Mencari 3 topik paling mirip
        sims = cosine_similarity([pref_vec], topic_vecs)[0]
        top_topics = np.argsort(sims)[::-1][:3]

        results = []
        conn = ensure_connection_dict()
        with conn.cursor() as cursor:
            try:
                for topic_id in top_topics:
                    cursor.execute("""
                        SELECT d.id, d.judul, dv.vector FROM documents d JOIN vector_docs_bins dv ON d.id = dv.id_doc WHERE d.topik_dominan = %s LIMIT %s
                    """, (int(topic_id), top_n_docs))
                    rows = cursor.fetchall()
                    print(f"[DEBUG] Jumlah dokumen dari topic_id={topic_id}: {len(rows)}")
                    for row in rows:
                        doc_vec = np.array(json.loads(row['vector']))
                        sim_docs = cosine_similarity([pref_vec], [doc_vec])[0][0]
                        results.append({
                            'id': row['id'],
                            'topic_id': int(topic_id), 
                            'judul': row['judul'], 
                            'similarity': sim_docs
                        })
                # Debug
            except Exception as e:
                print("Error: ", e)
                traceback.print_exc()
                raise
        # Mengurutkan hasil
        # return sorted(results, key=lambda x: -x['similarity'])[:5]
        # Mengurutkan hasil setelah loop selesai
        sorted_results = sorted(results, key=lambda x: -x['similarity'])

        # Cetak 5 hasil terbaik dari list yang sudah diurutkan
        print("Hasil preferensi setelah disortir: ", sorted_results[:5])

        # Kembalikan 5 hasil terbaik
        return sorted_results[:5]

    # Similarity antara query dengan dokumen terpilih
    def get_top_similarities(preprocessed_title, dominant_topic):
        query_vector = get_fasttext_vector(preprocessed_title)
        
        conn = ensure_connection_dict()
        with conn.cursor() as cursor:
            # Ambil dokumen dengan topik sama
            cursor.execute("SELECT d.id, d.judul, dv.vector FROM documents d JOIN vector_docs_bins dv ON d.id = dv.id_doc WHERE d.topik_dominan = %s", (dominant_topic,))
            rows = cursor.fetchall()

        similarities = []
        for row in rows:
            try:
                doc_vec = np.array(json.loads(row['vector']))
                sim = cosine_similarity([query_vector], [doc_vec])[0][0]
                similarities.append({
                    'id': row['id'], 
                    'judul': row['judul'], 
                    'similarity': sim
                    })
            except Exception as e:
                print("Error parsing vektor untuk judul: ", row['judul'], e)
                continue

        # Mengurutkan similarity tertinggi
        return sorted(similarities, key=lambda x: -x['similarity'])[:10]
        
    # Hasil rekomendasi
    pref_results = []
    general_results = []
    if user_token:
        print("[DEBUG] user_token ada:", user_token)
        if session_id:
            print("[DEBUG] session_id ada:", session_id)
            try:
                pref_results = get_preference_similarities()[:5]
                print("[DEBUG] hasil pref_results:", pref_results)
                general_results = get_top_similarities(preprocessed_title, dominant_topic)[:5]
            except Exception as e:
                print("Error saat ambil rekomendasi dengan preferensi: ", e)
        else:
            try:
                general_results = get_top_similarities(preprocessed_title, dominant_topic)[:10]
            except Exception as e:
                print("Tidak bisa menampilkan rekomendasi dari topik: ", e)
    else:
        general_results = get_top_similarities(preprocessed_title, dominant_topic)[:10]

    # Simpan log rekomendasi jika pakai cookie
    if session.get('consent_given'):
        conn = ensure_connection_tuple()
        with conn.cursor() as cursor:
            # Simpan log rekomendasi
            for result in pref_results:
                cursor.execute("""INSERT INTO log_recommendations (session_id, user_query, id_doc, similarity) 
                               VALUES ((SELECT id FROM user_sessions WHERE session_id = %s), %s, %s, %s)""",
                    (session_id, title, result['id'], result['similarity']))
            for result in general_results:
                cursor.execute("""INSERT INTO log_recommendations (session_id, user_query, id_doc, similarity) 
                               VALUES ((SELECT id FROM user_sessions WHERE session_id = %s), %s, %s, %s)""",
                    (session_id, title, result['id'], result['similarity']))
            conn.commit()
    # End sistem Rekomendasi ===

    # Pagination
    page = request.args.get('page', 1, type=int)
    per_page = 10
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

    response = make_response(render_template('output.html', 
                           result_searching=data, 
                           jumlah_baris=jumlah_baris, 
                           page=page, 
                           total_pages=total_pages,
                           title=title,
                           pref_results=pref_results,
                           general_results=general_results))
    
    # Set cookie
    if session.get('consent_given') and hasattr(g, 'set_user_cookie'):
        response.set_cookie('user_token', g.set_user_cookie, max_age=60*60*24*7)
    return response

@app.route('/detail/<int:index>', methods=['GET'])
def detail_ta(index):
    df = pd.read_csv('dataset_ta.csv')
    if index >= len(df) or index < 0:
        return "Index out of range", 404

    detail_data = {
        'index': index,
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
    session_id = session.get('session_id') or request.form.get('session_id')
    if not session_id:
        session_id = None
        
    if not relevant_docs and not irrelevant_docs:
        return redirect(url_for('hasil_search_ta', judul=query))

    conn = ensure_connection_tuple()
    try:
        with conn.cursor() as cursor:
            for doc_id in relevant_docs:
                cursor.execute("INSERT INTO relevance_feedback (query, document_id, relevance, session_id) VALUES (%s, %s, %s, %s)", (query, doc_id, 1, session_id))
            for doc_id in irrelevant_docs:
                cursor.execute("INSERT INTO relevance_feedback (query, document_id, relevance, session_id) VALUES (%s, %s, %s, %s)", (query, doc_id, 0, session_id))
            conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error saving relevance feedback: {e}")

    return redirect(url_for('hasil_search_ta', judul=query))

if __name__ == '__main__':
    app.run(debug=True)
