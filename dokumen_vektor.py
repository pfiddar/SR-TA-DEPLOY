import pandas as pd
import MySQLdb

# Load csv
df = pd.read_csv('model/fasttext/dokumen_vektor.csv')

# Koneksi ke DB
conn = MySQLdb.connect(
    host="localhost",
    user="root",
    passwd="",
    db="stki"
)
cursor = conn.cursor()

# Upload dokumen vektor
for _, row in df.iterrows():
    # Cari id dokumen berdasarkan judul
    cursor.execute("SELECT id FROM documents WHERE judul = %s", (row['judul'],))
    result = cursor.fetchone()

    if result:
        id_doc = result[0]
        vector = row['vektor']
        cursor.execute(
            "INSERT INTO vector_docs (id_doc, vector) VALUES (%s, %s)", (id_doc, vector)
        )

conn.commit()
cursor.close()
conn.close()

print("Vektor dokumen berhasil disimpan ke DB")