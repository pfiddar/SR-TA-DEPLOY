import pandas as pd
import MySQLdb

# Load topik dominan.csv
df = pd.read_csv('topik_dominan.csv')

# Koneksi ke DB
conn = MySQLdb.connect(
    host="localhost",
    user="root",
    passwd="",
    db="stki"
)
cursor = conn.cursor()

# Update topik dominan ke semua dokumen
for _, row in df.iterrows():
    cursor.execute("UPDATE documents SET topik_dominan = %s WHERE judul = %s", (row['topik_dominan'], row['judul']))

conn.commit()
cursor.close()
conn.close()

print("Topik dominan berhasil diupdate ke DB")