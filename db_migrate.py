import pandas as pd
import mysql.connector
import sys
import os
from datetime import datetime

def create_database_connection():
    """
    Membuat koneksi ke database MySQL
    """
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',  # Ganti dengan username MySQL Anda
            password='',  # Ganti dengan password MySQL Anda
            database='sentiment_data'
        )
        return connection
    except mysql.connector.Error as error:
        print(f"Error saat menghubungkan ke MySQL: {error}")
        return None

def setup_database():
    """
    Setup database dari file schema
    """
    try:
        # Koneksi tanpa database untuk membuat database jika belum ada
        connection = mysql.connector.connect(
            host='localhost',
            user='root',  # Ganti dengan username MySQL Anda
            password=''   # Ganti dengan password MySQL Anda
        )
        
        cursor = connection.cursor()
        
        # Membuat database jika belum ada
        cursor.execute("CREATE DATABASE IF NOT EXISTS sentiment_data")
        cursor.close()
        connection.close()
        
        # Koneksi dengan database untuk eksekusi schema
        connection = create_database_connection()
        if not connection:
            return False
        
        cursor = connection.cursor()            # Membaca file schema
        schema_file = 'db_schema.sql'
        with open(schema_file, 'r') as file:
            schema_script = file.read()
          # Eksekusi setiap statement dalam schema
        statements = schema_script.split(';')
        for statement in statements:
            statement = statement.strip()
            if statement:
                try:
                    # Skip comments and empty lines
                    if statement.startswith('--') or statement == '':
                        continue
                        
                    # For CREATE INDEX statements, extract table name and index name properly
                    if "CREATE INDEX" in statement.upper():
                        parts = statement.split()
                        if len(parts) >= 4:
                            index_name = parts[2]  # Get index name
                            table_name = parts[4].split('(')[0]  # Get table name
                            
                            # Check if index already exists
                            try:
                                cursor.execute(f"SHOW INDEX FROM {table_name} WHERE Key_name = '{index_name}'")
                                if cursor.fetchone():
                                    print(f"Index {index_name} on {table_name} already exists, skipping creation.")
                                    continue
                            except mysql.connector.Error as err:
                                print(f"Could not check index: {err}")
                                # Continue to try creating the index
                    
                    cursor.execute(statement)
                except mysql.connector.Error as error:
                    # If error is duplicate key, just continue
                    if "Duplicate key name" in str(error):
                        print(f"Index already exists, skipping: {error}")
                    else:
                        print(f"Error executing statement: {error}")
                        print(f"Statement: {statement}")
        
        connection.commit()
        cursor.close()
        connection.close()
        return True
    
    except mysql.connector.Error as error:
        print(f"Error saat setup database: {error}")
        return False

def import_data_from_csv(csv_file):
    """
    Import data dari CSV ke database
    """
    try:
        # Baca file CSV
        df = pd.read_csv(csv_file)
        print(f"Total baris dalam CSV: {len(df)}")
        
        # Koneksi ke database
        connection = create_database_connection()
        if not connection:
            return False
            
        cursor = connection.cursor()
        
        # Hapus data yang sudah ada (opsional)
        confirm = input("Hapus data yang sudah ada di database? (y/n): ")
        if confirm.lower() == 'y':
            cursor.execute("DELETE FROM dataset_splits")
            cursor.execute("DELETE FROM sentiments")
            cursor.execute("DELETE FROM metadata")
            cursor.execute("DELETE FROM texts")
            print("Data lama telah dihapus")
        
        # Import data baris per baris
        rows_imported = 0
        for _, row in df.iterrows():
            # Format tanggal jika ada
            created_at = row.get('created_at', None)
            if created_at and not pd.isna(created_at):
                try:
                    created_at = datetime.strptime(created_at, '%a %b %d %H:%M:%S %z %Y')
                except ValueError:
                    try:
                        created_at = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        created_at = None
            else:
                created_at = None
                  # Handle NaN values for all fields
            text = row['text'] if not pd.isna(row['text']) else ""
            preprocessed_text = row['preprocessed_text'] if not pd.isna(row['preprocessed_text']) else ""
            source_dataset = row.get('source_dataset', 'imported') if not pd.isna(row.get('source_dataset', 'imported')) else 'imported'
            sentiment = row['sentiment'] if not pd.isna(row['sentiment']) else 'neutral'
            confidence = row['confidence'] if not pd.isna(row['confidence']) else 0.0
            favorite_count = row.get('favorite_count', 0) if not pd.isna(row.get('favorite_count', 0)) else 0
            location = row.get('location', None) if not pd.isna(row.get('location', None)) else None
            username = row.get('username', None) if not pd.isna(row.get('username', None)) else None
            
            # Insert ke tabel texts
            cursor.execute(
                "INSERT INTO texts (text, preprocessed_text, created_at, source_dataset) VALUES (%s, %s, %s, %s)",
                (
                    text,
                    preprocessed_text,
                    created_at,
                    source_dataset
                )
            )
            text_id = cursor.lastrowid
            
            # Insert ke tabel sentiments
            cursor.execute(
                "INSERT INTO sentiments (text_id, sentiment, confidence) VALUES (%s, %s, %s)",
                (
                    text_id,
                    sentiment,
                    confidence
                )
            )
            
            # Insert ke tabel metadata
            cursor.execute(
                "INSERT INTO metadata (text_id, favorite_count, location, username) VALUES (%s, %s, %s, %s)",
                (
                    text_id,
                    favorite_count,
                    location,
                    username
                )
            )
            
            rows_imported += 1
            if rows_imported % 100 == 0:
                print(f"Imported {rows_imported} rows...")
        
        # Commit perubahan
        connection.commit()
        print(f"Berhasil mengimpor {rows_imported} baris data")
        
        cursor.close()
        connection.close()
        return True
    
    except Exception as e:
        print(f"Error saat mengimpor data: {e}")
        return False

if __name__ == "__main__":
    print("Setup Database untuk Sentiment Analysis")
    
    # Setup database
    print("Setting up database...")
    if not setup_database():
        print("Gagal setup database. Keluar program.")
        sys.exit(1)
    
    # Import data
    csv_path = 'static/data/labeled_data.csv'
    if not os.path.exists(csv_path):
        print(f"File {csv_path} tidak ditemukan.")
        sys.exit(1)
        
    print(f"Mulai proses import data dari {csv_path}...")
    if import_data_from_csv(csv_path):
        print("Import data selesai")
    else:
        print("Gagal import data")
