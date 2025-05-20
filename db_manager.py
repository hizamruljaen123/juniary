import mysql.connector
from mysql.connector import Error
import pandas as pd
import json
import pickle
import io

class DatabaseManager:
    """
    Class untuk mengelola koneksi dan operasi database
    """
    def __init__(self, host='localhost', user='root', password='', database='sentiment_data'):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        
    def connect(self):
        """
        Membuat koneksi ke database
        """
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            return True
        except Error as e:
            print(f"Error connecting to MySQL database: {e}")
            return False
    
    def disconnect(self):
        """
        Menutup koneksi database
        """
        if self.connection and self.connection.is_connected():
            self.connection.close()
    
    def execute_query(self, query, params=None, commit=False):
        """
        Eksekusi query dan return cursor
        """
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()
                
            cursor = self.connection.cursor(dictionary=True)
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            if commit:
                self.connection.commit()
            
            return cursor
        except Error as e:
            print(f"Error executing query: {e}")
            print(f"Query: {query}")
            print(f"Params: {params}")
            return None
    
    def fetch_all_data(self, limit=None):
        """
        Ambil semua data berlabel dari database
        """
        try:
            query = "SELECT * FROM labeled_data_view"
            if limit:
                query += f" LIMIT {limit}"
                
            cursor = self.execute_query(query)
            if cursor:
                results = cursor.fetchall()
                cursor.close()
                return pd.DataFrame(results)
            return None
        except Error as e:
            print(f"Error fetching data: {e}")
            return None
    
    def get_data_by_sentiment(self, sentiment):
        """
        Ambil data berdasarkan jenis sentimen
        """
        try:
            query = "SELECT * FROM labeled_data_view WHERE sentiment = %s"
            cursor = self.execute_query(query, (sentiment,))
            if cursor:
                results = cursor.fetchall()
                cursor.close()
                return pd.DataFrame(results)
            return None
        except Error as e:
            print(f"Error fetching data by sentiment: {e}")
            return None
    
    def save_text_with_sentiment(self, text, preprocessed_text, sentiment, confidence, 
                                favorite_count=0, created_at=None, location=None, 
                                username=None, source_dataset='user_input'):
        """
        Simpan teks baru beserta hasil analisis sentimen
        """
        try:
            # Insert teks
            query_text = """
                INSERT INTO texts (text, preprocessed_text, created_at, source_dataset) 
                VALUES (%s, %s, %s, %s)
            """
            cursor = self.execute_query(query_text, 
                                    (text, preprocessed_text, created_at, source_dataset), 
                                    commit=True)
            
            if not cursor:
                return False
            
            text_id = cursor.lastrowid
            cursor.close()
            
            # Insert sentimen
            query_sentiment = """
                INSERT INTO sentiments (text_id, sentiment, confidence)
                VALUES (%s, %s, %s)
            """
            cursor = self.execute_query(query_sentiment, 
                                      (text_id, sentiment, confidence), 
                                      commit=True)
            
            if not cursor:
                return False
            
            cursor.close()
            
            # Insert metadata
            query_metadata = """
                INSERT INTO metadata (text_id, favorite_count, location, username)
                VALUES (%s, %s, %s, %s)
            """
            cursor = self.execute_query(query_metadata, 
                                      (text_id, favorite_count, location, username), 
                                      commit=True)
            
            if not cursor:
                return False
            
            cursor.close()
            return True
            
        except Error as e:
            print(f"Error saving text with sentiment: {e}")
            return False
    
    def get_sentiment_stats(self):
        """
        Ambil statistik sentimen dari database
        """
        try:
            # Count by sentiment
            query_sentiment_counts = """
                SELECT sentiment, COUNT(*) as count 
                FROM sentiments 
                GROUP BY sentiment
            """
            cursor = self.execute_query(query_sentiment_counts)
            if not cursor:
                return None
                
            sentiment_counts = {row['sentiment']: row['count'] for row in cursor.fetchall()}
            cursor.close()
            
            # Average confidence
            query_avg_confidence = "SELECT AVG(confidence) as avg_confidence FROM sentiments"
            cursor = self.execute_query(query_avg_confidence)
            if not cursor:
                return None
                
            avg_confidence = cursor.fetchone()['avg_confidence']
            cursor.close()
            
            # Average favorite count
            query_avg_likes = "SELECT AVG(favorite_count) as avg_likes FROM metadata"
            cursor = self.execute_query(query_avg_likes)
            if not cursor:
                return None
                
            avg_likes = cursor.fetchone()['avg_likes']
            cursor.close()
            
            # Dataset counts
            query_dataset_counts = """
                SELECT source_dataset, COUNT(*) as count 
                FROM texts 
                GROUP BY source_dataset
            """
            cursor = self.execute_query(query_dataset_counts)
            if not cursor:
                return None
                
            dataset_counts = {row['source_dataset']: row['count'] for row in cursor.fetchall()}
            cursor.close()
            
            # Total entries
            query_total = "SELECT COUNT(*) as total FROM texts"
            cursor = self.execute_query(query_total)
            if not cursor:
                return None
                
            total_entries = cursor.fetchone()['total']
            cursor.close()
            
            return {
                'total_entries': total_entries,
                'sentiment_counts': sentiment_counts,
                'avg_confidence': avg_confidence,
                'avg_likes': avg_likes,
                'dataset_counts': dataset_counts
            }
            
        except Error as e:
            print(f"Error getting sentiment stats: {e}")
            return None
    
    def save_model(self, model_name, model_obj, vectorizer_obj, accuracy):
        """
        Simpan model ke database
        """
        try:
            # Serialize objects
            model_data = pickle.dumps(model_obj)
            vectorizer_data = pickle.dumps(vectorizer_obj)
            
            # Check if model exists
            query_check = "SELECT id FROM models WHERE model_name = %s"
            cursor = self.execute_query(query_check, (model_name,))
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                # Update existing model
                query = """
                    UPDATE models 
                    SET model_data = %s, vectorizer_data = %s, accuracy = %s, created_at = NOW()
                    WHERE model_name = %s
                """
                cursor = self.execute_query(query, 
                                         (model_data, vectorizer_data, accuracy, model_name), 
                                         commit=True)
            else:
                # Insert new model
                query = """
                    INSERT INTO models (model_name, model_data, vectorizer_data, accuracy)
                    VALUES (%s, %s, %s, %s)
                """
                cursor = self.execute_query(query, 
                                         (model_name, model_data, vectorizer_data, accuracy), 
                                         commit=True)
            
            if not cursor:
                return False
                
            cursor.close()
            return True
            
        except Error as e:
            print(f"Error saving model: {e}")
            return False
    
    def get_model(self, model_name):
        """
        Ambil model dari database
        """
        try:
            query = "SELECT model_data, vectorizer_data, accuracy FROM models WHERE model_name = %s"
            cursor = self.execute_query(query, (model_name,))
            
            if not cursor:
                return None
                
            result = cursor.fetchone()
            cursor.close()
            
            if not result:
                return None
                
            model = pickle.loads(result['model_data'])
            vectorizer = pickle.loads(result['vectorizer_data'])
            accuracy = result['accuracy']
            
            return {
                'model': model,
                'vectorizer': vectorizer,
                'accuracy': accuracy
            }
            
        except Error as e:
            print(f"Error getting model: {e}")
            return None
    
    def save_rules(self, rule_type, rule_data):
        """
        Simpan rules ke database
        """
        try:
            # Convert to JSON
            rule_data_json = json.dumps(rule_data)
            
            # Delete existing rules of this type
            query_delete = "DELETE FROM rules WHERE rule_type = %s"
            cursor = self.execute_query(query_delete, (rule_type,), commit=True)
            cursor.close()
            
            # Insert new rules
            for rule_name, rule_content in rule_data.items():
                query = """
                    INSERT INTO rules (rule_type, rule_name, rule_data)
                    VALUES (%s, %s, %s)
                """
                cursor = self.execute_query(query, 
                                         (rule_type, rule_name, json.dumps(rule_content)), 
                                         commit=True)
                cursor.close()
                
            return True
            
        except Error as e:
            print(f"Error saving rules: {e}")
            return False
    
    def load_rules(self):
        """
        Method alias for get_rules() to maintain compatibility with SentimentAnalyzer
        """
        return self.get_rules()
        
    def get_rules(self):
        """
        Ambil semua rules dari database
        """
        try:
            query = "SELECT rule_type, rule_name, rule_data FROM rules"
            cursor = self.execute_query(query)
            
            if not cursor:
                return None
                
            results = cursor.fetchall()
            cursor.close()
            
            # Convert to dictionary format expected by analyzer
            rules = {
                "confidence_rules": {},
                "sentiment_classification": {},
                "popularity_rules": {},
                "impact_rules": {}
            }
            
            for row in results:
                rule_type = row['rule_type']
                rule_name = row['rule_name']
                rule_data = json.loads(row['rule_data'])
                
                if rule_type == 'confidence_rules':
                    rules['confidence_rules'][rule_name] = rule_data
                elif rule_type == 'sentiment_classification':
                    rules['sentiment_classification'][rule_name] = rule_data
                elif rule_type == 'popularity_rules':
                    rules['popularity_rules'][rule_name] = rule_data
                elif rule_type == 'impact_rules':
                    rules['impact_rules'][rule_name] = rule_data
            
            return rules
            
        except Error as e:
            print(f"Error getting rules: {e}")
            return None
    def clear_dataset_splits(self):
        """
        Hapus semua pembagian dataset (train/test)
        """
        try:
            query_clear = "DELETE FROM dataset_splits"
            cursor = self.execute_query(query_clear, commit=True)
            cursor.close()
            return True
        except Error as e:
            print(f"Error clearing dataset splits: {e}")
            return False
            
    def save_dataset_split(self, train_ids, test_ids):
        """
        Simpan pembagian dataset (train/test)
        """
        try:
            # Clear existing splits
            self.clear_dataset_splits()
            
            # Insert training data
            for text_id in train_ids:
                query = "INSERT INTO dataset_splits (text_id, split_type) VALUES (%s, 'train')"
                cursor = self.execute_query(query, (text_id,), commit=True)
                cursor.close()
            
            # Insert test data
            for text_id in test_ids:
                query = "INSERT INTO dataset_splits (text_id, split_type) VALUES (%s, 'test')"
                cursor = self.execute_query(query, (text_id,), commit=True)
                cursor.close()
                
            return True
            
        except Error as e:
            print(f"Error saving dataset split: {e}")
            return False
    
    def get_dataset_split(self):
        """
        Ambil pembagian dataset (train/test)
        """
        try:
            # Get training data
            query_train = """
                SELECT t.* 
                FROM labeled_data_view t
                JOIN dataset_splits ds ON t.id = ds.text_id
                WHERE ds.split_type = 'train'
            """
            cursor = self.execute_query(query_train)
            if not cursor:
                return None
                
            train_data = cursor.fetchall()
            cursor.close()
            
            # Get test data
            query_test = """
                SELECT t.* 
                FROM labeled_data_view t
                JOIN dataset_splits ds ON t.id = ds.text_id
                WHERE ds.split_type = 'test'
            """
            cursor = self.execute_query(query_test)
            if not cursor:
                return None
                
            test_data = cursor.fetchall()
            cursor.close()
            
            return {
                'train': pd.DataFrame(train_data) if train_data else pd.DataFrame(),
                'test': pd.DataFrame(test_data) if test_data else pd.DataFrame()
            }
            
        except Error as e:
            print(f"Error getting dataset split: {e}")
            return None
    
    def export_to_csv(self, file_path):
        """
        Export semua data ke CSV
        """
        try:
            df = self.fetch_all_data()
            if df is not None:
                df.to_csv(file_path, index=False)
                return True
            return False
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False
