from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for, session, Response, stream_with_context
import json
import re
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils
from collections import defaultdict
import io
import graphviz
import shutil
import matplotlib.pyplot as plt
from sklearn import tree as sktree
from collections import Counter
import pickle
import time
import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from labeling import stream_process_dataset
from db_manager import DatabaseManager
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from labeling import stream_process_dataset

class SentimentAnalyzer:
    def __init__(self):
        # Initialize database manager and connect
        self.db_manager = DatabaseManager()
        self.db_connected = self.db_manager.connect()
        
        if self.db_connected:
            print("✅ Database connection successful for SentimentAnalyzer")
        else:
            print("⚠️ Warning: Database connection failed. Using CSV fallback.")
        
        self.rules = self.load_rules_from_db()
        
        # Try to load the transformer model, but continue if it fails
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("w11wo/indonesian-roberta-base-sentiment-classifier")
            self.model = AutoModelForSequenceClassification.from_pretrained("w11wo/indonesian-roberta-base-sentiment-classifier")
            self.model.eval()
            print("✅ Transformer model loaded successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not load transformer model: {e}")
            self.tokenizer = None
            self.model = None

        # Fallback to CSV file paths
        self.labeled_data_path = 'static/data/labeled_data.csv'
        self.default_data_path = 'static/data/labeled_data.csv'

        # Initialize with default data if available and no labeled data exists
        if not os.path.exists(self.labeled_data_path) and os.path.exists(self.default_data_path):
            shutil.copy2(self.default_data_path, self.labeled_data_path)
            if os.path.exists(self.labeled_data_path):
                df = pd.read_csv(self.labeled_data_path)
                self.rules = generate_rules_from_data(df)

    def load_rules_from_db(self):
        """
        Load classification rules from database
        """
        rules = self.db_manager.load_rules()
        if rules is None:
            return load_rules()  # Fallback to file-based rules
        return rules

    def analyze_text(self, text):
        text = self.preprocess_text(text)

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            confidence = torch.max(probabilities).item()

        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        sentiment = sentiment_map[prediction.item()]

        result = {
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence
        }

        return result    
    def get_or_create_labeled_dataset(self):
        """Get existing labeled dataset or create from default data"""
        # Try to get data from database first
        if hasattr(self, 'db_manager') and self.db_manager.connection:
            df = self.db_manager.fetch_all_data()
            if df is not None and not df.empty:
                return df

        # Fallback to CSV if database failed or is empty
        if hasattr(self, 'labeled_data_path') and os.path.exists(self.labeled_data_path):
            return pd.read_csv(self.labeled_data_path)
        elif hasattr(self, 'default_data_path') and os.path.exists(self.default_data_path):
            df = pd.read_csv(self.default_data_path)
            if hasattr(self, 'labeled_data_path'):
                df.to_csv(self.labeled_data_path, index=False)
            return df
        return None    
    def create_labeled_dataset(self, df1=None, df2=None):
        """Create labeled dataset from provided dataframes or default data"""
        # If no data provided, try to load from database first
        if df1 is None or df2 is None:
            if hasattr(self, 'db_manager') and self.db_manager.connection:
                df = self.db_manager.fetch_all_data()
                if df is not None and not df.empty:
                    return df

            # Fallback to CSV if database failed or is empty
            if hasattr(self, 'default_data_path') and os.path.exists(self.default_data_path):
                df = pd.read_csv(self.default_data_path)
                if hasattr(self, 'labeled_data_path'):
                    df.to_csv(self.labeled_data_path, index=False)
                self.rules = generate_rules_from_data(df)
                return df
            return None

        labeled_data = []

        for df in [df1, df2]:
            for _, row in df.iterrows():
                if pd.isna(row['full_text']):
                    continue

                analysis = self.analyze_text(row['full_text'])
                extended_analysis = self.apply_rules(analysis, row['favorite_count'])
                analysis.update(extended_analysis)
                entry = {
                    'text': row['full_text'],
                    'preprocessed_text': analysis['text'],
                    'sentiment': analysis['sentiment'],
                    'confidence': analysis['confidence'],
                    'favorite_count': row['favorite_count'],
                    'created_at': row['created_at'],
                    'location': row['location'],
                    'username': row['username'],
                    'source_dataset': 'data_latih_1' if df is df1 else 'data_latih_2'
                }
                labeled_data.append(entry)

        df_labeled = pd.DataFrame(labeled_data)

        # Save to database if available
        if hasattr(self, 'db_manager') and self.db_manager.connection:
            # Insert new records into database
            for _, row in df_labeled.iterrows():
                self.db_manager.save_sentiment_data(
                    text=row['text'], 
                    preprocessed_text=row['preprocessed_text'],
                    sentiment=row['sentiment'],
                    confidence=row['confidence'],
                    favorite_count=row['favorite_count'],
                    created_at=row['created_at'],
                    location=row['location'],
                    username=row['username'],
                    source_dataset=row['source_dataset']
                )
        # Fallback to CSV storage
        elif hasattr(self, 'labeled_data_path'):
            df_labeled.to_csv(self.labeled_data_path, index=False)

        # Generate new rules based on labeled data
        self.rules = generate_rules_from_data(df_labeled)

        return df_labeled

    def apply_rules(self, analysis, favorite_count=0):
        result = {}

        # Apply confidence rules
        for level, rule in self.rules['confidence_rules'].items():
            if analysis['confidence'] >= rule['threshold']:
                result['confidence_level'] = rule['label']
                break

        # Apply sentiment classification
        sentiment = analysis['sentiment']
        if sentiment in self.rules['sentiment_classification']:
            for level, rule in self.rules['sentiment_classification'][sentiment].items():
                if analysis['confidence'] >= rule['confidence_threshold']:
                    result['sentiment_category'] = rule['label']
                    break

        # Apply popularity rules
        if 'popularity_rules' in self.rules:
            if favorite_count >= self.rules['popularity_rules'].get('viral', {}).get('threshold', 1000):
                result['popularity'] = self.rules['popularity_rules']['viral']['categories'][sentiment]
            elif favorite_count >= self.rules['popularity_rules'].get('popular', {}).get('min_threshold', 100):
                result['popularity'] = self.rules['popularity_rules']['popular']['label']
            else:
                result['popularity'] = self.rules['popularity_rules'].get('normal', {}).get('label', 'Normal')

        return result

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def generate_plots(self, df):
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distribusi Sentimen', 'Distribusi Confidence', 
                          'Jumlah Data per Dataset', 'Aktivitas Like')
        )

        # 1. Sentiment Distribution
        # Read labeled data
        labeled_df = pd.read_csv('static/data/labeled_data.csv')

        # Calculate sentiment distribution
        sentiment_counts = labeled_df['sentiment'].value_counts()
        colors = {'negative': '#e74c3c', 'neutral': '#3498db', 'positive': '#2ecc71'}
        fig.add_trace(
            go.Bar(x=sentiment_counts.index, y=sentiment_counts.values,
                  marker_color=[colors[sent] for sent in sentiment_counts.index],
                  name='Sentiment'),
            row=1, col=1
        )

        # 2. Confidence Distribution
        fig.add_trace(
            go.Histogram(x=df['confidence'], nbinsx=20,
                        marker_color='#2980b9',
                        name='Confidence'),
            row=1, col=2
        )

        # 3. Dataset Distribution
        dataset_counts = df['source_dataset'].value_counts()
        fig.add_trace(
            go.Bar(x=dataset_counts.index, y=dataset_counts.values,
                  marker_color='#9b59b6',
                  name='Dataset'),
            row=2, col=1
        )

        # 4. Likes Distribution by Sentiment
        fig.add_trace(
            go.Box(x=df['sentiment'], y=df['favorite_count'],
                  marker_color='#f1c40f',
                  name='Likes'),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return json.loads(fig.to_json())

    def generate_rules_visualization(self):
        dot = graphviz.Digraph(comment='Sentiment Rules')
        dot.attr(rankdir='TB')

        # Add nodes and edges based on rules
        for category, rules in self.rules.items():
            with dot.subgraph(name=f'cluster_{category}') as c:
                c.attr(label=category)
                for rule_name, rule_data in rules.items():
                    c.node(rule_name, rule_data.get('label', rule_name))
                    if 'threshold' in rule_data:
                        condition = f"confidence >= {rule_data['threshold']}"
                        c.edge('input', rule_name, label=condition)

        return dot.pipe().decode('utf-8')

def load_rules():
    try:
        # Try to load from database first
        db_manager = DatabaseManager()
        if db_manager.connect():
            rules = db_manager.load_rules()
            db_manager.disconnect()
            if rules:
                return rules

        # Fallback to file-based rules
        with open('model_rules.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "confidence_rules": {},
            "sentiment_classification": {},
            "popularity_rules": {
                "viral": {"threshold": 1000, "categories": {"positive": "Viral Positive", "neutral": "Viral Neutral", "negative": "Viral Negative"}},
                "popular": {"min_threshold": 100, "label": "Popular"},
                "normal": {"label": "Normal"}
            },
            "impact_rules": {}
        }

def generate_rules_from_data(df):
    rules = defaultdict(lambda: defaultdict(int))

    # Analyze patterns in data
    for _, row in df.iterrows():
        sentiment = row['sentiment']
        confidence = row['confidence']
        likes = row['favorite_count']

        # Count sentiment-confidence patterns
        conf_range = int(confidence * 10) / 10  # Round to nearest 0.1
        rules['confidence_patterns'][f"{sentiment}_{conf_range}"] += 1

        # Count likes patterns
        like_range = 'high' if likes >= 1000 else 'medium' if likes >= 100 else 'low'
        rules['likes_patterns'][f"{sentiment}_{like_range}"] += 1

    # Generate rules based on patterns
    rule_set = {
        "confidence_rules": {},
        "sentiment_classification": {},
        "popularity_rules": {},
        "impact_rules": {}
    }

    # Process confidence patterns
    for pattern, count in rules['confidence_patterns'].items():
        sentiment, conf = pattern.split('_')
        conf = float(conf)
        if count > 10:  # Only consider patterns with significant occurrence
            if conf >= 0.8:
                level = "very_high"
            elif conf >= 0.6:
                level = "high"
            elif conf >= 0.4:
                level = "medium"
            else:
                level = "low"

            rule_set["confidence_rules"][f"{sentiment}_{level}"] = {
                "threshold": conf,
                "count": count,
                "label": f"{level.replace('_', ' ').title()} {sentiment.title()}"
            }

    # Save generated rules
    with open('model_rules.json', 'w') as f:
        json.dump(rule_set, f, indent=4)

    # Generate human-readable rules
    human_rules = []
    for category, rules_dict in rule_set.items():
        human_rules.append(f"\nKATEGORI: {category.upper()}")
        for rule_name, rule_data in rules_dict.items():
            condition = f"confidence >= {rule_data['threshold']}" if 'threshold' in rule_data else ""
            human_rules.append(f"JIKA {condition} MAKA {rule_data['label']}")

    with open('sentiment_rules.txt', 'w') as f:
        f.write("\n".join(human_rules))

    return rule_set

app = Flask(__name__)
app.secret_key = 'sentiment_analyzer_secret_key'  # Digunakan untuk sesi

# Define login credentials
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'admin'

# Initialize database if it doesn't exist
from db_init import init_database
init_database()

# Initialize analyzer
analyzer = SentimentAnalyzer()

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('home'))
        else:
            error = 'Username atau password salah!'

    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/')
def home():
    try:
        # Check if user is logged in
        if not session.get('logged_in'):
            return redirect(url_for('login'))

        # Get or create labeled dataset
        df = analyzer.get_or_create_labeled_dataset()

        if df is not None:
            # Generate plots
            plots_json = analyzer.generate_plots(df)

            # Generate bar chart for positive, negative, and neutral sentiment counts
            sentiment_counts = df['sentiment'].value_counts()
            sentiment_percentages = (sentiment_counts / sentiment_counts.sum()) * 100

            # Always show all three sentiments
            sentiments = ['positive', 'negative', 'neutral']
            count_values = [sentiment_counts.get(s, 0) for s in sentiments]
            percent_values = [sentiment_percentages.get(s, 0) for s in sentiments]
            color_map = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#3498db'}
            percent_color_map = {'positive': '#27ae60', 'negative': '#c0392b', 'neutral': '#2980b9'}

            bar_chart = go.Figure(data=[
                go.Bar(
                    name='Jumlah Sentimen',
                    x=sentiments,
                    y=count_values,
                    marker_color=[color_map[s] for s in sentiments]
                ),
                go.Bar(
                    name='Persentase Sentimen',
                    x=sentiments,
                    y=percent_values,
                    marker_color=[percent_color_map[s] for s in sentiments]
                )
            ])
            bar_chart.update_layout(
                title='Distribusi Sentimen (Jumlah dan Persentase)',
                xaxis_title='Sentimen',
                yaxis_title='Nilai',
                barmode='group'
            )
            plots_json['sentiment_bar_chart'] = json.loads(bar_chart.to_json())

            # Prepare summary statistics
            stats = {
                'total_entries': len(df),
                'sentiment_counts': df['sentiment'].value_counts().to_dict(),
                'avg_confidence': df['confidence'].mean(),
                'avg_likes': df['favorite_count'].mean(),
                'dataset_counts': df['source_dataset'].value_counts().to_dict()
            }

            # Load evaluation result if exists
            eval_result = None
            try:
                with open('static/data/eval_result.json', 'r') as f:
                    eval_result = json.load(f)
            except Exception:
                eval_result = None

            # Load training results if exists
            training_results = None
            try:
                with open('static/data/training_results.json', 'r') as f:
                    training_results = json.load(f)
            except Exception:
                training_results = None

            return render_template('index.html',
                               plots=plots_json,
                               stats=stats,
                               data=df.to_dict('records'),
                               eval_result=eval_result,
                               training_results=training_results)
        else:
            return render_template('index.html',
                               plots=None,
                               stats=None,
                               data=None)
    except Exception as e:
        return str(e), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    try:
        # Check if user is logged in
        if not session.get('logged_in'):
            return jsonify({'error': 'Unauthorized access'}), 401

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        text = data.get('comment', '').strip()
        if not text:
            return jsonify({'error': 'No text provided for analysis'}), 400
            
        result = analyzer.analyze_text(text)
        return jsonify(result)
    except KeyError as e:
        return jsonify({'error': f'Missing required field: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/analyze-data', methods=['GET'])
def analyze_data():
    try:
        # Check if user is logged in
        if not session.get('logged_in'):
            return jsonify({'error': 'Unauthorized access'}), 401

        # Try to use existing labeled data first
        df = analyzer.get_or_create_labeled_dataset()

        if df is None:
            # If no labeled data exists, try to create from training data
            try:
                df1 = pd.read_csv('static/data/data_latih_1.csv')
                df2 = pd.read_csv('static/data/data_latih_2.csv')
                df = analyzer.create_labeled_dataset(df1, df2)
            except FileNotFoundError:
                # If no training data exists, create from default data
                df = analyzer.create_labeled_dataset()

        if df is not None:
            return jsonify({
                'success': True,
                'message': 'Analysis complete'
            })
        else:
            return jsonify({
                'error': 'No data available for analysis'
            }, 400)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/upload-dataset', methods=['POST'])
def upload_dataset():
    try:
        # Check if user is logged in
        if not session.get('logged_in'):
            return jsonify({'error': 'Unauthorized access'}), 401

        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files are allowed'}), 400

        # Save the uploaded file
        save_path = os.path.join('static', 'data', file.filename)
        file.save(save_path)

        return jsonify({
            'success': True,
            'message': 'File uploaded successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def build_c50_tree(X, y, feature_names, depth=0, max_depth=None):
    # X: numpy array (samples x features), y: labels, feature_names: list of str


    def entropy(labels):
        counts = np.bincount(labels)
        probs = counts / len(labels)
        return -np.sum([p * np.log2(p) for p in probs if p > 0])

    def split_info(sizes):
        total = sum(sizes)
        return -sum((s/total) * np.log2(s/total) for s in sizes if s > 0)

    def gain_ratio(X_col, y):
        values = np.unique(X_col)
        subsets = [y[X_col == v] for v in values]
        ent = entropy(y)
        weighted_ent = sum((len(sub)/len(y)) * entropy(sub) for sub in subsets)
        info_gain = ent - weighted_ent
        s_info = split_info([len(sub) for sub in subsets])
        return info_gain / s_info if s_info > 0 else 0

    # If all labels are the same, return leaf
    if len(set(y)) == 1:
        return {"class": int(y[0])}
    if max_depth is not None and depth >= max_depth:
        most_common = Counter(y).most_common(1)[0][0]
        return {"class": int(most_common)}

    # Find best feature to split
    best_feature = None
    best_gain = -1
    for i in range(X.shape[1]):
        gr = gain_ratio(X[:, i], y)
        if gr > best_gain:
            best_gain = gr
            best_feature = i

    if best_feature is None or best_gain == 0:
        most_common = Counter(y).most_common(1)[0][0]
        return {"class": int(most_common)}

    tree = {"feature": feature_names[best_feature], "splits": {}}
    values = np.unique(X[:, best_feature])
    for v in values:
        idx = X[:, best_feature] == v
        subtree = build_c50_tree(X[idx], y[idx], feature_names, depth+1, max_depth)
        tree["splits"][str(v)] = subtree
    return tree

@app.route('/api/rules-visualization')
def get_rules_visualization():
    try:
        # Check if user is logged in
        if not session.get('logged_in'):
            return jsonify({'error': 'Unauthorized access'}), 401

        rules_viz = analyzer.generate_rules_visualization()
        return jsonify({
            'success': True,
            'visualization': rules_viz
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/train-model', methods=['POST'])
def train_model():
    try:
        # Check if user is logged in
        if not session.get('logged_in'):
            return jsonify({'error': 'Unauthorized access'}), 401

        # Inisialisasi log tahapan proses pelatihan
        training_logs = []

        training_logs.append("Memulai proses pelatihan model...")

        data = request.get_json()
        split_percent = int(data.get('split_percent', 80))

        # Get advanced parameters from request
        tokenization_params = data.get('tokenization', {})
        vectorization_params = data.get('vectorization', {})
        decision_tree_params = data.get('decision_tree', {})
        training_logs.append(f"Parameter diterima: split data {split_percent}% latih, {100-split_percent}% uji")

        # Try to use database first
        db_manager = DatabaseManager()
        if db_manager.connect():
            training_logs.append("Memuat dataset dari database...")
            df = db_manager.fetch_all_data()
            if df is not None and not df.empty:
                training_logs.append(f"Dataset dimuat: {len(df)} data")

                training_logs.append("Mengacak dataset untuk memastikan distribusi yang merata...")
                df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

                train_size = int(len(df) * split_percent / 100)
                df_train = df.iloc[:train_size]
                df_test = df.iloc[train_size:]

                training_logs.append(f"Membagi dataset: {len(df_train)} data latih, {len(df_test)} data uji")

                # Save split information to database                training_logs.append("Menyimpan informasi split data ke database...")

                # Collect train and test IDs
                train_ids = df_train['id'].tolist()
                test_ids = df_test['id'].tolist()

                # Save train/test splits
                db_manager.save_dataset_split(train_ids, test_ids)

                # Also save to CSV for backward compatibility
                df_test.to_csv('static/data/test_data.csv', index=False)
                training_logs.append("Data split berhasil disimpan")

                db_manager.disconnect()
            else:
                training_logs.append("Database kosong, mencoba menggunakan file CSV...")
                db_manager.disconnect()
                # Fall back to CSV
                training_logs.append("Memuat dataset dari file CSV...")
                df = pd.read_csv('static/data/labeled_data.csv')
                training_logs.append(f"Dataset dimuat: {len(df)} data")

                training_logs.append("Mengacak dataset untuk memastikan distribusi yang merata...")
                df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

                train_size = int(len(df) * split_percent / 100)
                df_train = df.iloc[:train_size]
                df_test = df.iloc[train_size:]

                training_logs.append(f"Membagi dataset: {len(df_train)} data latih, {len(df_test)} data uji")

                # Save test set to CSV
                df_test.to_csv('static/data/test_data.csv', index=False)
                training_logs.append("Menyimpan data uji ke file")
        else:
            # Fall back to CSV if database connection fails
            training_logs.append("Koneksi database gagal, menggunakan file CSV...")
            df = pd.read_csv('static/data/labeled_data.csv')
            training_logs.append(f"Dataset dimuat: {len(df)} data")

            training_logs.append("Mengacak dataset untuk memastikan distribusi yang merata...")
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

            train_size = int(len(df) * split_percent / 100)
            df_train = df.iloc[:train_size]
            df_test = df.iloc[train_size:]

            training_logs.append(f"Membagi dataset: {len(df_train)} data latih, {len(df_test)} data uji")

            # Save test set
            df_test.to_csv('static/data/test_data.csv', index=False)
            training_logs.append("Menyimpan data uji ke file")

        # Text preprocessing based on parameters
        training_logs.append("Mulai tahap pra-pemrosesan teks...")

        # 1. Stopwords removal
        remove_stopwords_flag = tokenization_params.get('remove_stopwords', True)
        if remove_stopwords_flag:
            training_logs.append("Menghapus stopwords dari teks...")
            stopwords = set(ENGLISH_STOP_WORDS)
            def remove_stopwords(text):
                return " ".join([w for w in text.split() if w not in stopwords])
            df_train['preprocessed_text'] = df_train['preprocessed_text'].apply(remove_stopwords)
            df_test['preprocessed_text'] = df_test['preprocessed_text'].apply(remove_stopwords)
          # 2. Stemming (if enabled)
        use_stemming = tokenization_params.get('use_stemming', False)
        if use_stemming:
            training_logs.append("Menerapkan stemming pada kata...")
            from nltk.stem.porter import PorterStemmer
            stemmer = PorterStemmer()

            def stem_text(text):
                return " ".join([stemmer.stem(word) for word in text.split()])

            df_train['preprocessed_text'] = df_train['preprocessed_text'].apply(stem_text)
            df_test['preprocessed_text'] = df_test['preprocessed_text'].apply(stem_text)

        # 3. Lemmatization (if enabled)
        use_lemmatization = tokenization_params.get('use_lemmatization', False)
        if use_lemmatization:
            training_logs.append("Menerapkan lemmatization pada kata...")
            from nltk.stem import WordNetLemmatizer
            lemmatizer = WordNetLemmatizer()

            def lemmatize_text(text):
                return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

            df_train['preprocessed_text'] = df_train['preprocessed_text'].apply(lemmatize_text)
            df_test['preprocessed_text'] = df_test['preprocessed_text'].apply(lemmatize_text)        # Vectorization based on parameters
        vectorization_method = vectorization_params.get('method', 'count')
        max_features = int(vectorization_params.get('max_features', 2000))

        training_logs.append(f"Melakukan vektorisasi dengan metode {vectorization_method}, max features: {max_features}...")
        if vectorization_method == 'tfidf':
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=max_features)
        else:  # default to CountVectorizer
            vectorizer = CountVectorizer(max_features=max_features)

        X_train = vectorizer.fit_transform(df_train['preprocessed_text'])
        y_train = df_train['sentiment']

        X_test = vectorizer.transform(df_test['preprocessed_text'])
        y_test = df_test['sentiment']

        training_logs.append(f"Jumlah fitur setelah vektorisasi: {len(vectorizer.get_feature_names_out())}")

        # Decision Tree parameters
        max_depth_str = decision_tree_params.get('max_depth', 'none')
        max_depth = None if max_depth_str == 'none' else int(max_depth_str)

        min_samples_split = int(decision_tree_params.get('min_samples_split', 2))
        min_samples_leaf = int(decision_tree_params.get('min_samples_leaf', 1))
        criterion = decision_tree_params.get('criterion', 'gini')

        training_logs.append(f"Parameter Decision Tree: max_depth={max_depth_str}, criterion={criterion}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}")

        # Create and train the model
        training_logs.append("Mulai pelatihan model decision tree...")
        clf = DecisionTreeClassifier(
            random_state=42, 
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion
        )
        clf.fit(X_train, y_train)
        training_logs.append("Pelatihan model selesai")        # Evaluate the model
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        training_logs.append(f"Evaluasi model pada data uji: akurasi = {accuracy:.4f} ({accuracy*100:.2f}%)")

        # Generate a timestamp-based model name
        model_name = f"sentiment_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Save model to database if available
        db_manager = DatabaseManager()
        if db_manager.connect():
            training_logs.append("Menyimpan model ke database...")
            # Serialize model and vectorizer
            model_saved = db_manager.save_model(
                model_name=model_name,
                model_obj=clf,
                vectorizer_obj=vectorizer,
                accuracy=accuracy
            )

            if model_saved:
                training_logs.append("Model berhasil disimpan ke database")
            else:
                training_logs.append("Gagal menyimpan model ke database, mencoba menyimpan ke disk...")
                # Fall back to file-based storage
                with open('static/data/sentiment_model.pkl', 'wb') as f:
                    pickle.dump({'model': clf, 'vectorizer': vectorizer}, f)
                training_logs.append("Model berhasil disimpan ke disk")

            db_manager.disconnect()
        else:
            # Fall back to file-based storage
            training_logs.append("Koneksi database gagal, menyimpan model ke disk...")
            with open('static/data/sentiment_model.pkl', 'wb') as f:
                pickle.dump({'model': clf, 'vectorizer': vectorizer}, f)
            training_logs.append("Model berhasil disimpan ke disk")        # Save rules as JSON using manual C5.0-like function
        training_logs.append("Membuat struktur pohon keputusan C5.0...")
        sentiment_labels = sorted(y_train.unique())
        label_to_int = {label: i for i, label in enumerate(sentiment_labels)}
        int_to_label = {i: label for label, i in label_to_int.items()}

        y_train_int = np.array([label_to_int[y] for y in y_train])
        
        # Check if converting to dense array will cause memory issues
        # Calculate memory requirement: rows * cols * 8 bytes (int64)
        memory_required_mb = (X_train.shape[0] * X_train.shape[1] * 8) / (1024 * 1024)
        
        if memory_required_mb > 100:  # If more than 100MB required
            training_logs.append(f"Dataset terlalu besar untuk membuat pohon C5.0 (memerlukan {memory_required_mb:.1f}MB), menggunakan scikit-learn tree sebagai gantinya...")
            # Use simplified tree rules from scikit-learn
            c50_tree = {"message": "Tree too large for C5.0 format", "sklearn_tree_used": True}
        else:
            X_train_arr = X_train.toarray()
            feature_names = vectorizer.get_feature_names_out().tolist()
            c50_tree = build_c50_tree(X_train_arr, y_train_int, feature_names)
        
        training_logs.append("Struktur pohon keputusan berhasil dibuat")        # Recursively convert integer classes to sentiment labels in the tree
        def convert_classes(tree):
            if "class" in tree:
                tree["class"] = int_to_label.get(tree["class"], str(tree["class"]))
            if "splits" in tree:
                for k, v in tree["splits"].items():
                    convert_classes(v)
            return tree

        # Only convert classes if we have a proper tree structure
        if "sklearn_tree_used" not in c50_tree:
            c50_tree = convert_classes(c50_tree)
        
        with open('static/data/rules.json', 'w') as f:
            json.dump(c50_tree, f, indent=2)        # Save human-readable rules as TXT
        def rules_to_text(tree, indent=0):
            pad = "    " * indent
            if "sklearn_tree_used" in tree:
                return "Pohon keputusan terlalu besar untuk ditampilkan dalam format C5.0.\nGunakan visualisasi SVG untuk melihat struktur pohon."
            elif "class" in tree:
                return pad + f"MAKA Sentimen = {tree['class']}\n"
            elif "feature" in tree and "splits" in tree:
                lines = []
                splits = list(tree["splits"].items())
                for i, (val, subtree) in enumerate(splits):
                    if i == 0:
                        lines.append(pad + f"JIKA '{tree['feature']}' = '{val}'")
                    else:
                        lines.append(pad + f"JIKA TIDAK, JIKA '{tree['feature']}' = '{val}'")
                    lines.append(rules_to_text(subtree, indent + 1))
                return "\n".join(lines)
            return ""
        
        rules_txt = rules_to_text(c50_tree)
        with open('sentiment_rules.txt', 'w', encoding='utf-8') as f:
            f.write(rules_txt)
        training_logs.append("Aturan keputusan disimpan dalam format teks")

        # Generate and save visualization as image (SVG) using matplotlib
        training_logs.append("Membuat visualisasi pohon keputusan...")

        # Create the decision tree visualization
        plt.figure(figsize=(40, 20))
        sktree.plot_tree(
            clf,
            feature_names=vectorizer.get_feature_names_out().tolist(),
            class_names=[str(c) for c in clf.classes_],
            filled=True,
            rounded=True,
            fontsize=14,
            proportion=False,  # ensure nodes are spaced out
            precision=2
        )
        plt.tight_layout()

        # Dynamically set SVG size based on tree size
        depth = clf.get_depth()
        leaves = clf.get_n_leaves()
        width = max(20, leaves * 2)
        height = max(10, depth * 3)
        fig = plt.gcf()
        fig.set_size_inches(width, height)
          # Save to static directory for direct access via URL
        plt.savefig("static/sentiment_rules_tree.svg", format="svg", bbox_inches="tight")
        plt.close()

        # Save a copy to static folder for compatibility
        plt.figure(figsize=(40, 20))
        sktree.plot_tree(
            clf,
            feature_names=vectorizer.get_feature_names_out().tolist(),
            class_names=[str(c) for c in clf.classes_],
            filled=True,
            rounded=True,
            fontsize=14,
            proportion=False,
            precision=2
        )
        plt.savefig("static/data/sentiment_rules_tree.svg", format="svg", bbox_inches="tight")
        plt.close()

        # Evaluate on test set
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
        report = classification_report(y_test, y_pred, output_dict=True, labels=clf.classes_)        # Save confusion matrix and metrics
        eval_result = {
            'confusion_matrix': cm.tolist(),
            'labels': list(clf.classes_),
            'classification_report': report
        }
        with open('static/data/eval_result.json', 'w') as f:
            json.dump(eval_result, f, indent=2)

        training_logs.append("Proses pelatihan selesai!")
        training_logs.append(f"Akurasi model: {accuracy*100:.2f}%")

        # Save complete training results as JSON for frontend display
        training_results = {
            'success': True, 
            'eval_result': eval_result,
            'accuracy': float(accuracy),
            'training_logs': training_logs
        }

        with open('static/data/training_results.json', 'w') as f:
            json.dump(training_results, f, indent=2)

        return jsonify(training_results)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/data-distribution', methods=['GET'])
def get_data_distribution():
    try:
        # Check if user is logged in
        if not session.get('logged_in'):
            return jsonify({'error': 'Unauthorized access'}), 401

        total_data = 0
        train_size = 0
        test_size = 0
        db_processed_successfully = False

        db_manager = DatabaseManager()
        if db_manager.connect():
            summary = db_manager.get_data_split_summary()
            db_manager.disconnect() # Disconnect after use

            if summary:
                total_data = summary['total_data']
                train_size = summary['train_data']
                test_size = summary['test_data']

                # If dataset_splits table is empty (or doesn't reflect splits) 
                # but texts table has data, consider all data as 'train' for this view's purpose
                # if no explicit split is defined.
                if total_data > 0 and train_size == 0 and test_size == 0:
                    train_size = total_data 
                    test_size = 0

                # Ensure consistency, though counts should be correct from summary
                if train_size + test_size > total_data and test_size > 0 : # If test data exists, adjust train
                    train_size = total_data - test_size
                    if train_size < 0: train_size = 0 # Ensure non-negative
                elif train_size + test_size > total_data and train_size > 0: # If train data exists, adjust test (less likely scenario)
                    test_size = total_data - train_size
                    if test_size < 0: test_size = 0 # Ensure non-negative


                db_processed_successfully = True

        if not db_processed_successfully:
            # Fallback to CSV if database connection failed or summary was None
            if os.path.exists('static/data/labeled_data.csv'):
                df_all = pd.read_csv('static/data/labeled_data.csv')
                total_data = len(df_all)

                if os.path.exists('static/data/test_data.csv'):
                    df_test = pd.read_csv('static/data/test_data.csv')
                    test_size = len(df_test)
                    # Ensure train_size is calculated correctly and non-negative
                    train_size = total_data - test_size
                    if train_size < 0:
                        train_size = 0 # Avoid negative train_size if test_data is larger than labeled_data
                        total_data = test_size # Adjust total_data if test_data is the only source of truth for size
                else:
                    # If test_data.csv does not exist, assume all labeled_data is for training
                    train_size = total_data
                    test_size = 0
            # If labeled_data.csv also doesn't exist, total_data, train_size, test_size remain 0.

        return jsonify({
            'total_data': total_data,
            'train_size': train_size,
            'test_size': test_size,
            'train_percent': round((train_size / total_data * 100) if total_data > 0 else 0, 1),
            'test_percent': round((test_size / total_data * 100) if total_data > 0 else 0, 1)
        })
    except Exception as e:
        print(f"Error in /api/data-distribution: {e}") # Log the exception server-side
        return jsonify({'error': str(e)}), 400

@app.route('/api/test-data-distribution', methods=['GET'])
def get_test_data_distribution():
    try:
        # Check if user is logged in
        if not session.get('logged_in'):
            return jsonify({'error': 'Unauthorized access'}), 401

        # Check if test data exists
        if os.path.exists('static/data/test_data.csv'):
            df_test = pd.read_csv('static/data/test_data.csv')

            # Calculate sentiment distribution
            sentiment_counts = df_test['sentiment'].value_counts().to_dict()

            # Calculate percentages
            total = sum(sentiment_counts.values())
            sentiment_percentages = {k: round(v / total * 100, 1) for k, v in sentiment_counts.items()}

            # Ensure all sentiments are represented
            sentiments = ['positive', 'negative', 'neutral']
            for sentiment in sentiments:
                if sentiment not in sentiment_counts:
                    sentiment_counts[sentiment] = 0
                    sentiment_percentages[sentiment] = 0

            return jsonify({
                'total_data': total,
                'sentiment_counts': sentiment_counts,
                'sentiment_percentages': sentiment_percentages
            })
        else:
            return jsonify({
                'error': 'Test data file not found'
            }, 404)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/open-excel-csv/<path:filename>')
def open_excel_csv(filename):
    try:
        # Check if user is logged in
        if not session.get('logged_in'):
            return jsonify({'error': 'Unauthorized access'}), 401

        # Mendapatkan full path file CSV
        file_path = os.path.join(os.getcwd(), 'static', 'data', filename)

        # Pastikan file ada
        if not os.path.exists(file_path):
            return jsonify({'error': f'File {filename} tidak ditemukan'}), 404

        # Pastikan file adalah CSV
        if not filename.endswith('.csv'):
            return jsonify({'error': 'Hanya file CSV yang dapat dibuka dengan Excel'}), 400

        # Menggunakan os.system untuk membuka Excel dengan file CSV
        cmd = f'start excel.exe "{file_path}"'
        os.system(cmd)

        return jsonify({
            'success': True,
            'message': f'File {filename} sedang dibuka dengan Excel'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stream-analyze-data', methods=['POST'])
def stream_analyze_data():
    try:
        # Check if user is logged in
        if not session.get('logged_in'):
            return jsonify({'error': 'Unauthorized access'}), 401

        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files are allowed'}), 400

        # Save the uploaded file
        file_path = os.path.join('static', 'data', file.filename)
        file.save(file_path)

        # Stream the processing logs
        def generate_logs():
            yield "data: Memulai proses analisis data...\n\n"
            try:
                # Load and preprocess the data
                df = pd.read_csv(file_path)
                yield "data: Dataset dimuat, mulai pra-pemrosesan...\n\n"

                # Preprocess text column
                df['preprocessed_text'] = df['full_text'].apply(analyzer.preprocess_text)
                yield "data: Pra-pemrosesan teks selesai\n\n"

                # Analyze sentiment
                df['sentiment'] = df['preprocessed_text'].apply(lambda x: analyzer.analyze_text(x)['sentiment'])
                yield "data: Analisis sentimen selesai\n\n"

                # Calculate confidence scores
                df['confidence'] = df['preprocessed_text'].apply(lambda x: analyzer.analyze_text(x)['confidence'])
                yield "data: Perhitungan skor kepercayaan selesai\n\n"

                # Save the labeled data
                df.to_csv(analyzer.labeled_data_path, index=False)
                yield "data: Data berhasil disimpan\n\n"

                # Generate plots
                plots_json = analyzer.generate_plots(df)
                yield f"data: {json.dumps(plots_json)}\n\n"

                # Prepare summary statistics
                stats = {
                    'total_entries': len(df),
                    'sentiment_counts': df['sentiment'].value_counts().to_dict(),
                    'avg_confidence': df['confidence'].mean(),
                    'avg_likes': df['favorite_count'].mean(),
                    'dataset_counts': df['source_dataset'].value_counts().to_dict()
                }
                yield f"data: {json.dumps(stats)}\n\n"

                # Load evaluation result if exists
                eval_result = None
                try:
                    with open('static/data/eval_result.json', 'r') as f:
                        eval_result = json.load(f)
                except Exception:
                    eval_result = None
                yield f"data: {json.dumps(eval_result)}\n\n"

                yield "data: Proses analisis data selesai\n\n"
            except Exception as e:
                yield f"data: Error: {str(e)}\n\n"

        return Response(stream_with_context(generate_logs()), content_type='text/event-stream')
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/preview-dataset', methods=['POST'])
def preview_dataset():
    try:
        # Check if user is logged in
        if not session.get('logged_in'):
            return jsonify({'error': 'Unauthorized access'}), 401

        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files are allowed'}), 400

        # Save the uploaded file temporarily
        temp_path = os.path.join('static', 'data', 'temp_upload.csv')
        file.save(temp_path)

        # Read the CSV file
        try:
            df = pd.read_csv(temp_path, encoding='utf-8')
        except Exception as e:
            try:
                # Try another encoding if utf-8 fails
                df = pd.read_csv(temp_path, encoding='latin1')
            except Exception as e:
                return jsonify({'error': f'Error reading CSV file: {str(e)}'}), 400

        # Check if required column exists
        if 'full_text' not in df.columns:
            return jsonify({'error': 'The CSV file must contain a "full_text" column with the text to analyze'}), 400

        # Check for missing columns
        missing_columns = []
        required_columns = ['full_text', 'favorite_count', 'created_at', 'location', 'username']
        for col in required_columns:
            if col != 'full_text' and (col not in df.columns or df[col].isna().all()):
                missing_columns.append(col)

        # Get a preview of the data (10 rows)
        preview_df = df.head(10).copy()

        # Convert DataFrame to dictionary for JSON serialization
        preview_data = preview_df.fillna('').to_dict('records')

        return jsonify({
            'success': True,
            'total_rows': len(df),
            'columns': df.columns.tolist(),
            'preview_data': preview_data,
            'missing_columns': missing_columns,
            'message': 'Preview generated successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/process-dataset', methods=['POST'])
def process_dataset():
    # Check if user is logged in
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized access'}), 401

    # Define the streaming response
    def generate():
        try:
            # Load the temporary uploaded file
            temp_path = os.path.join('static', 'data', 'temp_upload.csv')

            if not os.path.exists(temp_path):
                yield json.dumps({'log': 'Error: No uploaded file found. Please upload a file first.', 'type': 'error'}) + '\n'
                return

            # Read the CSV file
            try:
                df = pd.read_csv(temp_path, encoding='utf-8')
            except Exception:
                try:
                    df = pd.read_csv(temp_path, encoding='latin1')
                except Exception as e:
                    yield json.dumps({'log': f'Error reading CSV file: {str(e)}', 'type': 'error'}) + '\n'
                    return

            # Check if required column exists
            if 'full_text' not in df.columns:
                yield json.dumps({'log': 'Error: The CSV file must contain a "full_text" column with the text to analyze', 'type': 'error'}) + '\n'
                return

            # Report missing columns but continue
            missing_cols = []
            for col in ['favorite_count', 'created_at', 'location', 'username']:
                if col not in df.columns:
                    missing_cols.append(col)
                    # Add empty column
                    df[col] = ''

            if missing_cols:
                yield json.dumps({'log': f'Info: Adding missing columns: {", ".join(missing_cols)}', 'type': 'info'}) + '\n'

            # Function to send progress updates
            def send_progress(percent, message, msg_type='info'):
                nonlocal progress_data
                progress_data = json.dumps({
                    'progress': percent,
                    'log': message,
                    'type': msg_type
                }) + '\n'            # Process the dataset with database saving
            progress_data = ''
            
            # Use a streaming approach that saves to database during processing
            total_rows = len(df)
            batch_size = min(50, max(10, total_rows // 20))
            processed_data = []
            saved_to_db_count = 0
              # Check if database is available
            use_database = hasattr(analyzer, 'db_manager') and hasattr(analyzer, 'db_connected') and analyzer.db_connected and analyzer.db_manager.connection
            if use_database:
                yield json.dumps({'log': 'Database connection available, saving data during processing...', 'type': 'info'}) + '\n'
            else:
                yield json.dumps({'log': 'Database not available, will save to CSV file...', 'type': 'info'}) + '\n'
            
            # Process in batches
            for i in range(0, total_rows, batch_size):
                batch_df = df.iloc[i:i+batch_size].copy()
                
                # Extract text and favorite count for batch processing
                texts = batch_df['full_text'].tolist()
                favorite_counts = [0] * len(texts)
                
                if 'favorite_count' in batch_df.columns:
                    favorite_counts = batch_df['favorite_count'].fillna(0).tolist()
                
                # Update progress
                progress_percent = int((i / total_rows) * 80)  # Reserve 80% for processing, 20% for saving
                send_progress(progress_percent, f"Processing batch {i//batch_size + 1}/{(total_rows+batch_size-1)//batch_size}...")
                yield progress_data
                
                # Process batch using the analyzer
                from labeling import analyze_text_batch
                batch_results = analyze_text_batch(analyzer, texts, favorite_counts)
                
                # Process each row in the batch
                for j, row in batch_df.iterrows():
                    idx = j - i  # Index within the batch
                    result = batch_results[idx]
                    
                    if result is None or 'error' in result:
                        continue
                    
                    # Create entry with all required fields
                    entry = {
                        'text': row['full_text'],
                        'preprocessed_text': result['text'],
                        'sentiment': result['sentiment'],
                        'confidence': result['confidence'],
                        'favorite_count': int(row.get('favorite_count', 0)) if pd.notna(row.get('favorite_count', 0)) else 0,
                        'created_at': row.get('created_at') if pd.notna(row.get('created_at')) else None,
                        'location': row.get('location') if pd.notna(row.get('location')) else None,
                        'username': row.get('username') if pd.notna(row.get('username')) else None,
                        'source_dataset': 'uploaded_data'
                    }
                    
                    # Save to database immediately if available
                    if use_database:
                        success = analyzer.db_manager.save_sentiment_data(
                            text=entry['text'],
                            preprocessed_text=entry['preprocessed_text'],
                            sentiment=entry['sentiment'],
                            confidence=entry['confidence'],
                            favorite_count=entry['favorite_count'],
                            created_at=entry['created_at'],
                            location=entry['location'],
                            username=entry['username'],
                            source_dataset=entry['source_dataset']
                        )
                        if success:
                            saved_to_db_count += 1
                    
                    # Also keep in memory for CSV fallback and stats
                    processed_data.append(entry)
            
            # Convert to DataFrame for statistics and CSV fallback
            output_df = pd.DataFrame(processed_data)
            
            # Update progress to 90%
            send_progress(90, "Finalizing data processing...")
            yield progress_data
            
            # Final save operations
            if use_database:
                yield json.dumps({'log': f'Successfully saved {saved_to_db_count} records to database!', 'type': 'success'}) + '\n'
                # Also save CSV for compatibility
                labeled_data_path = os.path.join('static', 'data', 'labeled_data.csv')
                output_df.to_csv(labeled_data_path, index=False)
                yield json.dumps({'log': 'Data also saved to CSV file for compatibility', 'type': 'info'}) + '\n'
            else:
                # Fallback to CSV if database is not available
                labeled_data_path = os.path.join('static', 'data', 'labeled_data.csv')
                output_df.to_csv(labeled_data_path, index=False)
                yield json.dumps({'log': f'Data saved to CSV file ({len(output_df)} records)', 'type': 'info'}) + '\n'

            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

            # Regenerate rules based on new data
            analyzer.rules = generate_rules_from_data(output_df)

            # Calculate sentiment statistics
            sentiment_counts = output_df['sentiment'].value_counts().to_dict()

            # Ensure all sentiment types are represented
            for sentiment in ['positive', 'negative', 'neutral']:
                if sentiment not in sentiment_counts:
                    sentiment_counts[sentiment] = 0

            # Create statistics object
            stats = {
                'total_entries': len(output_df),
                'sentiment_counts': sentiment_counts
            }

            # Send statistics
            yield json.dumps({
                'stats': stats
            }) + '\n'

            yield json.dumps({
                'progress': 100,
                'log': f'Pelabelan selesai! Data telah disimpan sebagai labeled_data.csv dengan {len(output_df)} baris.',
                'type': 'success'
            }) + '\n'

        except Exception as e:
            yield json.dumps({
                'progress': 0,
                'log': f'Error dalam pemrosesan: {str(e)}',
                'type': 'error'
            }) + '\n'

    # Return the streaming response
    return Response(stream_with_context(generate()), content_type='application/json')

@app.route('/api/preview-file-split', methods=['POST'])
def preview_file_split():
    try:
        data = request.get_json()
        filename = data.get('filename')
        split_percentage_str = data.get('split_percentage')

        if not filename or split_percentage_str is None:
            return jsonify({"error": "Missing filename or split_percentage"}), 400

        try:
            split_percentage = float(split_percentage_str)
            if not (0 <= split_percentage <= 100):
                raise ValueError("Split percentage must be between 0 and 100.")
        except ValueError as e:
            return jsonify({"error": f"Invalid split_percentage: {str(e)}"}), 400

        # Construct the full path to the file within the 'static/data/' directory
        # Ensure app.static_folder is correctly configured if not using the default 'static'
        file_path = os.path.join(app.static_folder, 'data', filename)

        if not os.path.exists(file_path):
            # Try to construct path relative to app.root_path if static_folder is not directly applicable
            # This is a fallback, ensure your file serving strategy is consistent
            current_dir_path = os.path.join(app.root_path, 'static', 'data', filename)
            if not os.path.exists(current_dir_path):
                 return jsonify({"error": f"File not found: {filename}. Checked: {file_path} and {current_dir_path}"}), 404
            file_path = current_dir_path


        try:
            df = pd.read_csv(file_path, on_bad_lines='skip')
        except pd.errors.EmptyDataError:
            # For an empty file, return 0 rows but reflect the requested split percentages
            return jsonify({
                "total_rows": 0,
                "train_rows": 0,
                "test_rows": 0,
                "train_percent": split_percentage,
                "test_percent": 100 - split_percentage,
                "message": "The file is empty or not a valid CSV."
            }), 200
        except Exception as e:
            return jsonify({"error": f"Error reading CSV file: {str(e)}"}), 500

        total_rows = len(df)

        if total_rows == 0:
            # If CSV has headers but no data rows
            return jsonify({
                "total_rows": 0,
                "train_rows": 0,
                "test_rows": 0,
                "train_percent": split_percentage, # Reflect requested split
                "test_percent": 100 - split_percentage, # Reflect requested split
                "message": "The file contains no data rows (it might only have headers)."
            }), 200

        train_rows = int(round(total_rows * (split_percentage / 100.0)))
        test_rows = total_rows - train_rows

        # Calculate actual percentages based on potentially rounded row counts
        actual_train_percent = (train_rows / total_rows) * 100 if total_rows > 0 else 0
        actual_test_percent = (test_rows / total_rows) * 100 if total_rows > 0 else 0

        return jsonify({
            "total_rows": total_rows,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "train_percent": actual_train_percent,
            "test_percent": actual_test_percent
        })

    except Exception as e:
        # Log the exception e for server-side debugging
        app.logger.error(f"Unexpected error in /api/preview-file-split: {str(e)}")
        return jsonify({"error": "An unexpected error occurred on the server.", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) # Ensure app.run is correctly configured
