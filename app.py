from flask import Flask, request, jsonify, render_template
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

class SentimentAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("w11wo/indonesian-roberta-base-sentiment-classifier")
        self.model = AutoModelForSequenceClassification.from_pretrained("w11wo/indonesian-roberta-base-sentiment-classifier")
        self.model.eval()
        self.labeled_data_path = 'static/data/labeled_data.csv'

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
        
        return {
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence
        }

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def create_labeled_dataset(self, df1, df2):
        labeled_data = []
        
        for df in [df1, df2]:
            for _, row in df.iterrows():
                if pd.isna(row['full_text']):
                    continue
                
                analysis = self.analyze_text(row['full_text'])
                
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
        df_labeled.to_csv(self.labeled_data_path, index=False)
        
        return df_labeled

    def generate_plots(self, df):
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distribusi Sentimen', 'Distribusi Confidence', 
                          'Jumlah Data per Dataset', 'Aktivitas Like')
        )

        # 1. Sentiment Distribution
        sentiment_counts = df['sentiment'].value_counts()
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

        # Update layout with better formatting
        fig.update_layout(
            height=800,
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Convert to JSON while preserving the Plotly figure structure
        plotly_json = json.loads(fig.to_json())
        return json.dumps(plotly_json)

app = Flask(__name__)

# Initialize analyzer
analyzer = SentimentAnalyzer()

@app.route('/')
def home():
    try:
        if os.path.exists(analyzer.labeled_data_path):
            df = pd.read_csv(analyzer.labeled_data_path)
            
            # Generate plots
            plots_json = analyzer.generate_plots(df)
            
            # Prepare summary statistics
            stats = {
                'total_entries': len(df),
                'sentiment_counts': df['sentiment'].value_counts().to_dict(),
                'avg_confidence': df['confidence'].mean(),
                'avg_likes': df['favorite_count'].mean(),
                'dataset_counts': df['source_dataset'].value_counts().to_dict()
            }
            
            return render_template('index.html', 
                                plots=plots_json,
                                stats=stats,
                                data=df.to_dict('records'))
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
        data = request.json
        text = data['comment']
        result = analyzer.analyze_text(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/analyze-data', methods=['GET'])
def analyze_data():
    try:
        df1 = pd.read_csv('static/data/data_latih_1.csv')
        df2 = pd.read_csv('static/data/data_latih_2.csv')
        df_labeled = analyzer.create_labeled_dataset(df1, df2)
        return jsonify({
            'success': True,
            'message': 'Analysis complete'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)