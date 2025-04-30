from flask import Flask, request, jsonify, render_template
import json
import math
import re
from collections import Counter
from flask_cors import CORS
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
CORS(app)

class SentimentAnalyzer:
    def __init__(self):
        self.rules_file = 'model_rules.json'
        self.stop_words = {
            'yang', 'di', 'ke', 'dari', 'pada', 'dalam', 'untuk', 'dengan', 
            'dan', 'atau', 'ini', 'itu', 'juga', 'sudah', 'saya', 'anda', 
            'dia', 'mereka', 'kita', 'akan', 'bisa', 'ada', 'tidak', 'saat',
            'oleh', 'setelah', 'para', 'sampai', 'ketika', 'seperti', 'bagi'
        }
        
        self.kata_positif = {
            'bagus': 1.5, 'mantap': 1.8, 'suka': 1.3, 'puas': 1.6, 
            'recommended': 1.7, 'keren': 1.5, 'sempurna': 2.0, 'terbaik': 1.8,
            'memuaskan': 1.4, 'perfect': 1.9, 'worth': 1.5, 'ok': 1.2
        }
        
        self.kata_negatif = {
            'buruk': -1.5, 'jelek': -1.4, 'kecewa': -1.8, 'rusak': -1.6,
            'mahal': -1.3, 'lambat': -1.4, 'palsu': -2.0, 'jangan': -1.2,
            'mengecewakan': -1.9, 'tidak': -1.0, 'gagal': -1.7, 'rugi': -1.5
        }
        
        self.tree_rules = {'rules': [], 'feature_weights': {}}
        self.load_rules()
        
    def tokenize(self, text):
        # Case folding
        text = text.lower()
        # Hapus karakter khusus
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # Split ke tokens
        tokens = text.split()
        # Hapus stop words
        tokens = [token for token in tokens if token not in self.stop_words]
        return tokens

    def extract_features(self, text, metrics):
        tokens = self.tokenize(text)
        
        # Hitung skor sentimen dari kata-kata
        skor_positif = sum(self.kata_positif.get(word, 0) for word in tokens)
        skor_negatif = sum(self.kata_negatif.get(word, 0) for word in tokens)
        
        # Hitung metrik engagement
        likes_weight = math.log1p(metrics['likes']) if metrics['likes'] > 0 else 0
        dislikes_weight = -math.log1p(metrics['dislikes']) if metrics['dislikes'] > 0 else 0
        
        features = {
            'skor_positif': skor_positif,
            'skor_negatif': skor_negatif,
            'jumlah_token': len(tokens),
            'jumlah_tanda_seru': text.count('!'),
            'jumlah_tanda_tanya': text.count('?'),
            'skor_likes': likes_weight,
            'skor_dislikes': dislikes_weight,
            'sentiment_score': skor_positif + skor_negatif + likes_weight + dislikes_weight
        }
        
        return features, tokens

    def train(self, training_data):
        processed_data = []
        for item in training_data:
            features, tokens = self.extract_features(item['text'], item['metrics'])
            processed_data.append({
                'text': item['text'],
                'tokens': tokens,
                'features': features,
                'sentiment': item['sentiment']
            })
        
        self.generate_rules(processed_data)
        self.save_rules()
        return self.visualize_tree(processed_data)

    def generate_rules(self, data):
        self.tree_rules['rules'] = []
        
        # Generate rules based on sentiment scores
        for item in data:
            score = item['features']['sentiment_score']
            
            if score > 2:
                sentiment = 'positive'
            elif score < -1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
                
            rule = {
                'score_range': [score-0.5, score+0.5],
                'predicted_sentiment': sentiment,
                'actual_sentiment': item['sentiment'],
                'features': item['features']
            }
            self.tree_rules['rules'].append(rule)

    def save_rules(self):
        with open(self.rules_file, 'w') as f:
            json.dump(self.tree_rules, f, indent=2)

    def load_rules(self):
        try:
            with open(self.rules_file, 'r') as f:
                self.tree_rules = json.load(f)
                return True
        except FileNotFoundError:
            return False

    def visualize_tree(self, data):
        plt.figure(figsize=(15, 10))
        
        # Create scatter plot of sentiment scores
        scores = [item['features']['sentiment_score'] for item in data]
        sentiments = [item['sentiment'] for item in data]
        
        # Sort data points for better visualization
        sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k])
        scores = [scores[i] for i in sorted_indices]
        sentiments = [sentiments[i] for i in sorted_indices]
        
        # Plot points with different colors based on sentiment
        sentiment_colors = {
            'positive': '#2ecc71',  # Green
            'negative': '#e74c3c',  # Red
            'neutral': '#3498db'    # Blue
        }
        colors = [sentiment_colors[s] for s in sentiments]
        
        plt.scatter(range(len(scores)), scores, c=colors, s=100)
        
        # Add decision boundaries
        plt.axhline(y=2, color='g', linestyle='--', label='Positive threshold')
        plt.axhline(y=-1, color='r', linestyle='--', label='Negative threshold')
        
        # Customize plot
        plt.title('Sentiment Distribution and Decision Boundaries', fontsize=14, pad=20)
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Sentiment Score', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        
        # Add sentiment zones
        plt.fill_between(plt.xlim(), 2, plt.ylim()[1], alpha=0.1, color='g', label='Positive Zone')
        plt.fill_between(plt.xlim(), plt.ylim()[0], -1, alpha=0.1, color='r', label='Negative Zone')
        plt.fill_between(plt.xlim(), -1, 2, alpha=0.1, color='b', label='Neutral Zone')
        
        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        buf.seek(0)
        plt.close()
        
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def predict(self, text, metrics):
        features, tokens = self.extract_features(text, metrics)
        score = features['sentiment_score']
        
        matching_rules = []
        for rule in self.tree_rules['rules']:
            if rule['score_range'][0] <= score <= rule['score_range'][1]:
                matching_rules.append(rule)
        
        if score > 2:
            sentiment = 'positive'
        elif score < -1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
            
        return {
            'sentiment': sentiment,
            'features': features,
            'tokens': tokens,
            'score': score,
            'matching_rules': matching_rules
        }

# Initialize analyzer
analyzer = SentimentAnalyzer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        with open('training_data.json', 'r') as f:
            training_data = json.load(f)['data']
        
        tree_image = analyzer.train(training_data)
        
        return jsonify({
            'success': True,
            'message': 'Model berhasil dilatih',
            'tree_visualization': tree_image,
            'rules': analyzer.tree_rules
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        text = data['comment']
        metrics = {
            'likes': int(data.get('likes', 0)),
            'dislikes': int(data.get('dislikes', 0))
        }
        
        result = analyzer.predict(text, metrics)
        
        return jsonify({
            'tokens': result['tokens'],
            'features': result['features'],
            'sentiment': result['sentiment'],
            'score': result['score'],
            'matching_rules': result['matching_rules']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/rules', methods=['GET'])
def get_rules():
    return jsonify(analyzer.tree_rules)

if __name__ == '__main__':
    app.run(debug=True)