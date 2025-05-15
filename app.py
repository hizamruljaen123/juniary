from flask import Flask, request, jsonify, render_template, send_file
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

class SentimentAnalyzer:
    def __init__(self):
        self.rules = load_rules()
        self.tokenizer = AutoTokenizer.from_pretrained("w11wo/indonesian-roberta-base-sentiment-classifier")
        self.model = AutoModelForSequenceClassification.from_pretrained("w11wo/indonesian-roberta-base-sentiment-classifier")
        self.model.eval()
        self.labeled_data_path = 'static/data/labeled_data.csv'
        self.default_data_path = 'static/data/labeled_data.csv'
        
        # Initialize with default data if available and no labeled data exists
        if not os.path.exists(self.labeled_data_path) and os.path.exists(self.default_data_path):
            shutil.copy2(self.default_data_path, self.labeled_data_path)
            if os.path.exists(self.labeled_data_path):
                df = pd.read_csv(self.labeled_data_path)
                self.rules = generate_rules_from_data(df)

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
        if os.path.exists(self.labeled_data_path):
            return pd.read_csv(self.labeled_data_path)
        elif os.path.exists(self.default_data_path):
            df = pd.read_csv(self.default_data_path)
            df.to_csv(self.labeled_data_path, index=False)
            return df
        return None

    def create_labeled_dataset(self, df1=None, df2=None):
        # If no data provided, use default data
        if df1 is None or df2 is None:
            if os.path.exists(self.default_data_path):
                df = pd.read_csv(self.default_data_path)
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

# Initialize analyzer
analyzer = SentimentAnalyzer()

@app.route('/')
def home():
    try:
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

            return render_template('index.html',
                               plots=plots_json,
                               stats=stats,
                               data=df.to_dict('records'),
                               eval_result=eval_result)
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
            }), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/upload-dataset', methods=['POST'])
def upload_dataset():
    try:
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
    import numpy as np
    from collections import Counter

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
        data = request.get_json()
        split_percent = int(data.get('split_percent', 80))
        df = pd.read_csv('static/data/labeled_data.csv')
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

        train_size = int(len(df) * split_percent / 100)
        df_train = df.iloc[:train_size]
        df_test = df.iloc[train_size:]

        # Save test set
        df_test.to_csv('static/data/test_data.csv', index=False)

        # DecisionTreeClassifier on preprocessed_text (vectorized, stopword removal, max depth unlimited)
        from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import confusion_matrix, classification_report
        import json

        # Remove stopwords from preprocessed_text
        stopwords = set(ENGLISH_STOP_WORDS)
        def remove_stopwords(text):
            return " ".join([w for w in text.split() if w not in stopwords])
        df_train['preprocessed_text'] = df_train['preprocessed_text'].apply(remove_stopwords)
        df_test['preprocessed_text'] = df_test['preprocessed_text'].apply(remove_stopwords)

        vectorizer = CountVectorizer()
        X_train = vectorizer.fit_transform(df_train['preprocessed_text'])
        y_train = df_train['sentiment']

        X_test = vectorizer.transform(df_test['preprocessed_text'])
        y_test = df_test['sentiment']

        clf = DecisionTreeClassifier(random_state=42, max_depth=None)
        clf.fit(X_train, y_train)

        # Save model (pickle)
        import pickle
        with open('static/data/sentiment_model.pkl', 'wb') as f:
            pickle.dump({'model': clf, 'vectorizer': vectorizer}, f)

        # Save rules as JSON using manual C5.0-like function
        sentiment_labels = sorted(y_train.unique())
        label_to_int = {label: i for i, label in enumerate(sentiment_labels)}
        int_to_label = {i: label for label, i in label_to_int.items()}

        y_train_int = np.array([label_to_int[y] for y in y_train])
        X_train_arr = X_train.toarray()
        feature_names = vectorizer.get_feature_names_out().tolist()

        c50_tree = build_c50_tree(X_train_arr, y_train_int, feature_names)

        # Recursively convert integer classes to sentiment labels in the tree
        def convert_classes(tree):
            if "class" in tree:
                tree["class"] = int_to_label.get(tree["class"], str(tree["class"]))
            if "splits" in tree:
                for k, v in tree["splits"].items():
                    convert_classes(v)
            return tree

        c50_tree = convert_classes(c50_tree)
        with open('static/data/rules.json', 'w') as f:
            json.dump(c50_tree, f, indent=2)

        # Save human-readable rules as TXT
        def rules_to_text(tree, indent=0):
            pad = "    " * indent
            if "class" in tree:
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

        # Generate and save visualization as image (PNG) using matplotlib
        import matplotlib.pyplot as plt
        from sklearn import tree as sktree

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
        plt.savefig("static/data/sentiment_rules_tree.svg", format="svg", bbox_inches="tight")
        plt.close()

        # Evaluate on test set
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
        report = classification_report(y_test, y_pred, output_dict=True, labels=clf.classes_)

        # Save confusion matrix and metrics
        eval_result = {
            'confusion_matrix': cm.tolist(),
            'labels': list(clf.classes_),
            'classification_report': report
        }
        with open('static/data/eval_result.json', 'w') as f:
            json.dump(eval_result, f, indent=2)

        return jsonify({'success': True, 'eval_result': eval_result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
if __name__ == '__main__':
    app.run(debug=True)