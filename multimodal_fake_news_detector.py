# Multimodal Fake News Detection System
# Author: Reet Agarwal, Roll No: 2205057
# Complete Implementation

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import warnings
warnings.filterwarnings('ignore')

class MultimodalFakeNewsDetector:
    """
    Complete Multimodal Fake News Detection System

    This system combines textual and visual features to detect fake news
    using machine learning approaches.
    """

    def __init__(self):
        self.models = {}
        self.results = {}
        self.tfidf_vectorizer = None
        self.text_scaler = None
        self.image_scaler = None

    def preprocess_text(self, text):
        """Preprocess text data"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text

    def extract_text_features(self, text):
        """Extract linguistic features from text"""
        features = {}
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()])
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['capital_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0

        # Fake news linguistic indicators
        fake_indicators = ['breaking', 'shocking', 'urgent', 'exposed', 'secret', 'incredible']
        features['fake_indicator_count'] = sum(1 for word in fake_indicators if word in text.lower())

        return features

    def create_dataset(self, num_samples=2000):
        """Create synthetic dataset for demonstration"""
        np.random.seed(42)

        fake_patterns = [
            "BREAKING: Shocking revelation about {topic}",
            "You won't BELIEVE what happened to {person}",
            "URGENT: This will change everything about {topic}",
            "EXPOSED: The truth about {topic} they don't want you to know"
        ]

        real_patterns = [
            "Research published in journal shows {finding} about {topic}",
            "Study conducted by {institution} reveals {finding}",
            "According to official data, {finding} regarding {topic}",
            "Analysis by experts indicates {finding} in {topic}"
        ]

        topics = ["climate change", "technology", "healthcare", "economy", "education", "politics"]
        findings = ["significant trends", "important changes", "new developments", "updated statistics"]
        institutions = ["university", "research center", "government agency"]

        data = []
        for i in range(num_samples):
            if i % 2 == 0:  # Real news
                pattern = np.random.choice(real_patterns)
                text = pattern.format(
                    finding=np.random.choice(findings),
                    topic=np.random.choice(topics),
                    institution=np.random.choice(institutions)
                )
                text += " The methodology followed standard protocols and peer review processes."
                label = 0
                # Real news images (more standardized features)
                image_features = np.random.normal(0, 0.5, 50)
            else:  # Fake news
                pattern = np.random.choice(fake_patterns)
                text = pattern.format(
                    topic=np.random.choice(topics),
                    person="celebrity"
                )
                text += " Share this INCREDIBLE discovery with EVERYONE you know RIGHT NOW!"
                label = 1
                # Fake news images (more varied/sensational features)
                image_features = np.random.normal(0.3, 1.0, 50)

            data.append({
                'text': text,
                'image_features': image_features,
                'label': label
            })

        return pd.DataFrame(data)

    def prepare_features(self, df):
        """Prepare all feature types"""
        # Text preprocessing
        df['processed_text'] = df['text'].apply(self.preprocess_text)

        # Extract linguistic features
        text_features_list = [self.extract_text_features(text) for text in df['processed_text']]
        text_features_df = pd.DataFrame(text_features_list)

        # TF-IDF features
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
            tfidf_features = self.tfidf_vectorizer.fit_transform(df['processed_text']).toarray()
        else:
            tfidf_features = self.tfidf_vectorizer.transform(df['processed_text']).toarray()

        # Image features
        image_features = np.array(df['image_features'].tolist())

        # Scale features
        if self.text_scaler is None:
            self.text_scaler = StandardScaler()
            text_features_scaled = self.text_scaler.fit_transform(text_features_df)
        else:
            text_features_scaled = self.text_scaler.transform(text_features_df)

        if self.image_scaler is None:
            self.image_scaler = StandardScaler()
            image_features_scaled = self.image_scaler.fit_transform(image_features)
        else:
            image_features_scaled = self.image_scaler.transform(image_features)

        # Combine all features
        multimodal_features = np.concatenate([
            tfidf_features,
            text_features_scaled,
            image_features_scaled
        ], axis=1)

        return multimodal_features, tfidf_features, text_features_scaled, image_features_scaled

    def train(self, df):
        """Train all model variants"""
        X, X_text, X_text_features, X_image = self.prepare_features(df)
        y = df['label'].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Prepare separate feature sets for different models
        X_train_tfidf = X_train[:, :500]  # First 500 features are TF-IDF
        X_train_image = X_train[:, -50:]  # Last 50 features are image features

        X_test_tfidf = X_test[:, :500]
        X_test_image = X_test[:, -50:]

        # Train models
        print("Training models...")

        # Text-only model
        self.models['text_only'] = LogisticRegression(random_state=42, max_iter=1000)
        self.models['text_only'].fit(X_train_tfidf, y_train)

        # Image-only model
        self.models['image_only'] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models['image_only'].fit(X_train_image, y_train)

        # Simple concatenation
        self.models['concatenation'] = LogisticRegression(random_state=42, max_iter=1000)
        self.models['concatenation'].fit(X_train, y_train)

        # Advanced multimodal
        self.models['multimodal'] = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5, random_state=42
        )
        self.models['multimodal'].fit(X_train, y_train)

        # Evaluate models
        test_data = {
            'text_only': X_test_tfidf,
            'image_only': X_test_image,
            'concatenation': X_test,
            'multimodal': X_test
        }

        for model_name, model in self.models.items():
            y_pred = model.predict(test_data[model_name])
            y_prob = model.predict_proba(test_data[model_name])[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
            auc = roc_auc_score(y_test, y_prob)

            self.results[model_name] = {
                'accuracy': accuracy, 'precision': precision,
                'recall': recall, 'f1': f1, 'auc': auc
            }

        return X_train, X_test, y_train, y_test

    def predict(self, text, image_features):
        """Make prediction on new data"""
        # Create temporary dataframe
        temp_df = pd.DataFrame({
            'text': [text],
            'image_features': [image_features],
            'label': [0]  # dummy label
        })

        # Prepare features
        X, _, _, _ = self.prepare_features(temp_df)

        # Get prediction from best model
        best_model = self.models['multimodal']
        prediction = best_model.predict(X)[0]
        probability = best_model.predict_proba(X)[0]

        return prediction, probability

    def save_model(self, filename):
        """Save trained model"""
        model_data = {
            'models': self.models,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'text_scaler': self.text_scaler,
            'image_scaler': self.image_scaler,
            'results': self.results
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        """Load trained model"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)

        self.models = model_data['models']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.text_scaler = model_data['text_scaler']
        self.image_scaler = model_data['image_scaler']
        self.results = model_data['results']
        print(f"Model loaded from {filename}")

    def print_results(self):
        """Print formatted results"""
        print("\n" + "="*80)
        print("MULTIMODAL FAKE NEWS DETECTION - RESULTS")
        print("Author: Reet Agarwal, Roll No: 2205057")
        print("="*80)

        print(f"{'Method':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10}")
        print("-" * 80)

        for model_name, results in self.results.items():
            print(f"{model_name:<20} {results['accuracy']:.4f}     {results['precision']:.4f}     "
                  f"{results['recall']:.4f}     {results['f1']:.4f}     {results['auc']:.4f}")

        print("-" * 80)

# Main execution
if __name__ == "__main__":
    # Initialize detector
    detector = MultimodalFakeNewsDetector()

    # Create and prepare dataset
    print("Creating dataset...")
    df = detector.create_dataset(2000)
    print(f"Dataset created with {len(df)} samples")

    # Train models
    X_train, X_test, y_train, y_test = detector.train(df)

    # Print results
    detector.print_results()

    # Example prediction
    sample_text = "BREAKING: Shocking discovery will change everything!"
    sample_image = np.random.randn(50)

    prediction, probability = detector.predict(sample_text, sample_image)
    print(f"\nExample Prediction:")
    print(f"Text: {sample_text}")
    print(f"Prediction: {'Fake' if prediction == 1 else 'Real'}")
    print(f"Confidence: {max(probability):.4f}")

    # Save model
    detector.save_model('fake_news_detector.pkl')

    print("\nMultimodal Fake News Detection System Complete!")
    print("Author: Reet Agarwal, Roll No: 2205057")
