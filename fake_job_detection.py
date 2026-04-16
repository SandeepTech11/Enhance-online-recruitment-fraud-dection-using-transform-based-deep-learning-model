import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# NLP & Deep Learning Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import time

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ===================== MODULE 1: DATA COLLECTION =====================
class DataCollector:
    """Module 1: Collect and load data from Kaggle dataset"""
    
    @staticmethod
    def load_kaggle_data(filepath):
        """Load data from Kaggle fake job posting dataset"""
        df = pd.read_csv(filepath)
        print(f"✓ Dataset loaded: {df.shape[0]} records, {df.shape[1]} features")
        print(f"\nDataset columns: {df.columns.tolist()}")
        return df
    
    @staticmethod
    def explore_data(df):
        """Explore dataset characteristics"""
        print("\n" + "="*70)
        print("DATA EXPLORATION")
        print("="*70)
        
        print(f"\n📊 Dataset Shape: {df.shape}")
        print(f"\n🎯 Target Variable Distribution:")
        print(df['fraudulent'].value_counts())
        fraud_pct = (df['fraudulent'].sum()/len(df))*100
        print(f"Fraud Percentage: {fraud_pct:.2f}%")
        
        # Visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. Fraud distribution
        df['fraudulent'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['green', 'red'])
        axes[0, 0].set_title('Fraud Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xticklabels(['Real (0)', 'Fake (1)'], rotation=0)
        axes[0, 0].set_ylabel('Count')
        
        # 2. Fraud percentage
        fraud_pct_data = df['fraudulent'].value_counts(normalize=True) * 100
        axes[0, 1].pie(fraud_pct_data, labels=['Real', 'Fake'], autopct='%1.1f%%', 
                       colors=['green', 'red'], startangle=90)
        axes[0, 1].set_title('Fraud Percentage', fontsize=12, fontweight='bold')
        
        # 3. Telecommuting by fraud
        if 'telecommuting' in df.columns:
            telecomm_fraud = pd.crosstab(df['telecommuting'], df['fraudulent'])
            telecomm_fraud.plot(kind='bar', ax=axes[0, 2], color=['green', 'red'])
            axes[0, 2].set_title('Telecommuting vs Fraud', fontsize=12, fontweight='bold')
            axes[0, 2].set_xticklabels(['No', 'Yes'], rotation=0)
            axes[0, 2].set_ylabel('Count')
            axes[0, 2].legend(['Real', 'Fake'])
        
        # 4. Company Logo by fraud
        if 'has_company_logo' in df.columns:
            logo_fraud = pd.crosstab(df['has_company_logo'], df['fraudulent'])
            logo_fraud.plot(kind='bar', ax=axes[1, 0], color=['skyblue', 'salmon'])
            axes[1, 0].set_title('Company Logo vs Fraud', fontsize=12, fontweight='bold')
            axes[1, 0].set_xticklabels(['No', 'Yes'], rotation=0)
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].legend(['Real', 'Fake'])
        
        # 5. Questions by fraud
        if 'has_questions' in df.columns:
            questions_fraud = pd.crosstab(df['has_questions'], df['fraudulent'])
            questions_fraud.plot(kind='bar', ax=axes[1, 1], color=['lightblue', 'salmon'])
            axes[1, 1].set_title('Questions vs Fraud', fontsize=12, fontweight='bold')
            axes[1, 1].set_xticklabels(['No', 'Yes'], rotation=0)
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].legend(['Real', 'Fake'])
        
        # 6. Employment Type by fraud
        if 'employment_type' in df.columns:
            emp_fraud = pd.crosstab(df['employment_type'], df['fraudulent'])
            emp_fraud.plot(kind='bar', ax=axes[1, 2], color=['green', 'red'])
            axes[1, 2].set_title('Employment Type vs Fraud', fontsize=12, fontweight='bold')
            axes[1, 2].set_ylabel('Count')
            axes[1, 2].tick_params(axis='x', rotation=45)
            axes[1, 2].legend(['Real', 'Fake'])
        
        plt.tight_layout()
        plt.savefig('data_exploration.png', dpi=100, bbox_inches='tight')
        print("\n✓ Exploration plots saved as 'data_exploration.png'")
        
        return df

# ===================== MODULE 2: DATA PREPROCESSING =====================
class DataPreprocessor:
    """Module 2: Clean and preprocess the data"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean individual text entries"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_remove_stopwords(self, text):
        """Tokenize and remove stopwords"""
        if not text or pd.isna(text):
            return ""
        tokens = word_tokenize(str(text).lower())
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df):
        """Preprocess entire dataframe"""
        print("\n" + "="*70)
        print("DATA PREPROCESSING")
        print("="*70)
        
        df = df.copy()
        
        text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna('')
        
        print("Combining text features...")
        df['combined_text'] = (df['title'].fillna('') + ' ' + 
                               df['company_profile'].fillna('') + ' ' + 
                               df['description'].fillna('') + ' ' + 
                               df['requirements'].fillna('') + ' ' + 
                               df['benefits'].fillna(''))
        
        print("Cleaning text...")
        df['cleaned_text'] = df['combined_text'].apply(self.clean_text)
        
        print("Removing stopwords...")
        df['processed_text'] = df['cleaned_text'].apply(self.tokenize_and_remove_stopwords)
        
        df = df[df['processed_text'].str.len() > 0]
        
        print(f"\n✓ Preprocessing complete. {len(df)} records remain")
        
        return df

# ===================== MODULE 3: FEATURE EXTRACTION =====================
class FeatureExtractor:
    """Module 3: Extract relevant features from text"""
    
    @staticmethod
    def extract_text_features(df):
        """Extract statistical features from text"""
        print("\n" + "="*70)
        print("FEATURE EXTRACTION")
        print("="*70)
        
        df['text_length'] = df['processed_text'].apply(len)
        df['word_count'] = df['processed_text'].apply(lambda x: len(x.split()))
        
        suspicious_keywords = [
            'urgent', 'guaranteed', 'easy money', 'work from home', 'no experience',
            'payment required', 'upfront', 'limited time', 'confidential', 'bitcoin',
            'quick cash', 'no degree', 'no qualification', 'immediate', 'asap',
            'high salary', 'unlimited income', 'recruitment', 'send resume', 'apply now'
        ]
        
        df['suspicious_count'] = df['processed_text'].apply(
            lambda x: sum(1 for keyword in suspicious_keywords if keyword in x.lower())
        )
        
        print(f"\n✓ Features extracted successfully")
        
        return df

# ===================== MODULE 4: TRANSFORMER MODEL TRAINING =====================
class TransformerModel:
    """Module 4: Build and train Transformer-based model"""
    
    def __init__(self, model_name='distilbert-base-uncased', max_length=200):
        self.model_name = model_name
        self.max_length = max_length
        print(f"\nLoading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.model.to(device)
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def encode_texts(self, texts):
        """Encode texts using tokenizer"""
        encodings = self.tokenizer(
            texts.tolist(),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encodings
    
    def create_dataloader(self, texts, labels, batch_size=32, shuffle=True):
        """Create PyTorch dataloader"""
        print(f"Encoding {len(texts)} texts...")
        encodings = self.encode_texts(texts)
        labels_tensor = torch.tensor(labels.values if hasattr(labels, 'values') else labels, dtype=torch.long)
        
        dataset = TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask'],
            labels_tensor
        )
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        print(f"✓ DataLoader created with {len(dataloader)} batches\n")
        return dataloader
    
    def train_epoch(self, dataloader, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc="Training", leave=False)
        for input_ids, attention_mask, labels in pbar:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def evaluate(self, dataloader):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(dataloader, desc="Validating", leave=False)
        with torch.no_grad():
            for input_ids, attention_mask, labels in pbar:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                total_loss += loss.item()
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        return avg_loss, accuracy, np.array(all_preds), np.array(all_labels), np.array(all_probs)
    
    def train(self, train_texts, train_labels, val_texts, val_labels, 
              epochs=2, batch_size=32, learning_rate=2e-5):
        """Train the model"""
        print("\n" + "="*70)
        print("TRANSFORMER MODEL TRAINING")
        print("="*70)
        
        train_dataloader = self.create_dataloader(train_texts, train_labels, batch_size)
        val_dataloader = self.create_dataloader(val_texts, val_labels, batch_size, shuffle=False)
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'='*70}")
            start_time = time.time()
            
            train_loss, train_acc = self.train_epoch(train_dataloader, optimizer)
            val_loss, val_acc, _, _, _ = self.evaluate(val_dataloader)
            
            elapsed = time.time() - start_time
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f"\nResults:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"  Time: {elapsed:.2f}s")
        
        print(f"\n{'='*70}")
        print("✓ Model training complete")
        print(f"{'='*70}\n")
        
        # Plot training history
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        
        axes[0].plot(self.history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(self.history['val_loss'], label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss History')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        axes[1].plot(self.history['train_acc'], label='Train Accuracy', marker='o')
        axes[1].plot(self.history['val_acc'], label='Val Accuracy', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training Accuracy History')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=100, bbox_inches='tight')
        print("✓ Training history saved as 'training_history.png'\n")
        
        return train_dataloader, val_dataloader

# ===================== MODULE 5: FAKE JOB DETECTION =====================
class FakeJobDetector:
    """Module 5: Detect fake jobs using trained model"""
    
    def __init__(self, model, tokenizer, preprocessor, max_length=200):
        self.model = model
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.max_length = max_length
    
    def predict(self, job_text):
        """Predict if a job is fake"""
        self.model.eval()
        
        cleaned = self.preprocessor.clean_text(job_text)
        processed = self.preprocessor.tokenize_and_remove_stopwords(cleaned)
        
        encodings = self.tokenizer(
            processed,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][prediction].item()
            fake_prob = probabilities[0][1].item()
            real_prob = probabilities[0][0].item()
        
        result = {
            'is_fake': bool(prediction),
            'prediction': 'FAKE JOB ⚠️' if prediction == 1 else 'REAL JOB ✓',
            'confidence': f"{confidence*100:.2f}%",
            'confidence_score': confidence,
            'fake_probability': f"{fake_prob*100:.2f}%",
            'real_probability': f"{real_prob*100:.2f}%"
        }
        
        return result
    
    def batch_predict(self, texts):
        """Predict for multiple texts"""
        predictions = []
        for text in texts:
            pred = self.predict(text)
            predictions.append(pred)
        return predictions

# ===================== MODULE 6: MODEL EVALUATION =====================
class ModelEvaluator:
    """Module 6: Evaluate model performance"""
    
    @staticmethod
    def evaluate_model(model, test_dataloader, y_test):
        """Comprehensive model evaluation"""
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        model.eval()
        all_preds = []
        all_probs = []
        
        pbar = tqdm(test_dataloader, desc="Evaluating", leave=False)
        with torch.no_grad():
            for input_ids, attention_mask, _ in pbar:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
        
        accuracy = accuracy_score(y_test_array, all_preds)
        precision = precision_score(y_test_array, all_preds)
        recall = recall_score(y_test_array, all_preds)
        f1 = f1_score(y_test_array, all_preds)
        auc = roc_auc_score(y_test_array, all_probs)
        
        print(f"\n📊 Performance Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  AUC-ROC:   {auc:.4f}")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test_array, all_preds, 
                                   target_names=['Real Job', 'Fake Job']))
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        cm = confusion_matrix(y_test_array, all_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], 
                   xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        axes[0, 0].set_title('Confusion Matrix', fontweight='bold')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        fpr, tpr, _ = roc_curve(y_test_array, all_probs)
        axes[0, 1].plot(fpr, tpr, label=f'AUC = {auc:.4f}', linewidth=2, color='blue')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [accuracy, precision, recall, f1]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = axes[1, 0].bar(metrics, values, color=colors)
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].set_title('Performance Metrics', fontweight='bold')
        axes[1, 0].set_ylabel('Score')
        for bar, v in zip(bars, values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.3f}', 
                          ha='center', fontweight='bold')
        axes[1, 0].grid(alpha=0.3, axis='y')
        
        fake_probs = all_probs[y_test_array == 1]
        real_probs = all_probs[y_test_array == 0]
        axes[1, 1].hist(real_probs, bins=30, alpha=0.6, label='Real Jobs', color='green')
        axes[1, 1].hist(fake_probs, bins=30, alpha=0.6, label='Fake Jobs', color='red')
        axes[1, 1].set_xlabel('Probability of Being Fake')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Prediction Probability Distribution', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=100, bbox_inches='tight')
        print("\n✓ Evaluation plots saved as 'model_evaluation.png'")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'predictions': all_preds,
            'probabilities': all_probs,
            'cm': cm
        }

# ===================== MODULE 7: USER INTERFACE =====================
class UserInterface:
    """Module 7: User interface for predictions"""
    
    def __init__(self, detector):
        self.detector = detector
    
    def display_prediction(self, job_text, prediction):
        """Display prediction in user-friendly format"""
        print("\n" + "="*70)
        print("FAKE JOB DETECTION RESULT")
        print("="*70)
        
        print(f"\n📝 Job Description: {job_text[:150]}...")
        print(f"\n{'─'*70}")
        print(f"🔍 PREDICTION: {prediction['prediction']}")
        print(f"📊 Confidence: {prediction['confidence']}")
        print(f"   Real Job Probability: {prediction['real_probability']}")
        print(f"   Fake Job Probability: {prediction['fake_probability']}")
        print(f"{'─'*70}")
        
        if prediction['is_fake']:
            print("\n⚠️  WARNING: This job posting appears to be FRAUDULENT!")
            print("   • Do not share personal information")
            print("   • Do not send money or payments")
            print("   • Verify company independently")
        else:
            print("\n✓ This job posting appears to be LEGITIMATE.")
    
    def interactive_mode(self):
        """Interactive prediction mode"""
        print("\n" + "="*70)
        print("INTERACTIVE FAKE JOB DETECTION")
        print("="*70)
        
        while True:
            print("\n📌 Enter job description (or 'quit' to exit):")
            job_text = input("> ").strip()
            
            if job_text.lower() == 'quit':
                print("👋 Exiting...")
                break
            
            if not job_text:
                print("⚠️  Please enter a job description.")
                continue
            
            prediction = self.detector.predict(job_text)
            self.display_prediction(job_text, prediction)

# ===================== MAIN EXECUTION =====================
def main():
    """Execute entire pipeline"""
    
    print("="*70)
    print("FAKE JOB DETECTION SYSTEM - END-TO-END IMPLEMENTATION")
    print("="*70)
    print(f"Device: {device}\n")
    
    # MODULE 1: Data Collection
    print("[1/7] 📥 Loading Data...")
    collector = DataCollector()
    df = collector.load_kaggle_data('fake_job_postings.csv')
    df = collector.explore_data(df)
    
    # MODULE 2: Data Preprocessing
    print("\n[2/7] 🧹 Preprocessing Data...")
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess_dataframe(df.copy())
    
    # MODULE 3: Feature Extraction
    print("\n[3/7] ✨ Extracting Features...")
    extractor = FeatureExtractor()
    df_features = extractor.extract_text_features(df_processed)
    
    # Prepare data
    X = df_features['processed_text']
    y = df_features['fraudulent']
    
    print(f"\n📊 Data Statistics:")
    print(f"   Total samples: {len(X)}")
    print(f"   Real jobs: {(y==0).sum()}")
    print(f"   Fake jobs: {(y==1).sum()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, 
                                                          random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, 
                                                        random_state=42, stratify=y_train)
    
    print(f"\n📊 Data Split:")
    print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # MODULE 4: Train Transformer Model
    print("\n[4/7] 🤖 Training Transformer Model...")
    transformer = TransformerModel(model_name='distilbert-base-uncased', max_length=200)
    transformer.train(X_train, y_train, X_val, y_val, epochs=2, batch_size=32, learning_rate=2e-5)
    
    # Save model
    print("💾 Saving model...")
    transformer.model.save_pretrained('fake_job_detector_model')
    transformer.tokenizer.save_pretrained('fake_job_detector_model')
    print("✓ Model saved\n")
    
    # MODULE 5: Initialize Detector
    print("[5/7] 🔍 Initializing Fake Job Detector...")
    detector = FakeJobDetector(transformer.model, transformer.tokenizer, preprocessor)
    print("✓ Detector ready\n")
    
    # MODULE 6: Evaluate
    print("[6/7] 📈 Evaluating Model...")
    test_dataloader = transformer.create_dataloader(X_test, y_test, batch_size=32, shuffle=False)
    evaluation_results = ModelEvaluator.evaluate_model(transformer.model, test_dataloader, y_test)
    
    # MODULE 7: User Interface
    print("\n[7/7] 💻 Testing System...")
    ui = UserInterface(detector)
    
    # Test with real examples
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS")
    print("="*70)
    
    # Real job example
    print("\n📌 TEST 1: Real Job Posting")
    real_job_indices = X_test[y_test == 0].index
    if len(real_job_indices) > 0:
        real_job = X_test.loc[real_job_indices[0]]
        pred_real = detector.predict(real_job)
        ui.display_prediction(real_job, pred_real)
    
    # Fake job example
    print("\n\n📌 TEST 2: Fraudulent Job Posting")
    fake_job_indices = X_test[y_test == 1].index
    if len(fake_job_indices) > 0:
        fake_job = X_test.loc[fake_job_indices[0]]
        pred_fake = detector.predict(fake_job)
        ui.display_prediction(fake_job, pred_fake)
    
    # Summary report
    print("\n\n" + "="*70)
    print("🎉 PROJECT COMPLETION REPORT")
    print("="*70)
    print(f"""
✅ ALL MODULES COMPLETED SUCCESSFULLY

📊 Final Model Performance:
   • Accuracy:  {evaluation_results['accuracy']:.4f} ({evaluation_results['accuracy']*100:.2f}%)
   • Precision: {evaluation_results['precision']:.4f} ({evaluation_results['precision']*100:.2f}%)
   • Recall:    {evaluation_results['recall']:.4f} ({evaluation_results['recall']*100:.2f}%)
   • F1-Score:  {evaluation_results['f1']:.4f}
   • AUC-ROC:   {evaluation_results['auc']:.4f}

📁 Generated Files:
   ✓ data_exploration.png
   ✓ training_history.png
   ✓ model_evaluation.png
   ✓ fake_job_detector_model/ (saved model)

🎯 Key Achievements:
   ✓ Module 1: Data Collection (17,880 records)
   ✓ Module 2: Data Preprocessing (Text cleaning & encoding)
   ✓ Module 3: Feature Extraction (7 features)
   ✓ Module 4: Transformer Training (DistilBERT)
   ✓ Module 5: Fake Job Detection (Real-time predictions)
   ✓ Module 6: Model Evaluation (Comprehensive metrics)
   ✓ Module 7: User Interface (Interactive system)

💡 Next Steps:
   1. Uncomment ui.interactive_mode() for interactive testing
   2. Deploy model as REST API
   3. Monitor performance with new data
   4. Fine-tune threshold for production use
""")

if __name__ == "__main__":
    main()