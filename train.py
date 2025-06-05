# Sentiment Analysis Model Training - Optimized LSTM Model
# This standalone script trains an LSTM model for sentiment analysis and saves it locally in Colab

import pandas as pd
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, SpatialDropout1D, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Create a directory for saving models
os.makedirs('/content/sentiment_model', exist_ok=True)

print("Loading and preprocessing dataset...")

# Load the dataset

df = pd.read_csv("/content/train.csv", encoding='latin1', on_bad_lines='skip')

# Display basic information
print(f"Dataset shape: {df.shape}")
print("\nSample data:")
print(df.head())

# Check sentiment distribution
print("\nSentiment distribution:")
print(df['sentiment'].value_counts())

# Data preprocessing
# Extract only the needed columns
data = df[['text', 'sentiment']].copy()
data = data.dropna()

# Convert sentiment labels to numeric values
sentiment_mapping = {
    'positive': 1,
    'negative': 0,
    'neutral': 2  # Adjust based on your dataset
}
data['sentiment_label'] = data['sentiment'].map(sentiment_mapping)
data = data.dropna(subset=['sentiment_label'])

# Text preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        # Remove non-alphanumeric characters
        text = re.sub(r'[^\w\s]', '', text)
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenization and lemmatization
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        
        tokens = nltk.word_tokenize(text)
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        
        return ' '.join(tokens)
    else:
        return ''

# Apply preprocessing to the text column
print("Preprocessing text data...")
data['processed_text'] = data['text'].apply(preprocess_text)

# Split data into features and target
X = data['processed_text']
y = data['sentiment_label']

# Split into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, {y_test.shape}")

# Prepare text data for LSTM
print("Tokenizing text for LSTM model...")
max_words = 10000  # Maximum number of words to keep
max_len = 100      # Maximum sequence length

# Tokenize text
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

# Save the tokenizer
with open('/content/sentiment_model/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Get vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print(f"Vocabulary size: {vocab_size}")

# Convert target to categorical if needed (multi-class)
num_classes = len(np.unique(y_train))
if num_classes > 2:
    from tensorflow.keras.utils import to_categorical
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)
    print(f"Using categorical labels for {num_classes} classes")
else:
    y_train_cat = y_train
    y_test_cat = y_test
    print("Using binary labels")

# Create Bidirectional LSTM model (more effective than standard LSTM)
print("Building and compiling the model...")
embedding_dim = 128

model = Sequential()
model.add(Embedding(input_dim=min(vocab_size, max_words), output_dim=embedding_dim, input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(64, dropout=0.2)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# Output layer based on number of classes
if num_classes > 2:
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
else:
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Callbacks for better training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint(
        filepath='/content/sentiment_model/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
]

# Train the model
print("Training the model...")
history = model.fit(
    X_train_pad, y_train_cat,
    validation_data=(X_test_pad, y_test_cat),
    epochs=10,
    batch_size=64,
    callbacks=callbacks
)

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.tight_layout()
plt.savefig('/content/sentiment_model/training_history.png')
plt.show()

# Evaluate the model
print("Evaluating the model...")
loss, accuracy = model.evaluate(X_test_pad, y_test_cat)
print(f"Test Accuracy: {accuracy:.4f}")

# Predictions
y_pred_prob = model.predict(X_test_pad)
if num_classes > 2:
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_test_decoded = np.argmax(y_test_cat, axis=1)
else:
    y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)
    y_test_decoded = y_test_cat

# Classification report
print("\nClassification Report:")
print(classification_report(y_test_decoded, y_pred))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test_decoded, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('/content/sentiment_model/confusion_matrix.png')
plt.show()

# Save the model and related files
print("Saving the model and metadata...")

# Save model architecture as JSON
model_json = model.to_json()
with open('/content/sentiment_model/model_architecture.json', 'w') as json_file:
    json_file.write(model_json)

# Save final model weights
model.save_weights('/content/sentiment_model/final_model_weights.h5')

# Save complete model
model.save('/content/sentiment_model/complete_model.h5')

# Save sentiment mapping for future reference
with open('/content/sentiment_model/sentiment_mapping.pkl', 'wb') as f:
    pickle.dump(sentiment_mapping, f)

# Create a simple function to test the model with new text
def test_sentiment(text):
    # Preprocess the text
    processed = preprocess_text(text)
    # Convert to sequence
    seq = tokenizer.texts_to_sequences([processed])
    # Pad sequence
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    # Predict
    prediction = model.predict(padded)[0]
    
    # Get the sentiment label
    if num_classes > 2:
        pred_class = np.argmax(prediction)
        confidence = prediction[pred_class]
    else:
        pred_class = 1 if prediction[0] > 0.5 else 0
        confidence = prediction[0] if pred_class == 1 else 1 - prediction[0]
    
    # Convert back to sentiment label
    reverse_mapping = {v: k for k, v in sentiment_mapping.items()}
    sentiment = reverse_mapping.get(pred_class, 'unknown')
    
    return sentiment, confidence

# Test with a few examples
test_examples = [
    "I love this product so much!",
    "I'm very disappointed with the service",
    "The weather is nice today"
]

print("\nTesting the model with example inputs:")
for example in test_examples:
    sentiment, confidence = test_sentiment(example)
    print(f"Text: '{example}'")
    print(f"Predicted sentiment: {sentiment} (confidence: {confidence:.4f})\n")

print("\nTraining and evaluation complete!")
print(f"All model files saved to: /content/sentiment_model/")

# Display files in the model directory
print("\nFiles saved in the model directory:")
for file in os.listdir('/content/sentiment_model'):
    print(f" - {file}")