# Cell 1: Import Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import Ascending import LabelEncoder

# Cell 2: Load the Dataset
# Load the SMS Spam Collection dataset
train_data = pd.read_csv('spam.csv', encoding='latin-1')
test_data = pd.read_csv('test.csv', encoding='latin-1')

# Rename columns for clarity
train_data = train_data[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
test_data = test_data[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})

# Cell 3: Data Preprocessing
# Encode labels (ham=0, spam=1)
label_encoder = LabelEncoder()
train_data['label'] = label_encoder.fit_transform(train_data['label'])
test_data['label'] = label_encoder.transform(test_data['label'])

# Tokenize and pad sequences
vocab_size = 5000
max_length = 100
embedding_dim = 50

tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(train_data['text'])

train_sequences = tokenizer.texts_to_sequences(train_data['text'])
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')

test_sequences = tokenizer.texts_to_sequences(test_data['text'])
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

# Cell 4: Build and Train the Neural Network
# Define the model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(16, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_padded, train_data['label'],
    epochs=5,
    validation_data=(test_padded, test_data['label']),
    batch_size=32,
    verbose=1
)

# Cell 5: Define Prediction Function
def predict_message(message):
    # Preprocess the input message
    sequence = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    
    # Predict probability
    prob = model.predict(padded, verbose=0)[0][0]
    
    # Determine label
    label = 'spam' if prob >= 0.5 else 'ham'
    
    return [float(prob), label]

# Cell 6: Test the Model
# Test cases
test_messages = [
    "Hey, how are you doing today?",
    "Congratulations! You've won a free iPhone! Click here to claim now!"
]

for msg in test_messages:
    prob, label = predict_message(msg)
    print(f"Message: {msg}")
    print(f"Prediction: {label} (Probability: {prob:.4f})\n")

# Evaluate model on test data
test_loss, test_accuracy = model.evaluate(test_padded, test_data['label'], verbose=0)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")