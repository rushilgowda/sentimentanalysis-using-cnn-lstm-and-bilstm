import pandas as pd

# Attempt to load the dataset with error handling
file_path = 'hotel.csv'

try:
    # Read the file, skipping problematic rows
    data = pd.read_csv(file_path, on_bad_lines='skip', engine='python')
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading the dataset: {e}")
    data = None

# Display first few rows if loading is successful
if data is not None:
    print(data.head())

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Bidirectional, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping

# Map ratings to binary sentiment (1 for positive, 0 for negative)
data['Sentiment'] = data['Rating'].apply(lambda x: 1 if x >= 4 else 0)

texts = data['Review']
labels = data['Sentiment']

# Preprocess the text data
max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, maxlen=max_len)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define model architectures
def build_cnn():
    model = Sequential([
        Embedding(max_words, 128, input_length=max_len),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(pool_size=4),
        Dropout(0.5),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm_cnn():
    model = Sequential([
        Embedding(max_words, 128, input_length=max_len),
        LSTM(128, return_sequences=True),
        Conv1D(64, 5, activation='relu'),
        MaxPooling1D(pool_size=4),
        Dropout(0.5),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_bilstm_cnn():
    model = Sequential([
        Embedding(max_words, 128, input_length=max_len),
        Bidirectional(LSTM(128, return_sequences=True)),
        Conv1D(64, 5, activation='relu'),
        MaxPooling1D(pool_size=4),
        Dropout(0.5),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate each model
def train_and_evaluate(model, X_train, y_train, X_test, y_test, epochs=5, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=2)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return accuracy

# Initialize and train models
accuracies = {}

cnn_model = build_cnn()
accuracies['CNN'] = train_and_evaluate(cnn_model, X_train, y_train, X_test, y_test, epochs=5,batch_size=32)

lstm_cnn_model = build_lstm_cnn()
accuracies['LSTM-CNN'] = train_and_evaluate(lstm_cnn_model, X_train, y_train, X_test, y_test,epochs=5, batch_size=32)
bilstm_cnn_model = build_bilstm_cnn()
accuracies['BiLSTM-CNN'] = train_and_evaluate(bilstm_cnn_model, X_train, y_train, X_test, y_test, epochs=5,batch_size=32)

# Plotting the accuracies
models = list(accuracies.keys())
accuracy_values = list(accuracies.values())

plt.figure(figsize=(10, 6))
plt.bar(models, accuracy_values, color=['blue', 'orange', 'green'])
plt.title('Model Accuracy Comparison', fontsize=16)
plt.xlabel('Models', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.ylim(0, 1)
for i, v in enumerate(accuracy_values):
    plt.text(i, v + 0.02, f"{v*100:.2f}%", ha='center', fontsize=12)
plt.show();

# Train and evaluate with history
def train_with_history(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=2,
                        callbacks=[early_stopping])
    return history

# Plot training and validation metrics
def plot_history(history, model_name):
    plt.figure(figsize=(14, 6))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.title(f'{model_name} Accuracy', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    plt.title(f'{model_name} Loss', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Train and visualize each model
models = {'CNN': build_cnn(), 'LSTM-CNN': build_lstm_cnn(), 'BiLSTM-CNN': build_bilstm_cnn()}
histories = {}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    history = train_with_history(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32)
    histories[model_name] = history
    plot_history(history, model_name)

# Evaluate models on the test set
test_accuracies = {}
for model_name, model in models.items():
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    test_accuracies[model_name] = test_accuracy
    print(f"{model_name} Test Accuracy: {test_accuracy:.2f}")

# Function to predict sentiment
def predict_sentiment(input_sentence, model, tokenizer, max_len):
    # Preprocess the input sentence
    sequence = tokenizer.texts_to_sequences([input_sentence])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)

    # Predict sentiment
    prediction = model.predict(padded_sequence)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return sentiment, prediction

# Interactive sentiment prediction
def user_defined_prediction():
    print("Choose a model for sentiment prediction:")
    print("1: CNN")
    print("2: LSTM-CNN")
    print("3: BiLSTM-CNN")

    # Get user choice
    choice = input("Enter the number corresponding to your choice: ")
    if choice == '1':
        model = cnn_model
        model_name = "CNN"
    elif choice == '2':
        model = lstm_cnn_model
        model_name = "LSTM-CNN"
    elif choice == '3':
        model = bilstm_cnn_model
        model_name = "BiLSTM-CNN"
    else:
        print("Invalid choice. Defaulting to CNN.")
        model = cnn_model
        model_name = "CNN"

    # Get user input sentence
    input_sentence = input("\nEnter a sentence to analyze sentiment: ")
    sentiment, probability = predict_sentiment(input_sentence, model, tokenizer, max_len)

    # Display result
    print(f"\nModel Used: {model_name}")
    print(f"Sentence: \"{input_sentence}\"")
    print(f"Predicted Sentiment: {sentiment} (Confidence: {probability:.2f})")

# Call the function for user-defined prediction
user_defined_prediction()
