import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

def load_data():
    """
    1. Load processed data (features, labels, label_encoder)
       from 'processed_train.pkl'.
    """
    filename = "processed_train.pkl"
    with open(filename, "rb") as f:
        X, y, label_encoder = pickle.load(f)
    return X, y, label_encoder

if __name__ == "__main__":
    # 2. Load the data
    X, y, train_le = load_data()

    # 3. Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("Training samples:", X_train.shape[0])
    print("Validation samples:", X_val.shape[0])

    # 4. Build a simple MLP model
    model = Sequential([
        # First Dense layer with input dimension
        Dense(128, input_dim=X_train.shape[1], activation='relu'),
        # Dropout layer to reduce overfitting
        Dropout(0.5),
        # Second Dense layer
        Dense(64, activation='relu'),
        # Another Dropout layer
        Dropout(0.5),
        # Final output layer with 'softmax' activation
        Dense(len(train_le.classes_), activation='softmax')
    ])

    # 5. Compile the model with optimizer, loss, and metrics
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 6. Print a summary of the model architecture
    model.summary()

    # 7. Train the model using the training data
    model.fit(
        X_train, y_train,
        epochs=10,
        validation_data=(X_val, y_val)
    )

    # 8. Save the trained model to disk
    model.save('model/asl_model.h5')
    print("Model training complete and saved to 'model/asl_model.h5'")

    # 9. Evaluate the model on the validation set
    test_loss, test_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {test_accuracy * 100:.2f}%")

