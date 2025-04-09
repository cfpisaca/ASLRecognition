import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Concatenate, Reshape
)
from tensorflow.keras.models import Model

# 1. Load the processed data from 'processed_train.pkl'
#    This file contains (X, y, label_encoder)
with open("processed_train.pkl", "rb") as f:
    X, y, label_encoder = pickle.load(f)

# 2. Reshape X: (N, 63) -> (N, 21, 3)
#    Each sample has 21 landmarks with (x, y, z) coordinates
X = X.reshape((-1, 21, 3))
num_samples = X.shape[0]
print("Total samples:", num_samples)

# 3. Get number of classes and print them
num_classes = len(label_encoder.classes_)
print("Number of classes:", num_classes)
print("Classes:", label_encoder.classes_)

# 4. Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, 
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")

# 5. Build adjacency matrix for the hand graph (21 nodes)
#    Based on MediaPipe hand connections with added self-loops
num_nodes = 21
A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
edges = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]
for i, j in edges:
    A[i, j] = 1
    A[j, i] = 1

for i in range(num_nodes):
    A[i, i] = 1

# 6. Normalize the adjacency matrix: Ĥ = D^(-1/2) * A * D^(-1/2)
D = np.diag(np.sum(A, axis=1))
D_inv_sqrt = np.linalg.inv(np.sqrt(D))
A_norm = D_inv_sqrt @ A @ D_inv_sqrt
A_norm_tf = tf.constant(A_norm, dtype=tf.float32)

# 7. Define a custom GCNLayer
class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, activation=None, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.w = self.add_weight(
            shape=(input_dim, self.output_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='gcn_weight'
        )
        super(GCNLayer, self).build(input_shape)
        
    def call(self, inputs, adj):
        x = tf.matmul(inputs, self.w)   # shape: (batch, 21, output_dim)
        x = tf.matmul(adj, x)           # graph propagation
        if self.activation is not None:
            x = self.activation(x)
        return x

# 8. Build the combined (hybrid) model that merges GCN + CNN branches
def build_combined_model(num_classes, adjacency_matrix):
    inputs = Input(shape=(21, 3), name='input_landmarks')

    # GCN branch
    gcn_out = GCNLayer(64, activation='relu')(inputs, adjacency_matrix)
    gcn_out = GCNLayer(64, activation='relu')(gcn_out, adjacency_matrix)
    gcn_out = Flatten()(gcn_out)
    gcn_out = Dense(64, activation='relu')(gcn_out)

    # CNN branch
    cnn_input = Reshape((21, 3, 1))(inputs)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(cnn_input)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)

    # Merge both branches
    merged = Concatenate()([gcn_out, x])
    merged = Dense(128, activation='relu')(merged)
    merged = Dropout(0.5)(merged)
    outputs = Dense(num_classes, activation='softmax')(merged)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 9. Build the model
model = build_combined_model(num_classes, A_norm_tf)
model.summary()

# 10. Train the model
epochs = 10
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=32
)

# 11. Evaluate the model
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {val_acc * 100:.2f}%")

# 12. Save the trained model
os.makedirs("model", exist_ok=True)
model.save("model/asl_combined_model.h5")
print("Saved combined GCN+CNN model to 'model/asl_combined_model.h5'")
