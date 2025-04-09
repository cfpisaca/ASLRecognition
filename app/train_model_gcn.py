import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# 1. Load the processed data from 'processed_train.pkl'
with open("processed_train.pkl", "rb") as f:
    X, y, label_encoder = pickle.load(f)

# 2. Reshape X: (N, 63) -> (N, 21, 3)
#    Each of the 21 landmarks has 3 coordinates: (x, y, z).
X = X.reshape(-1, 21, 3)
num_samples = X.shape[0]
print("Total samples:", num_samples)

# 3. Check the number of classes and list them
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
print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

# 5. Build the adjacency matrix for the hand graph
#    Here we define edges according to MediaPipe hand connections
num_nodes = 21
A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
edges = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20)# Pinky finger
]
for i, j in edges:
    A[i, j] = 1
    A[j, i] = 1
# Add self-loops for each node
for i in range(num_nodes):
    A[i, i] = 1

# 6. Compute the normalized adjacency matrix
#    Ĥ = D^(-1/2) * A * D^(-1/2)
D = np.diag(np.sum(A, axis=1))
D_inv_sqrt = np.linalg.inv(np.sqrt(D))
A_norm = D_inv_sqrt @ A @ D_inv_sqrt  # shape: (21, 21)
A_norm_tf = tf.constant(A_norm, dtype=tf.float32)

# 7. Define a custom GCNLayer
class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, activation=None, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)
        
    def build(self, input_shape):
        # input_shape: (batch_size, num_nodes, input_dim)
        input_dim = input_shape[-1]
        self.w = self.add_weight(
            shape=(input_dim, self.output_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='w'
        )
        super(GCNLayer, self).build(input_shape)
        
    def call(self, inputs, adj):
        # inputs: (batch_size, num_nodes, input_dim)
        # Multiply inputs by the weight matrix
        x = tf.matmul(inputs, self.w)  # => (batch_size, num_nodes, output_dim)
        # Propagate through the graph
        x = tf.matmul(adj, x)  # => (batch_size, num_nodes, output_dim)
        if self.activation is not None:
            x = self.activation(x)
        return x

# 8. Build the GCN model architecture
inputs = tf.keras.Input(shape=(num_nodes, 3))          # shape: (21, 3)
x = GCNLayer(64, activation='relu')(inputs, A_norm_tf) # First GCN layer
x = GCNLayer(64, activation='relu')(x, A_norm_tf)      # Second GCN layer
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# 9. Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 10. Train the GCN model
epochs = 10
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    validation_data=(X_val, y_val)
)

# 11. Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# 12. Save the trained GCN model
os.makedirs("model", exist_ok=True)
model.save("model/asl_gcn_model.h5")
print("GCN model training complete. Saved to 'model/asl_gcn_model.h5'.")

