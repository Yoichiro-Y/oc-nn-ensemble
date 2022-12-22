import numpy as np
import tensorflow as tf

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Select only the normal and abnormal data
normal_data = x_train[y_train == 1]
abnormal_data = x_train[y_train == 7]

# Concatenate the normal and abnormal data
x_train = np.concatenate((normal_data, abnormal_data), axis=0)
y_train = np.concatenate((np.ones(normal_data.shape[0]), np.zeros(abnormal_data.shape[0])))

# Shuffle the training data
p = np.random.permutation(len(x_train))
x_train = x_train[p]
y_train = y_train[p]

# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64)

# Evaluate the model on the test data
_, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# Compute the AUC
y_pred = model.predict(x_test)
auc = tf.metrics.AUC()
auc.update_state(y_test, y_pred)
print('AUC:', auc.result().numpy())

# Print the confusion matrix
y_pred = np.where(y_pred > 0.5, 1, 0)
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_pred)
print('Confusion matrix:')
print(cm)
