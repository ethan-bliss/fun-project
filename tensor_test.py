# just doing some stuff with tensorflow :)

import tensorflow as tf
import numpy as np

# training data
X = tf.range(-100, 100, 4)
y = X + 10

# split the sets
X_train = X[:40]
y_train = y[:40]

X_test = X[40:]
y_test = y[40:]


print(X_train)

# Create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, input_dim = 1, activation=None),
    tf.keras.layers.Dense(1)
])

# Compile model
model.compile(loss="mae",
             optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
             metrics=["mae"])
             
# Fit the model
model.fit(X_train, y_train, epochs=80)

print(model.predict([17.0, 50.0, 150.0]))