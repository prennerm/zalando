# train a machine learning algorithm on the fashion mnist dataset
# using a convolutional neural network
# the mnist dataset is a dataset of 60,000 28x28 grayscale images of 10 fashion categories

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# load the dataset into a pandas dataframe separating training and testing data
train_df = pd.read_csv("fashion-mnist_train.csv")
test_df = pd.read_csv("fashion-mnist_test.csv")

# separate the labels and the features
x_train = train_df.drop("label", axis=1).values
y_train = train_df["label"].values
x_test = test_df.drop("label", axis=1).values
y_test = test_df["label"].values

# reshape the data to fit the model
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# normalize the grayscale values to lie between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# build the model
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax"),
    ]
)

# train the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.fit(x_train, y_train, epochs=10)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# test the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

# draw a confusion matrix
y_pred = model.predict(x_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

# calculate confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# visualize confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Vorhergesagte")
plt.ylabel("Tatsächliche")
plt.show()
