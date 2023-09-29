# train a machine learning algorithm on the fashion mnist dataset
# using a convolutional neural network
# the mnist dataset is a dataset of 60,000 28x28 grayscale images of 10 fashion categories

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import random
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

st.title("Fashion MNIST - Convolutional Neural Network")
st.write(
    "This app is a demo of the Streamlit framework using the fashion MNIST dataset."
)

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
        Dropout(0.25),
        Dense(10, activation="softmax"),
    ]
)

# train the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
num_epochs = st.slider("Number of epochs", min_value=1, max_value=20, value=2)
model.fit(
    x_train, y_train, epochs=num_epochs, batch_size=32, validation_data=(x_test, y_test)
)

# test the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

# streamlit widgets for depiction
st.subheader("Test Accuracy")
st.write(f"Test Accuracy: {test_acc}")
st.write(f"Test Loss: {test_loss}")

# draw a confusion matrix
y_pred = model.predict(x_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

# calculate confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

st.subheader("Confusion matrix")
# visualize confusion matrix
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", ax=ax)
plt.xlabel("Predicted")
plt.ylabel("True")
st.pyplot(fig)

# classification report
st.subheader("Classification report")
print(classification_report(y_test, y_pred_classes))

# binarize labels for multiclass roc curve
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y_pred_bin = label_binarize(y_pred_classes, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# calculate roc curve
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(10):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# plot roc curve
st.subheader("ROC-Curve")
fig, ax = plt.subplots(figsize=(10, 7))
for i in range(10):
    plt.plot(fpr[i], tpr[i], label="ROC curve (area = %0.2f)" % roc_auc[i])
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
st.pyplot(fig)

# find misclassified images
misclassified_idx = []
for i in range(len(y_pred_classes)):
    if y_pred_classes[i] != y_test[i]:
        misclassified_idx.append(i)

# plot misclassified images
st.subheader("Misclassified images")
random.shuffle(misclassified_idx)
fig, axs = plt.subplots(3, 3, figsize=(10, 7))
for i in range(0, 9):
    ax = axs[i // 3, i % 3]
    ax.imshow(x_test[misclassified_idx[i]].reshape(28, 28), cmap=plt.get_cmap("gray"))
    ax.set_title(
        "Predicted: {} \n True: {}".format(
            y_pred_classes[misclassified_idx[i]], y_test[misclassified_idx[i]]
        )
    )
plt.subplots_adjust(hspace=0.5)
st.pyplot(fig)
