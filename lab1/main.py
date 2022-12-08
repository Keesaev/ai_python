import pandas
import numpy
import matplotlib.pyplot as plot
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# encode labels into tensor compatible labels
def encode_labels(labels):
    encoder = LabelEncoder()
    encoder.fit(labels)
    # to numbers [0, 0, ..., 1, 1, ..., 2, 2]
    encoded_labels = encoder.transform(labels)
    # to categories ([0, 0, 1], [0, 1, 0], [1, 0, 0]) 
    return to_categorical(encoded_labels)

# parse input from csv

dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values

X_train = dataset[:,0:4].astype(float)

Y_train = dataset[:,4]
Y_train = encode_labels(Y_train)

def print_history_stat(history):
    print("[Last epoch] Loss: ", history.history["loss"][-1], 
    " Accuracy: ", history.history["accuracy"][-1])

def compile_and_fit(model):
    model.compile(optimizer='adam',loss='categorical_crossentropy',
    metrics=['accuracy'])
    history = model.fit(X_train, Y_train, epochs=20, batch_size=10,
    validation_split=0.1, verbose=0)
    print_history_stat(history)
    return history

def test_1():
    print("Model 1 (16n):")
    his = compile_and_fit(Sequential([
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ]))
    plot.plot(his.history["loss"], label="16 neurons")

    print("Model 2 (32n):")
    his = compile_and_fit(Sequential([
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ]))
    plot.plot(his.history["loss"], label="32 neurons")

    print("Model 3 (64n):")
    his = compile_and_fit(Sequential([
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ]))
    plot.plot(his.history["loss"], label="64 neurons")
    plot.legend()
    plot.show()
    plot.clf()

def test_2():
    print("1x 16n layer:")
    his = compile_and_fit(Sequential([
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ]))
    plot.plot(his.history["loss"], label="1x 16n layer")

    print("2x 16n layers:")
    his = compile_and_fit(Sequential([
        Dense(16, activation='relu'),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ]))
    plot.plot(his.history["loss"], label="2x 16n layers")

    print("3x 16n layers:")
    his = compile_and_fit(Sequential([
        Dense(16, activation='relu'),
        Dense(16, activation='relu'),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ]))
    plot.plot(his.history["loss"], label="3x 16n layers")

    print("4x 16n layers:")
    his = compile_and_fit(Sequential([
        Dense(16, activation='relu'),
        Dense(16, activation='relu'),
        Dense(16, activation='relu'),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ]))
    plot.plot(his.history["loss"], label="4x 16n layers")

    plot.legend()
    plot.show()
    plot.clf()


exit()
plot.plot(history.history["loss"])
plot.plot(history.history["accuracy"])
plot.show()