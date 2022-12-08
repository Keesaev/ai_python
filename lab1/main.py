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

# for evaluation take 5 cases from every category
# intervals -> [0; 5] + [50; 55] + [100; 105]

X_eval = dataset[0:5,0:4].astype(float)
X_eval = numpy.concatenate((X_eval, dataset[50:55,0:4].astype(float)))
X_eval = numpy.concatenate((X_eval, dataset[100:105,0:4].astype(float)))

Y_eval = dataset[0:5,4]
Y_eval = numpy.concatenate((Y_eval, dataset[50:55,4]))
Y_eval = numpy.concatenate((Y_eval, dataset[100:105,4]))
Y_eval = encode_labels(Y_eval)

# for training take rest of the cases:
# (5; 50) + (55; 100) + (105; 150)

X_train = dataset[5:50,0:4].astype(float)
X_train = numpy.concatenate((X_train, dataset[55:100,0:4].astype(float)))
X_train = numpy.concatenate((X_train, dataset[105:150,0:4].astype(float)))

Y_train = dataset[5:50,4]
Y_train = numpy.concatenate((Y_train, dataset[55:100,4]))
Y_train = numpy.concatenate((Y_train, dataset[105:150,4]))
Y_train = encode_labels(Y_train)

def print_fit_stat(model):
    print("Eval:")
    model.evaluate(X_eval, Y_eval, verbose=2)
    print("Predict (perfect is [1, 0, 0]):")
    print(model.predict([[5.1,3.5,1.4,0.2]]))

def compile_and_fit(model):
    model.compile(optimizer='adam',loss='categorical_crossentropy',
    metrics=['accuracy'])
    history = model.fit(X_train, Y_train, epochs=20, batch_size=10,
    validation_split=0.1, verbose=0)
    print_fit_stat(model)
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

#test_1()

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


test_2()

exit()
plot.plot(history.history["loss"])
plot.plot(history.history["accuracy"])
plot.show()