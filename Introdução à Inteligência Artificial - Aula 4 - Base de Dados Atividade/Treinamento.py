import csv
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(
            -1, 1, (hidden_size, output_size)
        )
        self.bias_hidden = np.random.uniform(-1, 1, (1, hidden_size))
        self.bias_output = np.random.uniform(-1, 1, (1, output_size))

    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            hidden_output = sigmoid(
                np.dot(X, self.weights_input_hidden) + self.bias_hidden
            )
            output = sigmoid(
                np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
            )

            output_error = y - output
            output_delta = output_error * sigmoid_derivative(output)

            hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
            hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

            self.weights_hidden_output += learning_rate * np.dot(
                hidden_output.T, output_delta
            )
            self.weights_input_hidden += learning_rate * np.dot(X.T, hidden_delta)

            self.bias_output += learning_rate * np.sum(output_delta, axis=0)
            self.bias_hidden += learning_rate * np.sum(hidden_delta, axis=0)

    def predict(self, X):
        hidden_output = sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        output = sigmoid(
            np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        )
        return output


def load_data(filename):
    dataset = []
    with open(filename, "r") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            dataset.append([float(value) for value in row[:-1]])
    return dataset


def class_to_one_hot(cls, num_classes):
    one_hot = [0] * num_classes
    one_hot[cls] = 1
    return one_hot


def one_hot_to_class(one_hot):
    return np.argmax(one_hot)


def normalize_data(dataset):
    min_values = np.min(dataset, axis=0)
    max_values = np.max(dataset, axis=0)
    normalized_dataset = (dataset - min_values) / (max_values - min_values)
    return normalized_dataset


train_data = load_data("iris-train.data")
test_data = load_data("iris-test.data")

train_data = normalize_data(train_data)
test_data = normalize_data(test_data)

train_classes = [class_to_one_hot(int(row[-1]), 3) for row in train_data]
test_classes = [class_to_one_hot(int(row[-1]), 3) for row in test_data]

input_size = 4
hidden_size = 5
output_size = 3

mlp = MLP(input_size, hidden_size, output_size)

learning_rate = 0.1
epochs = 1000
mlp.train(train_data, train_classes, learning_rate, epochs)

correct_predictions = 0
for inputs, target in zip(test_data, test_classes):
    output = mlp.predict(inputs)
    predicted_class = one_hot_to_class(output)
    actual_class = one_hot_to_class(target)
    if predicted_class == actual_class:
        correct_predictions += 1

accuracy = correct_predictions / len(test_data) * 100
print("Accuracy:", accuracy)
