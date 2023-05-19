import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf  ## TENSORFLOW 2.0
import numpy as np


def label_encode(label):
    if label == "Iris-setosa":
        return [1, 0, 0]
    elif label == "Iris-versicolor":
        return [0, 1, 0]
    elif label == "Iris-virginica":
        return [0, 0, 1]


def data_encode(file):
    X = []
    Y = []
    with open(file, "r") as train_file:
        for line in train_file:
            line = line.strip().split(",")
            X.append([float(x) for x in line[:4]])
            Y.append(label_encode(line[4]))
    return X, Y


train_X, train_Y = data_encode("iris-train.data")
test_X, test_Y = data_encode("iris-test.data")

learning_rate = 0.01
training_epochs = 5000
display_steps = 100

n_input = 4
n_hidden = 10
n_output = 3

X = tf.keras.Input(shape=(n_input,))
Y = tf.keras.Input(shape=(n_output,))

weights = {
    "hidden": tf.Variable(tf.random.normal([n_input, n_hidden])),
    "output": tf.Variable(tf.random.normal([n_hidden, n_output])),
}

bias = {
    "hidden": tf.Variable(tf.random.normal([n_hidden])),
    "output": tf.Variable(tf.random.normal([n_output])),
}


def model(X, weights, bias):
    layer1 = tf.add(tf.matmul(X, weights["hidden"]), bias["hidden"])
    layer1 = tf.nn.relu(layer1)
    output_layer = tf.matmul(layer1, weights["output"]) + bias["output"]
    return output_layer


pred = model(X, weights, bias)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=pred))

optimizer = tf.keras.optimizers.Adam(learning_rate)
trainable_variables = list(weights.values()) + list(bias.values())


@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        pred = model(inputs, weights, bias)
        loss_value = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=pred)
        )
    gradients = tape.gradient(loss_value, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return loss_value


for epoch in range(training_epochs):
    _, c = train_step(train_X, train_Y), loss
    if (epoch + 1) % display_steps == 0:
        print("Epoch:", epoch + 1, "Cost:", c)
print("Optimization Finished")

test_result = model(test_X, weights, bias)
correct_prediction = tf.equal(tf.argmax(test_result, 1), tf.argmax(test_Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print("Accuracy:", accuracy.numpy())
