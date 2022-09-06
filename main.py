import numpy as np
import tensorflow as tf
import math


class NeuralNetwork():

    def __init__(self):
        # seeding for random number generation
        np.random.seed(1)

        # converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
        self.synaptic_weightsA = np.random.uniform(-1.0, 1.0, (784, 16))
        self.synaptic_weightsB = np.random.uniform(-1.0, 1.0, (16, 16))
        self.synaptic_weightsC = np.random.uniform(-1.0, 1.0, (16, 10))

    def sigmoid(self, x):
        # applying the sigmoid function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # computing derivative to the Sigmoid function
        return x * (1 - x)

    def error(self, output, idealOutput):
        return np.square((output - idealOutput))

    def error_derivative(self, output, idealOutput):
        return 2 * (output - idealOutput)

    def backpropWeight(self, activationsA, activationsB):  # activationsA is a(L) and activationsB is a(L-1)
        weightAdjustmentMatrix = np.zeros((len(activationsB), len(activationsA)))
        for j in range(len(activationsA)):
            for k in range(len(activationsB)):
                weightAdjustmentMatrix[k][j] = (self.sigmoid(activationsB[k])) * (
                    self.sigmoid_derivative(activationsA[j]))

        return weightAdjustmentMatrix

    def backpropActivation(self, activationsA, activationsB, weightMatrix):  # activationsA is a(L), activationsB is a(L-1) and weight matrix is weightMatrix(L)
        activationDerivMatrix = np.zeros((len(activationsB)), len(activationsA))

    def train(self, training_inputs, training_outputs, training_iterations):
        # training the model to make accurate predictions while adjusting weights continually
        for iteration in range(training_iterations):
            for input in range(len(training_inputs)):
                # siphon the training data via  the neuron
                hiddenLayerA = self.think(training_inputs[input], self.synaptic_weightsA)
                hiddenLayerB = self.think(self.sigmoid(hiddenLayerA), self.synaptic_weightsB)
                output = self.think(self.sigmoid(hiddenLayerB), self.synaptic_weightsC)

                error = self.error(self.sigmoid(output), training_outputs[input])
                error_derivatives = self.error_derivative(self.sigmoid(output), training_outputs[input])

                # Adjusting Weight Matrix C
                adjustmentC = self.backprop(output, hiddenLayerB)
                for i in range(len(adjustmentC[0])):
                    for j in range(len(adjustmentC)):
                        adjustmentC[j][i] *= error_derivatives[i]
                self.synaptic_weightsC -= self.sigmoid(adjustmentC)

                adjustmentB = self.backpropWeight(hiddenLayerB, hiddenLayerA)
                activationGradientL1 = self.backpropActivation()

        # computing error rate for back-propagation
        # error = training_outputs - output
        # performing weight adjustments
        # adjustments = np.dot(hiddenLayerB.T, error * self.sigmoid_derivative(output))

        # self.synaptic_weightsC += adjustments

    def think(self, inputs, weight):
        # passing the inputs via the neuron to get output
        # converting values to floats

        inputs = inputs.astype(float)
        output = np.dot(inputs, weight)
        return output

    def test(self, testing_inputs, testing_outputs):
        right = 0
        wrong = 0
        for input in range(len(testing_inputs)):
            hiddenLayerA = self.think(testing_inputs[input], self.synaptic_weightsA)
            hiddenLayerB = self.think(self.sigmoid(hiddenLayerA), self.synaptic_weightsB)
            output = self.think(self.sigmoid(hiddenLayerB), self.synaptic_weightsC)
            output = np.argmax(self.sigmoid(output))
            correctOutput = np.argmax(testing_outputs[input])

            if (output == correctOutput):
                right += 1
            else:
                wrong += 1
        print(right)
        print(wrong)


if __name__ == "__main__":

    # initializing the neuron class
    neural_network = NeuralNetwork()

    a = np.array([1, 2, 3])

    print("Refactoring Database...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    tempArr = np.zeros(784)
    newX_train = np.zeros((len(x_train), 784))
    for i in range(len(x_train)):
        count = 0
        for j in range(len(x_train[0])):
            for k in range(len(x_train[0][0])):
                tempArr[count] = x_train[i][j][k] / 255
                count += 1
        newX_train[i] = tempArr

    newX_test = np.zeros((len(x_test), 784))
    for i in range(len(x_test)):
        count = 0
        for j in range(len(x_test[0])):
            for k in range(len(x_test[0][0])):
                tempArr[count] = x_train[i][j][k] / 255
                count += 1
        newX_test[i] = tempArr

    newY_train = np.zeros((len(y_train), 10))
    tempArr = np.zeros(10)
    for i in range(len(y_train)):
        tempArr = np.zeros(10)
        tempArr[y_train[i]] = 1.0
        newY_train[i] = tempArr

    newY_test = np.zeros((len(y_test), 10))
    tempArr = np.zeros(10)
    for i in range(len(y_test)):
        tempArr = np.zeros(10)
        tempArr[y_test[i]] = 1.0
        newY_test[i] = tempArr

    # training taking place
    print("Training...")
    neural_network.train(newX_train, newY_train, 1)

    print("Testing...")
    neural_network.test(newX_test, newY_test)