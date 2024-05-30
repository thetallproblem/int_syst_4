import numpy as np
import scipy.special as sc


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        self.lr = learning_rate

        self.weight = self.create_w()

        self.activation_function = lambda x: sc.expit(x)

    def create_w(self):
        list_weight = []

        if len(self.hnodes) == 0:
            list_weight.append(np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.inodes)))
        else:
            for i in range(len(self.hnodes)):
                if i == 0:
                    list_weight.append(np.random.normal(0.0, pow(self.hnodes[i], -0.5),
                                                        (self.hnodes[i], self.inodes)))
                    if len(self.hnodes) > 1:
                        list_weight.append(np.random.normal(0.0, pow(self.hnodes[i + 1], -0.5),
                                                            (self.hnodes[i + 1], self.hnodes[i])))
                    else:
                        list_weight.append(np.random.normal(0.0, pow(self.onodes, -0.5),
                                                            (self.onodes, self.hnodes[i])))
                elif i == len(self.hnodes) - 1:
                    list_weight.append(np.random.normal(0.0, pow(self.onodes, -0.5),
                                                        (self.onodes, self.hnodes[i])))
                else:
                    list_weight.append(np.random.normal(0.0, pow(self.hnodes[i + 1], -0.5),
                                                        (self.hnodes[i + 1], self.hnodes[i])))

        return list_weight

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        outputs_list = []
        outputs = 0
        i = 0

        for hn in self.weight:
            if i == 0:
                outputs = self.activation_function(np.dot(hn, inputs))
            else:
                outputs = self.activation_function(np.dot(hn, outputs))

            outputs_list.append(outputs)
            i += 1

        outputs_errors = targets - outputs

        if len(self.hnodes) == 0:
            self.weight[0] += self.lr * np.dot((outputs_errors * outputs_list[0] * (1.0 - outputs_list[0])),
                                               np.transpose(inputs))
        else:
            i = 0
            outputs_list = outputs_list[::-1]

            for out in outputs_list:
                if i == 0:
                    self.weight[-(i + 1)] += self.lr * np.dot((outputs_errors * out * (1.0 - out)),
                                                              np.transpose(outputs_list[i + 1]))
                elif i == len(outputs_list) - 1:
                    if len(self.weight) > 2:
                        errors = np.dot(self.weight[1].T, errors)
                        self.weight[0] += self.lr * np.dot((errors * out * (1.0 - out)), np.transpose(inputs))
                    elif len(self.weight) == 2:
                        errors = np.dot(self.weight[-1].T, outputs_errors)
                        self.weight[-(i + 1)] += self.lr * np.dot((errors * out * (1.0 - out)), np.transpose(inputs))
                elif i == 1:
                    errors = np.dot(self.weight[-1].T, outputs_errors)
                    self.weight[-(i + 1)] += self.lr * np.dot((errors * out * (1.0 - out)),
                                                              np.transpose(outputs_list[i + 1]))
                else:
                    errors = np.dot(self.weight[-i].T, errors)
                    self.weight[-(i + 1)] += self.lr * np.dot((errors * out * (1.0 - out)),
                                                              np.transpose(outputs_list[i + 1]))

                i += 1

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        for hn in self.weight:
            inputs = self.activation_function(np.dot(hn, inputs))

        return inputs
