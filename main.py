import numpy as np
import matplotlib.pyplot as plt
from class_neural import NeuralNetwork


input_nodes = 784
hidden_nodes = [128]
output_nodes = 10
learning_rate = 0.1


n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


file_name_train = "data_set/mnist_train_100.csv"
with open(file_name_train, 'r') as f_o:
   data_list = f_o.readlines()


np.random.shuffle(data_list)
for elem in data_list:
   all_values = elem.split(',')


   scaled_input = ((np.asfarray(all_values[1:])/255.0 * 0.99) + 0.01)
   targets = np.zeros(output_nodes) + 0.01


   targets[int(all_values[0])] = 0.99


   n.train(scaled_input, targets)


file_name_test = "data_set/mnist_test_10.csv"
with open(file_name_test, 'r') as f_o:
   test_list = f_o.readlines()


np.random.shuffle(test_list)
scorecard = []
i = 0
for elem in test_list:
   all_values = elem.split(',')
   scaled_input = ((np.asfarray(all_values[1:])/255.0 * 0.99) + 0.01)


   correct_output = int(all_values[0])


   result = np.transpose(n.query(scaled_input))


   output_num = np.argmax(result)


   if correct_output == output_num:
       scorecard.append(1)
   else:
       scorecard.append(0)


   if i == 0:
       plt.imshow(scaled_input.reshape((28, 28)), cmap='Greys')
       plt.show()


       print(f"Пример: получено {output_num} - должно: {all_values[0]}")


       i += 1


scorecard_array = np.asarray(scorecard)
per = (scorecard_array.sum() / scorecard_array.size) * 100


print(f"Процент точности: {per}%")

