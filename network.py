"""
network.py
~~~~~~~~~~
Модуль создания и обучения нейронной сети для распознавания рукописных цифр
с использованием метода градиентного спуска.

Группа: <ЕТ-442>
ФИО: <Коптелов Петр Денисович>
"""

# Стандартные библиотеки
import random

# Сторонние библиотеки
import numpy as np

""" ---Раздел описаний--- """


def sigmoid(z):
    """Сигмоидальная функция активации."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Производная сигмоидальной функции."""
    return sigmoid(z) * (1 - sigmoid(z))


class Network(object):
    """Класс, описывающий нейронную сеть."""

    def __init__(self, sizes):
        """
        Конструктор класса.
        sizes - список размеров слоев нейронной сети.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Случайные начальные смещения для каждого слоя (кроме входного)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # Случайные начальные веса связей между слоями
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Подсчет выходных сигналов сети при заданных входных сигналах."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """
        Стохастический градиентный спуск.
        training_data - обучающая выборка пар (x, y);
        epochs - количество эпох обучения;
        mini_batch_size - размер подвыборки;
        eta - скорость обучения;
        test_data - (необязательный) тестирующая выборка.
        """
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        training_data = list(training_data)
        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Один шаг градиентного спуска по подвыборке."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Алгоритм обратного распространения ошибки."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Прямое распространение
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Обратное распространение
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Оценка прогресса в обучении."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Вектор частных производных функции стоимости."""
        return (output_activations - y)


""" ---Конец раздела описаний--- """

""" ---Тело программы--- """
if __name__ == "__main__":
    net = Network([2, 3, 1])
    print('Сеть net:')
    print('Количество слоев:', net.num_layers)
    for i in range(net.num_layers):
        print('Количество нейронов в слое', i, ':', net.sizes[i])
    for i in range(net.num_layers - 1):
        print('W_', i + 1, ':')
        print(np.round(net.weights[i], 2))
        print('b_', i + 1, ':')
        print(np.round(net.biases[i], 2))
""" ---Конец тела программы--- """