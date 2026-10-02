"""
run_demo.py
~~~~~~~~~~~
Демонстрационный запуск: создание и обучение нейронной сети
для распознавания рукописных цифр MNIST.

Группа: <ЕТ-442>
ФИО: <Коптелов Петр Денисович>
"""

import mnist_loader
import network


if __name__ == "__main__":
    # Загружаем базу данных MNIST
    training_data, validation_data, test_data = \
        mnist_loader.load_data_wrapper()

    # Создаём нейронную сеть: 784 входа → 30 скрытых → 10 выходов
    net = network.Network([784, 30, 10])

    # Обучаем сеть: 30 эпох, размер подвыборки 10, скорость обучения 3.0
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)