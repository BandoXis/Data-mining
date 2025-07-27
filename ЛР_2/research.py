import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import pandas as pd

def func_sine_wave(x_array: np.ndarray, y_array: np.ndarray) -> [np.ndarray, str, callable]:
    return (x_array - 3.14) ** 2 + (y_array - 2.72) ** 2 + np.sin(3 * x_array + 1.41) + np.sin(
        4 * y_array - 1.73), "Синусоида", format_coord_sin_wave


def format_coord_sin_wave(x: np.float64, y: np.float64) -> str:
    return f'x={x:.2f}, y={y:.2f}, f(x, y)={func_sine_wave(x, y)[0]:.2f}'


def func_Himmelblau(x_array: np.ndarray, y_array: np.ndarray) -> [np.ndarray, str, callable]:
    return (x_array ** 2 + y_array - 11) ** 2 + (
            x_array + y_array ** 2 - 7) ** 2, "Химмельблау", format_coord_Himmelblau


def format_coord_Himmelblau(x: np.float64, y: np.float64) -> str:
    return f'x={x:.2f}, y={y:.2f}, f(x, y)={func_Himmelblau(x, y)[0]:.2f}'


def func_Rastrigin(x: np.ndarray, y: np.ndarray, A: float = 10) -> [np.ndarray, str, callable]:
    return A + (x ** 2 - A * np.cos(2 * np.pi * x)) + (
            y ** 2 - A * np.cos(2 * np.pi * y)), "Растригина", format_coord_Rastrigin


def format_coord_Rastrigin(x: np.float64, y: np.float64) -> str:
    return f'x={x:.2f}, y={y:.2f}, f(x, y)={func_Rastrigin(x, y)[0]:.2f}'


def func_Rosenbrock(x: np.ndarray, y: np.ndarray, A: float = 100) -> [np.ndarray, str, callable]:
    return (1 - x) ** 2 + A * (y - x ** 2) ** 2, "Розенброка", format_coord_Rosenbrock


def format_coord_Rosenbrock(x: np.float64, y: np.float64) -> str:
    return f'x={x:.2f}, y={y:.2f}, f(x, y)={func_Rosenbrock(x, y)[0]:.2f}'

def func_Eggholder(x_array, y_array):
    return -(y_array + 47) * np.sin(np.sqrt(np.abs(x_array / 2 + (y_array + 47)))) - x_array * np.sin(
        np.sqrt(np.abs(x_array - (y_array + 47)))), "Подставка", None

def crossover(parent1, parent2):  # пересечение
    alpha = random.random()
    child_x = alpha * parent1[0] + (1 - alpha) * parent2[0]
    child_y = alpha * parent1[1] + (1 - alpha) * parent2[1]
    return (child_x, child_y)


def mutate(solution, mutation_rate, min_value, max_value):
    child_x, child_y = solution
    if random.random() < mutation_rate:
        child_x += random.uniform(-0.1, 0.1)
        child_y += random.uniform(-0.1, 0.1)
        child_x = max(min(child_x, max_value), min_value)
        child_y = max(min(child_y, max_value), min_value)
    return (child_x, child_y)


def my_initialize_population(population_size, min, max):
    population = []
    for _ in range(population_size):
        x = random.uniform(min, max)
        y = random.uniform(min, max)
        population.append([x, y])
    return population


def calculate_fitness(ind, func):
    x, y = ind[0], ind[1]
    return func(x, y)[0]


if __name__ == "__main__":
    dic_func = {"sine_wave": [func_sine_wave, 0, 5],
                "Himmelblau": [func_Himmelblau, -6, 6],
                "Rastrigin": [func_Rastrigin, -5.12, 5.12],
                "Rosenbrock": [func_Rosenbrock, -3, 3],
                "Eggholder": [func_Eggholder, 0, 5]
                }

    population_size = [6, 12, 24, 48, 96]
    mutation_rate = [0.01, 0.1]
    crossover_rate = [0.1, 0.15, 0.3]
    namesList = ["sine_wave", "Himmelblau", "Rastrigin", "Rosenbrock","Eggholder"]

    num = 1000

    data = []
    for nameFunc in namesList:
        for pop_size in population_size:
            for cross_rate in crossover_rate:
                for mut_rate in mutation_rate:
                    print(f"Функция: {nameFunc}\nРазмер популяции: {pop_size}\nВероятность скрещивания: {cross_rate}\nВероятность мутации: {mut_rate}")
                    func_opt = dic_func[nameFunc][0]
                    min_x = dic_func[nameFunc][1]
                    max_x = dic_func[nameFunc][2]

                    x = np.linspace(min_x, max_x, num)
                    y = np.linspace(min_x, max_x, num)
                    X, Y = np.meshgrid(x, y)

                    f_array = func_opt(X, Y)[0]
                    true_min = f_array.ravel().min()

                    start_time = time.time()
                    population = my_initialize_population(pop_size, min_x, max_x)
                    fitness_scores = None
                    for i in range(20):
                        fitness_scores = [(ind, calculate_fitness(ind, func_opt)) for ind in population]

                        fitness_scores.sort(key=lambda x: x[1])

                        selected_parents = [solution for solution, _ in fitness_scores[:len(fitness_scores) // 2]]

                        new_population = []
                        # print(len(new_population))
                        while len(new_population) < len(fitness_scores):
                            parent1, parent2 = random.choices(selected_parents, k=2)
                            if random.random() < cross_rate:
                                child = crossover(parent1, parent2)
                                new_population.append(mutate(child, mut_rate, min_x, max_x))
                        # print(len(new_population))


                        population = new_population

                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"Время выполнения: {elapsed_time:.4f} секунд")
                    fitness_scores.sort(key=lambda x: x[1])
                    dev = [abs(true_min - ind[1]) for ind in fitness_scores]
                    mean_deviation = sum(dev) / len(dev)
                    print(f"Среднее отклонение от точки минимума: {mean_deviation:.4f}")
                    print()
                    print("----------------------")
                    data.append({
                        'Функция': nameFunc,
                        'Размер популяции': pop_size,
                        'Вероятность скрещивания': cross_rate,
                        'Вероятность мутации': mut_rate,
                        'Время выполнения': f'{elapsed_time:.4f}',
                        'Среднее отклонение от точки минимума': f'{mean_deviation:.4f}'
                    })
    df = pd.DataFrame(data)
    df.to_excel('результаты.xlsx', index=False)