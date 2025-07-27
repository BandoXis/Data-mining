import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


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


if __name__ == '__main__':
    name_func = "sine_wave"
    # name_func = "Himmelblau"
    # name_func = "Rastrigin"
    # name_func = "Rosenbrock"
    # name_func = "Eggholder"

    dic_func = {"sine_wave": [func_sine_wave, 0, 5],
                "Himmelblau": [func_Himmelblau, -6, 6],
                "Rastrigin": [func_Rastrigin, -5.12, 5.12],
                "Rosenbrock": [func_Rosenbrock, -3, 3]
                 # "Eggholder": [func_Eggholder, 0, 5]
                }

    func_opt = dic_func[name_func][0]
    min_x = dic_func[name_func][1]
    max_x = dic_func[name_func][2]

    num = 1000
    x = np.linspace(min_x, max_x, num)
    y = np.linspace(min_x, max_x, num)
    X, Y = np.meshgrid(x, y)

    f_array = func_opt(X, Y)[0]


    true_min = f_array.ravel().min()


    population_size = 100
    mutation_rate = 0.1
    crossover_rate = 0.3

    start_time = time.time()
    population = my_initialize_population(population_size, min_x, max_x)


    fig, ax = plt.subplots()
    im = ax.imshow(f_array, extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower',
                   cmap='viridis', alpha=0.5)
    plt.colorbar(im)
    contours = ax.contour(X, Y, f_array, 10, colors='black', alpha=0.4)

    x_min = X.ravel()[f_array.argmin()]
    y_min = Y.ravel()[f_array.argmin()]
    ax.plot([x_min], [y_min], marker='x', markersize=10, color="white")

    scatter = ax.scatter([ind[0] for ind in population], [ind[1] for ind in population], marker='x', color="black", s=5)
    iteration = 0


    def update(frame):
        global population
        global iteration
        fitness_scores = [(ind, calculate_fitness(ind, func_opt)) for ind in population]

        fitness_scores.sort(key=lambda x: x[1])

        selected_parents = [solution for solution, _ in fitness_scores[:len(fitness_scores) // 2]]

        new_population = []
        print(len(new_population))
        while len(new_population) < len(fitness_scores):
            parent1, parent2 = random.choices(selected_parents, k=2)
            if random.random() < crossover_rate:
                child = crossover(parent1, parent2)
                new_population.append(mutate(child, mutation_rate, min_x, max_x))
        print(len(new_population))
        print("----------")

        population = new_population

        scatter.set_offsets(np.column_stack(([ind[0] for ind in population], [ind[1] for ind in population])))
        ax.set_title(f'Iteration: {iteration}')

        if iteration > 20:
            ani.event_source.stop()
            fitness_scores.sort(key=lambda x: x[1])
            print(fitness_scores)
            print(fitness_scores[0][0], fitness_scores[0][1])
        iteration += 1
        return scatter,


    ani = FuncAnimation(fig, update, frames=range(20), interval=200)
    plt.show()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Время выполнения: {elapsed_time:.4f} секунд")
