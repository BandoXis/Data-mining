import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


def func_sine_wave(x_array: np.ndarray, y_array: np.ndarray) -> np.ndarray:
    return (x_array - 3.14) ** 2 + (y_array - 2.72) ** 2 + np.sin(3 * x_array + 1.41) + np.sin(
        4 * y_array - 1.73)


def func_Himmelblau(x_array: np.ndarray, y_array: np.ndarray) -> np.ndarray:
    return (x_array ** 2 + y_array - 11) ** 2 + (
            x_array + y_array ** 2 - 7) ** 2


def func_Rastrigin(x: np.ndarray, y: np.ndarray, A: float = 10) -> [np.ndarray]:
    return A + (x ** 2 - A * np.cos(2 * np.pi * x)) + (
            y ** 2 - A * np.cos(2 * np.pi * y))


def func_Eggholder(x_array, y_array):
    return -(y_array + 47) * np.sin(np.sqrt(np.abs(x_array / 2 + (y_array + 47)))) - x_array * np.sin(
        np.sqrt(np.abs(x_array - (y_array + 47))))


def func_Rosenbrock(x: np.ndarray, y: np.ndarray, A: float = 100) -> [np.ndarray, str, callable]:
    return (1 - x) ** 2 + A * (y - x ** 2) ** 2

def calculate_mean_deviation(x_pos_all: np.ndarray, y_pos_all: np.ndarray, x_opt: float, y_opt: float) -> float:
    distances = np.sqrt((x_pos_all - x_opt)**2 + (y_pos_all - y_opt)**2)
    mean_deviation = np.mean(distances)
    return mean_deviation

def display_plot(x_array: np.ndarray, y_array: np.ndarray, f_array: np.ndarray, f_name: str, x_pos: np.ndarray,
                 y_pos: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(f_array, extent=[x_array.min(), x_array.max(), y_array.min(), y_array.max()], origin='lower',
                   cmap='viridis', alpha=0.5)
    plt.colorbar(im)
    x_min = x_array.ravel()[f_array.argmin()]
    y_min = y_array.ravel()[f_array.argmin()]
    ax.plot([x_min], [y_min], marker='x', markersize=10, color="white")
    contours = ax.contour(x_array, y_array, f_array, 10, colors='black', alpha=0.4)
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
    ax.set_title(f"Изолинии функции {f_name}")
    scatter = ax.scatter(x_pos, y_pos, marker='x', color="black", s=5)

    def update(frame):
        scatter.set_offsets(np.column_stack((x_pos[frame], y_pos[frame])))
        # ax.text(0.95, 0.95, f'Iteration: {frame}', transform=ax.transAxes, ha='right', va='top', fontsize=10,
        #         color='black'
        ax.set_title(f'Iteration: {frame}')

    anim = FuncAnimation(fig, update, frames=x_pos.shape[0], interval=200)

    plt.show()

    # scatter = ax.scatter(x_pos, y_pos, marker='x', color="black", s=5)
    # def update(frame):
    #     scatter.set_offsets(np.column_stack((x_pos[frame], y_pos[frame])))
    #     # ax.text(0.95, 0.95, f'Iteration: {frame}', transform=ax.transAxes, ha='right', va='top', fontsize=10,
    #     #         color='black'
    #     ax.set_title( f'Iteration: {frame}')
    # anim = FuncAnimation(fig, update, frames=x_pos.shape[0], interval=100)
    # plt.show()
    # anim.save("my.gif", dpi=120, writer="pillow")


if __name__ == '__main__':
        # name_func = "sine_wave"
    # name_func = "Himmelblau"
    # name_func = "Rastrigin"
    # name_func = "Rosenbrock"
    name_func = "Eggholder"

    dic_func = {"sine_wave": [func_sine_wave, 0, 5],
                "Himmelblau": [func_Himmelblau, -6, 6],
                "Rastrigin": [func_Rastrigin, -5.12, 5.12],
                "Rosenbrock": [func_Rosenbrock, -3, 3],
                "Eggholder": [func_Eggholder, 0, 20]}

    func_opt = dic_func[name_func][0]
    min_x = dic_func[name_func][1]
    max_x = dic_func[name_func][2]

    num = 1000
    x = np.linspace(min_x, max_x, num)
    y = np.linspace(min_x, max_x, num)
    X, Y = np.meshgrid(x, y)
    # print(func_opt, min_x,max_x)
    f_array = func_opt(X, Y)

    population = 30

    OLD_position_x = np.random.uniform(X.min(), X.max(), population)
    OLD_position_y = np.random.uniform(Y.min(), Y.max(), population)
    Fitness = func_opt(OLD_position_x, OLD_position_y)

    array_Velocity_x = np.zeros(population)
    array_Velocity_y = np.zeros(population)

    frequency_max = 2  # максимальное значение частоты
    frequency_min = 1

    a_min = 1
    a_max = 4
    Loudness = a_min + (a_max - a_min) * np.random.rand(population)
    # print(Loudness)

    Pulse_max = 2
    Pulse_min = 0
    Pulse_Rate_start = Pulse_min + (Pulse_max - Pulse_min) * np.random.rand(population)
    Pulse_Rate = np.copy(Pulse_Rate_start)

    Br = 0.9
    Ba = 0.9
    # print(Pulse_Rate_start)
    # print(Pulse_Rate)

    x_pos_all = np.empty((0, population))
    y_pos_all = np.empty((0, population))

    best_index = np.argmin(Fitness)
    best_fit = Fitness.min()

    num_iter = 200
    start_time = time.time()
    for step in range(num_iter):

        # print(Fitness)
        # print(best_fit, best_index)

        for bat in range(population):

            Bat_Frequency = frequency_min + (frequency_max - frequency_min) * np.random.rand()

            array_Velocity_x[bat] += (OLD_position_x[best_index] - OLD_position_x[bat]) * Bat_Frequency
            array_Velocity_y[bat] += (OLD_position_y[best_index] - OLD_position_y[bat]) * Bat_Frequency

            Position_x = OLD_position_x[bat] + array_Velocity_x[bat]
            Position_y = OLD_position_y[bat] + array_Velocity_y[bat]

            Position_x = np.clip(Position_x, X.min(), X.max())
            Position_y = np.clip(Position_y, Y.min(), Y.max())

            # Fitness[bat] = func_opt(OLD_position_x[bat], OLD_position_y[bat])
            # best_fit = Fitness[best_index]

            if Pulse_min + (Pulse_max - Pulse_min) * np.random.rand() >= Pulse_Rate[bat]:
                k = 0
                while k < 5:
                    Position_x = OLD_position_x[bat] + np.mean(Loudness) * np.random.uniform(-1, 1)
                    Position_y = OLD_position_y[bat] + np.mean(Loudness) * np.random.uniform(-1, 1)

                    Position_x = np.clip(Position_x, X.min(), X.max())
                    Position_y = np.clip(Position_y, Y.min(), Y.max())

                    Fitness_value = func_opt(Position_x, Position_y)
                    if Fitness_value <= Fitness[bat]:
                        Fitness[bat] = Fitness_value
                        OLD_position_x[bat] = Position_x
                        OLD_position_y[bat] = Position_y
                        Loudness[bat] *= Ba
                        Pulse_Rate[bat] = Pulse_Rate_start[bat] * (1 - np.exp(-Br * (step + 1)))
                        break
                    k += 1
                if Fitness[bat] < best_fit:
                    best_index = bat
                    best_fit = Fitness[best_index]

        x_pos_all = np.concatenate((x_pos_all, OLD_position_x.reshape(1, population)))
        y_pos_all = np.concatenate((y_pos_all, OLD_position_y.reshape(1, population)))

        # print(array_Velocity_x)
        # print(array_Velocity_y)
        # print(Fitness)
        # print("---------------")
    end_time = time.time()
    execution_time = end_time - start_time
    x_opt = X.ravel()[f_array.argmin()]
    y_opt = Y.ravel()[f_array.argmin()]
    print(f"Время вычисления {name_func} для {num_iter} итераций: {execution_time}")
    mean_deviation = calculate_mean_deviation(x_pos_all, y_pos_all, x_opt, y_opt)
    print(f"Среднее отклонение точек от оптимума: {mean_deviation}")
    # print(array_Velocity)
    display_plot(X, Y, f_array, name_func, x_pos_all, y_pos_all)


