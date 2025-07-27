import math
import time

import numpy as np
import random
import operator
import matplotlib.pyplot as plt
import pandas as pd


class City:
    def __init__(self, index: int, max_x: float, max_y: float):
        self.index = index
        # self.x = random.uniform(0, max_x)
        # self.y = random.uniform(0, max_y)
        self.x = max_x
        self.y = max_y

    def calculate_distance(self, other_city):
        return np.hypot(self.x - other_city.x, self.y - other_city.y)

    def __repr__(self):
        return f"City(index={self.index}, x={self.x:.2f}, y={self.y:.2f})"


def create_fixed_cities():
    return [
        City(1, 0),
        City(0, 1),
        City(1, 1),
        City(0, 0),
        City(0, 2),
        City(0, 3),
    ]


def create_cities(count, max_x, max_y):
    return [City(i, max_x, max_y) for i in range(count)]


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def route_distance(self):
        if self.distance == 0:
            self.distance = sum(
                self.route[i].calculate_distance(self.route[i + 1])
                for i in range(len(self.route) - 1)
            ) + self.route[-1].calculate_distance(self.route[0])
        return self.distance

    def route_fitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.route_distance())
        return self.fitness


def generate_route(city_list):
    return random.sample(city_list, len(city_list))


def initial_population(pop_size, city_list):
    return [generate_route(city_list) for _ in range(pop_size)]


def rank_routes(population):
    fitness_results = [(i, Fitness(route).route_fitness(), route) for i, route in enumerate(population)]
    return sorted(fitness_results, key=operator.itemgetter(1), reverse=True)


def breed(parent1, parent2):
    child_p1 = []
    gene_a = int(random.random() * len(parent1))
    gene_b = int(random.random() * len(parent1))
    start_gene, end_gene = min(gene_a, gene_b), max(gene_a, gene_b)

    for i in range(start_gene, end_gene):
        child_p1.append(parent1[i])

    child_p2 = [gene for gene in parent2 if gene not in child_p1]

    return child_p1 + child_p2


def mutate(individual, mutation_rate):
    for swapped in range(len(individual)):
        if random.random() < mutation_rate:
            swap_with = int(random.random() * len(individual))
            individual[swapped], individual[swap_with] = individual[swap_with], individual[swapped]
    return individual


def mutate_population(population, mutation_rate):
    return [mutate(individual, mutation_rate) for individual in population]


def next_generation(current_gen, elite_size, mutation_rate):
    pop_ranked = rank_routes(current_gen)
    selected_parents = [pop_ranked[i][2] for i in range(len(current_gen) // 4)]
    next_gen = []

    next_gen.extend(current_gen[:elite_size])

    while len(next_gen) < len(current_gen):
        parent1, parent2 = random.choices(selected_parents, k=2)
        if random.random() < 0.5:
            child = breed(parent1, parent2)
            next_gen.append(child)

    next_gen = mutate_population(next_gen, mutation_rate)
    return next_gen


def plot_routes(routes, iteration):
    plt.clf()
    num_routes = len(routes)

    for idx, route in enumerate(routes):
        plt.figure(idx + 1)
        plt.clf()
        plt.scatter([city.x for city in route], [city.y for city in route], color='red', edgecolor='black')
        for i in range(len(route)):
            plt.plot([route[i - 1].x, route[i].x], [route[i - 1].y, route[i].y], 'b-')
            plt.text(route[i].x, route[i].y, f"{i + 1}", fontsize=10, ha='right', va='top')
        plt.plot([route[-1].x, route[0].x], [route[-1].y, route[0].y], 'b-')
        plt.title(f'Лучший путь {idx + 1} на итерации: {iteration}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.pause(0.001)


def plot_route2(routes, iteration):
    route = routes[0]
    plt.clf()
    plt.scatter([city.x for city in route], [city.y for city in route], color='red')
    for i in range(len(route)):
        plt.plot([route[i - 1].x, route[i].x], [route[i - 1].y, route[i].y], 'b-')
        plt.text(route[i].x, route[i].y, f"{i + 1}", fontsize=15, ha='right', va='top')
    plt.plot([route[-1].x, route[0].x], [route[-1].y, route[0].y], 'b-')
    plt.title(f'Лучший путь на итерации: {iteration}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.pause(0.4)


def plot_final_route(route):
    plt.figure()
    plt.scatter([city.x for city in route], [city.y for city in route], color='red')
    for i in range(len(route)):
        plt.plot([route[i - 1].x, route[i].x], [route[i - 1].y, route[i].y], 'b-')
        plt.text(route[i].x, route[i].y, f"{route[i].index}", fontsize=15, ha='right', va='top')
    plt.plot([route[-1].x, route[0].x], [route[-1].y, route[0].y], 'b-')
    plt.title('Лучший маршрут')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def genetic_algorithm(city_list, pop_size, elite_size, mutation_rate, generations, num_best_routes=3,
                      convergence_threshold=10):
    population = initial_population(pop_size, city_list)
    print(f"Начальная дистанция: {1 / rank_routes(population)[0][1]:.2f}")
    best_route_list = []
    best_route_list.append(1 / rank_routes(population)[0][1])
    last_best_distance = 1 / rank_routes(population)[0][1]
    unchanged_generations = 0

    for i in range(generations):
        population = next_generation(population, elite_size, mutation_rate)
        best_distance = 1 / rank_routes(population)[0][1]
        best_route_list.append(best_distance)
        if best_distance == last_best_distance:
            unchanged_generations += 1
        else:
            unchanged_generations = 0
            last_best_distance = best_distance
        if unchanged_generations >= convergence_threshold:
            print(f"Достигнут порог сходимости на итерации {i + 1}. Прерываем процесс.")
            break

        best_routes_indices = [rank_routes(population)[j][0] for j in range(num_best_routes)]
        best_routes = [population[index] for index in best_routes_indices]
        # plot_route2(best_routes, i + 1)

    print(f"Финальная дистанция: {best_distance:.2f}")
    best_route_index = rank_routes(population)[0][0]
    best_route = population[best_route_index]
    return best_route, best_route_list,best_distance


def create_triangle_cities():
    cities = []
    side_length = 100
    height = (np.sqrt(3) / 2) * side_length

    vertices = [
        (0, 0),
        (side_length, 0),
        (side_length / 2, height)
    ]

    # Adding vertices to cities list
    for i, (x, y) in enumerate(vertices):
        cities.append(City(i, x, y))

    # Adding remaining cities along the perimeter of the triangle
    points_per_side = 3
    for i in range(1, points_per_side):
        # Side 1
        cities.append(City(len(cities), i * (side_length / points_per_side), 0))
        # Side 2
        cities.append(
            City(len(cities), side_length - i * (side_length / points_per_side), i * (height / points_per_side)))
        # Side 3
        cities.append(City(len(cities), (side_length / 2) - i * (side_length / (2 * points_per_side)),
                           height - i * (height / points_per_side)))

    return cities


def create_circle_cities(city_count, radius=50):
    cities = []
    t_array = np.linspace(0, 2 * math.pi, city_count, endpoint=False)
    mid_x = 100
    mid_y = 100
    dist = 0
    for i, t in enumerate(t_array):
        x = mid_x + radius * math.cos(t)
        y = mid_y + radius * math.sin(t)
        cities.append(City(i, x, y))
        if len(cities) > 1:
            dist += cities[-2].calculate_distance(cities[-1])
    dist += cities[-1].calculate_distance(cities[0])
    return cities,dist


if __name__ == "__main__":
    # max_x, max_y = 250, 250

    city_counts = [5, 10, 15, 20, 40]
    population_size = [40, 80, 160,320]
    mutation_rate = [0.01, 0.1]

    data = []
    for city_count in city_counts:
        for pop_size in population_size:
            for mut_rate in mutation_rate:
                start_time = time.time()

                print(
                    f"Число городов: {city_count}\nРазмер популяции: {pop_size}\nВероятность мутации: {mut_rate}")

                city_list,best_lenght = create_circle_cities(city_count)

                best_route, best_route_list,best_distance = genetic_algorithm(city_list=city_list, pop_size=pop_size, elite_size=20,
                                                                mutation_rate=mut_rate, generations=90,
                                                                num_best_routes=3)

                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Длина окруждности: {best_lenght}")
                print(f"Время выполнения: {elapsed_time:.4f} секунд")
                print()
                print("----------------------")
                data.append({
                    'Число городов': city_count,
                    'Размер популяции': pop_size,
                    'Вероятность мутации': mut_rate,
                    'Время выполнения': f'{elapsed_time:.4f}',
                    'Длина окружности': f'{best_lenght:.2f}',
                    'Финальная дистанция': f'{best_distance:.2f}'
                })
    df = pd.DataFrame(data)
    df.to_excel('результаты_ЛР_3.xlsx', index=False)
    # city_list = create_cities(city_count, max_x, max_y)
    # city_list = create_triangle_cities()
    # city_list = create_circle_cities(city_count)
    # # print(city_list[0])
    # plt.ion()
    # best_route, best_route_list = genetic_algorithm(city_list=city_list, pop_size=200, elite_size=20,
    #                                                  mutation_rate=0.01, generations=90, num_best_routes=3)
    # plt.ioff()
    # plt.show()
    #
    # plt.figure()
    # plt.plot(range(len(best_route_list)), best_route_list)
    # plt.title('График изменений дистанции')
    # plt.xlabel('Номер итерации')
    # plt.ylabel('Дистанция')
    # plt.show()
    #
    # best_route_indices = [city.index for city in best_route]
    # print(f"Лучший маршрут: {best_route_indices}")
    #
    # plot_final_route(best_route)
