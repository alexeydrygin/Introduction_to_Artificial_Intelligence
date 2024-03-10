
# Этот код представляет собой реализацию генетического алгоритма на Python для решения задачи оптимизации путешествия между точками. Генетический алгоритм - это метод оптимизации, вдохновленный биологической эволюцией, который использует операции скрещивания и мутации для поиска оптимального решения задачи.

# 1. **Инициализация популяции**: Функция `generate_population` создает начальную популяцию из случайных бинарных строк, представляющих возможные маршруты между точками. Размер популяции и количество точек задаются параметрами функции.

# 2. **Определение целевой функции**: Функция `fitness` вычисляет пригодность маршрута, в данном случае считая количество посещенных точек. Предполагается, что время путешествия между каждой парой точек равно 1.

# 3. **Мутация**: Функция `mutate` вносит случайные изменения в хромосому, меняя 0 на 1 и наоборот с вероятностью, определенной параметром `mutation_rate`.

# 4. **Скрещивание**: Функция `crossover` выполняет скрещивание двух родительских хромосом, создавая два потомка. Скрещивание происходит в случайной точке, и каждый потомок получает часть генома от каждого родителя.

# 5. **Генетический алгоритм**: Функция `genetic_algorithm` реализует основной цикл генетического алгоритма. В каждой генерации популяция сортируется по пригодности, сохраняются лучшие два индивида, и остальные индивиды создаются через скрещивание и мутацию. Этот процесс повторяется заданное количество раз (`num_generations`).

# 6. **Параметры задачи**: Задаются параметры задачи, такие как размер популяции, количество точек, количество поколений и вероятность мутации.

# 7. **Запуск генетического алгоритма**: Инициализируется популяция и запускается генетический алгоритм.

# 8. **Вывод результата**: В конце работы алгоритма выводится лучший найденный маршрут.

# Этот код демонстрирует базовую структуру и принципы работы генетического алгоритма, который может быть адаптирован и модифицирован для решения различных задач оптимизации.


import random

# Шаг 1: Инициализация популяции
def generate_population(population_size, num_points):
    return [random.randint(0, 1) for _ in range(population_size * num_points)]

# Шаг 2: Определение целевой функции
def fitness(chromosome):
    # Предположим, что время путешествия между каждым пунктом равно 1
    return sum(chromosome)

# Шаг 3: Мутация
def mutate(chromosome, mutation_rate):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]  # Меняем 0 на 1 и наоборот
    return chromosome

# Шаг 4: Скрещивание
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Шаг 5: Генетический алгоритм
def genetic_algorithm(population, num_generations, mutation_rate):
    for generation in range(num_generations):
        population.sort(key=fitness, reverse=True)  # Сортировка по пригодности
        new_population = population[:2]  # Сохраняем лучшие два индивида

        while len(new_population) < len(population):
            # Выбираем двух родителей из лучших 10
            parent1 = random.choice(population[:10])
            parent2 = random.choice(population[:10])
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])

        population = new_population

    return population[0]  # Возвращаем лучшую хромосому


# Шаг 6: Параметры задачи
population_size = 1
num_points = 20
num_generations = 10
mutation_rate = 0.01

# Инициализация популяции
population = [generate_population(population_size, num_points)
              for _ in range(population_size)]

# Шаг 7: Запуск генетического алгоритма
best_route = genetic_algorithm(population, num_generations, mutation_rate)

# Шаг 8: Вывод результата
print("Лучший маршрут:", best_route)


