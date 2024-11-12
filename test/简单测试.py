def fill_without_duplicates(parent1: list[int], parent2: list[int],
                            startPoint: int,
                            endPoint: int) -> tuple[list[int], list[int]]:
    crossover_part_1 = parent1[startPoint:endPoint + 1]
    crossover_part_2 = parent2[startPoint:endPoint + 1]
    # 获取 parent1 中去除 crossover_part_2 的部分
    parent1_remaining = [
        elem for elem in parent1 if elem not in crossover_part_2
    ]
    # 获取 parent2 中去除 crossover_part_1 的部分
    parent2_remaining = [
        elem for elem in parent2 if elem not in crossover_part_1
    ]

    offspring_1 = parent1_remaining[:
                                    startPoint] + crossover_part_2 + parent1_remaining[
                                        startPoint:]
    offspring_2 = parent2_remaining[:
                                    startPoint] + crossover_part_1 + parent2_remaining[
                                        startPoint:]

    return offspring_1, offspring_2


A = [7, 6, 8, 9, 1, 2, 3, 4, 5]
B = [6, 9, 7, 8, 1, 2, 3, 4, 5]

startPoint = 2
endPoint = 4
print(fill_without_duplicates(A, B, startPoint, endPoint))
