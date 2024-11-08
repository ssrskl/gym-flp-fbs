# 遗传算法测试

from Algorithm.GAAlgorithm import GAAlgorithm


ga = GAAlgorithm(instance="O7-maoyan")

populations = ga._initialize_population()
for indival in populations:
    print(indival.permutationToArray())