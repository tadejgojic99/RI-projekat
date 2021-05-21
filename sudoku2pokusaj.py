import numpy
import numpy as np
import random
import copy

nDigits = 9

def Check(a):
    s = len(set(a))
    return (1.0/s)/nDigits, np.zeros(nDigits)

def initialize(individual):
    m = []
    row = numpy.arange(1, 10)

    random.shuffle(row)

    m.append(row)
    for i in range(1, 9):
        if i % 3 == 0:
            m.append(numpy.roll(m[-1], -1))
        else:
            m.append(numpy.roll(m[-1], -3))

    return m

class sudokuPuzzle:
    def __init__(self, originalDict):
        self.board = initialize(self)
        self.original = copy.deepcopy(originalDict)
        self.fitness = self.fitnessFunction()
    def __lt__(self, other):
        return self.fitness < other.fitness

    def fitnessFunction(self):
        val = 0
        for pos in self.original:
            (posI, posJ) = pos
            if self.board[posI][posJ] != self.original[(posI, posJ)]:
                val += 1
        return val

    def isRowFeasible(self, r, val):
        for j in range(0, nDigits):
            if self.board[r][j] == val:
                return False
        return True

    def isColumnFeasible(self, c, val):
        for i in range(0, nDigits):
            if self.board[i][c] == val:
                return False
        return True

    def isSubBoardFeasible(self, r, c, val):
        iStart = (r / 3) * 3
        jStart = (c / 3) * 3
        for i in range(iStart, iStart + 3):
            for j in range(jStart, jStart + 3):
                if self.board[i][j] == val:
                    return False
        return True

    def isFeasible(self, i1, j1, i2, j2, ind):
        if ind == 1:
            return self.isColumnFeasible(j1, self.board[i2][j2]) and self.isSubBoardFeasible(i1, j1, self.board[i2][j2])
        else:
            return self.isRowFeasible(i1, self.board[i2][j2]) and self.isColumnFeasible(j1, self.board[i2][
                j2]) and self.isSubBoardFeasible(i1, j1, self.board[i2][j2])


def mutation(individual, mutationRate):
    x = random.uniform(0,1)
    if x<mutationRate:
        if random.uniform(0,1) < 0.5:
            #vrste
            subSq = random.randrange(3)
            i = random.randrange(3)
            j = random.randrange(3)
            i = 3*subSq + i
            j = 3*subSq + j
            tmp = individual.board[i]
            individual.board[i] = individual.board[j]
            individual.board[j] = tmp
        else:
            #kolone
            subSq = random.randrange(3)
            i = random.randrange(3)
            j = random.randrange(3)
            i = 3 * subSq + i
            j = 3 * subSq + j
            for k in range(nDigits):
                tmp = individual.board[k][i]
                individual.board[k][i] = individual.board[k][j]
                individual.board[k][j] = tmp

def selection(population, tournamentRate):
    bestVal = float('inf')
    bestIndividual = -1
    n = len(population)
    for i in range(tournamentRate):
        j = random.randrange(n)
        if population[j].fitness < bestVal:
            bestVal = population[j].fitness
            bestIndividual = j

    return j

# 0# 0 1 2 3 4 5 6 7 8 0 0 + 0=0    1 + 0   2 + 0
# 1# 0 1 2 3 4 5 6 7 8
# 2# 0 1 2 3 4 5 6 7 8
# 3# 0 1 2 3 4 5 6 7 8 1*3 + 0  1*3 + 1   1*3 + 2
# 4# 0 1 2 3 4 5 6 7 8
# 5# 0 1 2 3 4 5 6 7 8
# 6# 0 1 2 3 4 5 6 7 8 2*3 + 0
# 7# 0 1 2 3 4 5 6 7 8
# 8# 0 1 2 3 4 5 6 7 8
def finish(individual):
    rowSets = [set(individual.board[i]) for i in range(nDigits)]
    columnSets = [set() for i in range(nDigits)]
    subSquareSets = [set() for i in range(nDigits)]
    for i in range(nDigits):
        for j in range(nDigits):
            columnSets[i].add(individual.board[j][i])
    #posto se kopiraju cele podtable skupovi za njih mogu biti prazni, pa cemo za svaku podtablu koja je bila prazna
    #samo dodavati tekuce elemente tj one koje smo u tom momentu postavili
    for i in range(nDigits):
        for j in range(nDigits):
            if individual.board[i][j] == 0:
                for num in range(1, 10):
                    if num not in rowSets[i] and num not in columnSets[j] and num not in subSquareSets[int(i/3)*2 + int(j/3)]:
                        individual.board[i][j] = num
                        rowSets[i].add(num)
                        columnSets[j].add(num)
                        subSquareSets[int(i / 3) * 2 + int(j / 3)].add(num)


def crossover(parent1, parent2, child1, child2):
    i = random.randrange(3)

    child1.board = np.zeros((nDigits, nDigits))
    child2.board = np.zeros((nDigits, nDigits))

    firstParentLimit = i*3 + 3

    for i in range(nDigits):
        for j in range(nDigits):
            if i < firstParentLimit and j < firstParentLimit:
                child1.board[i][j] = parent1.board[i][j]
                child2.board[i][j] = parent2.board[i][j]
            if i >= firstParentLimit and j >= firstParentLimit:
                child1.board[i][j] = parent2.board[i][j]
                child2.board[i][j] = parent1.board[i][j]

    finish(child1)
    finish(child2)


def GA(newPopulationSize, eliteSize, maxIters, mutationProbability, tournamentSize, originalD):
    population = []
    newPopulation = []
    duringIterations = False
    iterNum = 0
    for i in range(newPopulationSize):
        population.append(sudokuPuzzle(originalD))
        newPopulation.append(sudokuPuzzle(originalD))

    for i in range(maxIters):

        population.sort()
        # if i%500:
        #     print(population[0].fitness)
        if population[0].fitness == 0:
            print("Pronadjeno tokom {0:d}} iteracije!".format(i))
            duringIterations = True
            break
        for j in range(eliteSize):
            newPopulation[j] = population[j]
        for j in range(eliteSize, newPopulationSize, 2):
            k1 = selection(population, tournamentSize)
            k2 = selection(population, tournamentSize)
            crossover(population[k1], population[k2], newPopulation[j], newPopulation[j + 1])
            if i == 0 and j == eliteSize:
                print(population[k1].board)
                print(population[k2].board)
                print(newPopulation[j].board)
                print(newPopulation[j+1].board)
            mutation(newPopulation[j], mutationProbability)
            mutation(newPopulation[j + 1], mutationProbability)
            newPopulation[j].fitness = newPopulation[j].fitnessFunction()
            newPopulation[j + 1].fitness = newPopulation[j + 1].fitnessFunction()
        population[:] = newPopulation[:]

    population.sort()
    if not duringIterations:
        print("Pronadjeno je (najblize)resenje u poslednjoj iteraciji")
    print("Najblize resenje je:")
    for i in range(nDigits):
        print(population[0].board[i])


original = [[0, 0, 0, 0, 0, 1, 0, 9, 0],
            [5, 0, 0, 0, 0, 0, 6, 0, 0],
            [0, 0, 9, 3, 7, 0, 0, 0, 0],
            [9, 0, 6, 0, 5, 0, 3, 0, 0],
            [1, 0, 5, 9, 0, 0, 7, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 2, 0, 1, 7],
            [0, 0, 0, 0, 4, 0, 0, 0, 0],
            [3, 0, 0, 1, 9, 6, 0, 2, 0]
            ]
originalD = {}
for i in range(9):
    for j in range(9):
        if original[i][j] != 0:
            originalD[(i, j)] = original[i][j]

GA(150, 50, 10000, 0.5, 10, originalD)