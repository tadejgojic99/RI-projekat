import numpy as np
import random
import copy

nDigits = 9
def Check(a):
    s = len(set(a))
    return (1.0/s)/nDigits, np.zeros(nDigits)

def initialize(individual):
    m = original
    for i in range(nDigits):
        rowSet = set(original[i])
        for j in range(nDigits):
            if m[i][j] == 0:
                val = random.randrange(1, 10)
                while val in rowSet:
                    val = random.randrange(1, 10)
                m[i][j] = val
                rowSet.add(val)
    return m

class sudokuPuzzle:
    def __init__(self):
        self.board = initialize(self)
        self.fitness = self.fitnessFunction()

    def __lt__(self, other):
        return self.fitness > other.fitness

    def fitnessFunction(self):
        countRows = np.zeros(nDigits)
        countColumns = np.zeros(nDigits)
        countSubBoard = np.zeros(nDigits)
        sumRow = 0
        sumColumn = 0
        sumSubBoard = 0

        for i in range(nDigits):
            for j in range(nDigits):
                countRows[self.board[i][j] - 1] += 1
                countColumns[self.board[j][i] - 1] += 1
            sumRow, countRows = Check(countRows)
            sumColumn, countColumns = Check(countColumns)
        for i in range(0, nDigits, 3):
            for j in range(0, nDigits, 3):
                for x in range(0, 3):
                    for y in range(0, 3):
                        countSubBoard[self.board[i + x][j + y] - 1] += 1

                sumSubBoard, countSubBoard = Check(countSubBoard)
        
        return 1.0*(sumColumn + sumSubBoard + sumRow) / 3
 
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
        iStart = (r/3)*3
        jStart = (c/3)*3
        for i in range(iStart, iStart + 3):
            for j in range(jStart, jStart + 3):
                if self.board[i][j] == val:
                    return False
        return True

    def isFeasible(self, i1, j1, i2, j2, ind):
        if ind == 1:
            return self.isColumnFeasible(j1, self.board[i2][j2]) and self.isSubBoardFeasible(i1, j1, self.board[i2][j2])
        else:
            return self.isRowFeasible(i1, self.board[i2][j2]) and self.isColumnFeasible(j1, self.board[i2][j2]) and self.isSubBoardFeasible(i1, j1, self.board[i2][j2])

def mutation(individual, mutationProbability):
    ind = random.uniform(0, 1)
    if ind > mutationProbability:
        return
    else:
        r = random.randrange(0, 9)
        isOk = False
        while(isOk == False):
            r = random.randint(0, 8)
            c1 = random.randint(0, 8)
            random.seed()
            c2 = int(random.random()*10) % 9
            while(c1 == c2):
                print("KUKURIKU", c1, c2)
                random.seed()
                c1 = random.randint(0, 8)
                c2 = random.randint(0, 8)

            if original[r][c1] == 0 and original[r][c2] == 0 and individual.isFeasible(r, c1, r, c2, 1) and individual.isFeasible(r, c2, r, c1, 1):
                break
                tmp = individual.board[r][c1]
                individual.board[r][c1] = individual.board[r][c2]
                individual.board[r][c2] = tmp

def selection(population, tournamentSize):
    bestFitness = float('inf')
    bestCoord = -1
    n = len(population)
    for i in range(tournamentSize):
        j = random.randrange(n)
        if population[j].fitness < bestFitness:
            bestCoord = j
            bestFitness = population[j].fitness
    
    return bestCoord

def crossover(parent1, parent2, child1, child2):
    n = len(parent1.board[0])
    k = random.randrange(0, n)

    child1.board = copy.deepcopy(parent1.board)
    child2.board = copy.deepcopy(parent2.board)
    for i in range(k):
        child1.board[i] = copy.deepcopy(parent2.board[i])
        child2.board[i] = copy.deepcopy(parent1.board[i])
    #print(parent1.board)
    #print(parent2.board)
    #print(child1.board)
    #print(child2.board)


def GA(newPopulationSize, eliteSize, maxIters, mutationProbability, tournamentSize):
    population = []
    newPopulation = []
    duringIterations = False
    iterNum = 0
    for i in range(newPopulationSize):
        population.append(sudokuPuzzle())
        newPopulation.append(sudokuPuzzle())
    
    for i in range(maxIters):
        population.sort()
        if population[0].fitness == 1.0:
            print("Pronadjeno tokom {0:d}} iteracije!".format(i))
            duringIterations = True
            break
        for j in range(eliteSize):
            newPopulation[j] = population[j]
        for j in range(eliteSize, newPopulationSize, 2):
            k1 = selection(population, tournamentSize)
            k2 = selection(population, tournamentSize)
            crossover(population[k1], population[k2], newPopulation[j], newPopulation[j+1])
            mutation(newPopulation[j], mutationProbability)
            mutation(newPopulation[j+1], mutationProbability)
            newPopulation[j].fitness = newPopulation[j].fitnessFunction()
            newPopulation[j+1].fitness = newPopulation[j+1].fitnessFunction()
        population[:] = newPopulation[:]

    population.sort()
    if not duringIterations:
        print("Pronadjeno je (najblize)resenje u poslednjoj iteraciji")
    print("Najblize resenje je:")
    for i in range(nDigits):
            print(population[0].board[i])


original = [ [0,0,0, 0,0,1, 0,9,0],
             [5,0,0, 0,0,0, 6,0,0],
             [0,0,9, 3,7,0, 0,0,0],
             [9,0,6, 0,5,0, 3,0,0],
             [1,0,5, 9,0,0, 7,0,0],
             [0,2,0, 0,0,0, 0,0,0],
             [0,0,0, 0,0,2, 0,1,7],
             [0,0,0, 0,4,0, 0,0,0],
             [3,0,0, 1,9,6, 0,2,0]
           ]


GA(150, 50, 500, 0.05, 10)