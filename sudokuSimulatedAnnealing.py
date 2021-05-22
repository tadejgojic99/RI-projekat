import numpy
import numpy as np
import random
import copy

nDigits = 9

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


def fitnessFunction(individual):
    val = 0
    for pos in individual.original:
        (posI, posJ) = pos
        num = individual.original[(posI, posJ)]
        if individual.board[posI][posJ] != num:
            val += findValueInSameSubsquare(individual, num, posI, posJ)
    return val



def findValueInSameSubsquare(individual, num, i1, j1):
    # i i j su nam koord pocetka tj gornjeg levog polja kvadranta u kom trazimo vrednost
    i = 3 * int(i1 / 3)
    j = 3 * int(j1 / 3)
    endI = i + 3
    endJ = j + 3
    val = 0
    while i < endI:
        while j < endJ:
            if individual.board[i][j] == num:
                val = abs(i1 - i) + abs(j1 - j)
            j += 1
        i += 1
    return val

def invertRow(individual):
        # vrste
    subSq = random.randrange(3)
    i = random.randrange(3)
    j = random.randrange(3)
    i = 3 * subSq + i
    j = 3 * subSq + j
    swap(individual,i,j,True)
    return i, j
def invertColumn(individual):
        # kolone
    subSq = random.randrange(3)
    i = random.randrange(3)
    j = random.randrange(3)
    i = 3 * subSq + i
    j = 3 * subSq + j
    swap(individual,i,j,False)
    return i, j

def swap(individual, i, j, isRow):
    if isRow:
        tmp = copy.deepcopy(individual.board[i])
        individual.board[i] = copy.deepcopy(individual.board[j][:])
        individual.board[j] = copy.deepcopy(tmp)
    else:
        tmp = copy.deepcopy(individual.board[:][i])
        individual.board[:][i] = copy.deepcopy(individual.board[:][j])
        individual.board[:][j] = copy.deepcopy(tmp)

def SimulatedAnnealing(solution):
    currentValue = fitnessFunction(solution)
    bestValue = currentValue
    bestSol = copy.deepcopy(solution)
    i=1
    while i<20000:
        #invertujemo vrste
        i1, j1 = invertRow(solution)
        newValue = fitnessFunction(solution)
        if newValue == 0:
            return newValue, solution, i
        if newValue < bestValue:
            bestValue = newValue
            bestSol = copy.deepcopy(solution)
        else:
            p = 1.0/i**0.5
            q = random.uniform(0,1)
            if p<q:
                currentValue = newValue
            else:
                swap(solution, i1, j1, True)
        if newValue < bestValue:
            bestValue = newValue
            bestSol = copy.deepcopy(solution)
        #invertujemoKolone
        i1, j1 = invertColumn(solution)
        newValue = fitnessFunction(solution)
        if newValue == 0:
            return newValue, solution, i
        if newValue < bestValue:
            bestValue = newValue
            bestSol = copy.deepcopy(solution)

        else:
            p = 1.0 / i ** 0.5
            q = random.uniform(0, 1)
            if p < q:
                currentValue = newValue
            else:
                swap(solution, i1, j1, False)
        if newValue < bestValue:
            bestValue = newValue
            bestSol = copy.deepcopy(solution)


        i += 1

    return bestValue, bestSol, i
original = [
        [8,0,0,0,0,0,0,0,0],
        [0,0,3,6,0,0,0,0,0],
        [0,7,0,0,9,0,2,0,0],
        [0,5,0,0,0,7,0,0,0],
        [0,0,0,0,4,5,7,0,0],
        [0,0,0,1,0,0,0,3,0],
        [0,0,1,0,0,0,0,6,8],
        [0,0,8,5,0,0,0,1,0],
        [0,9,0,0,0,0,4,0,0]
    ]

originalD = {}
for i in range(9):
    for j in range(9):
        if original[i][j] != 0:
            originalD[(i, j)] = original[i][j]
print("start")
solution = sudokuPuzzle(originalD)
value, sol, iteration = SimulatedAnnealing(solution)
print(value)
for i in range(nDigits):
    res = ""
    for j in range(nDigits):
        res += str(sol.board[i][j]) + " "
    print(res)
print(iteration,"-ta iteracija")