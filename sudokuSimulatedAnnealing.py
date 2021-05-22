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

def invert(individual):
    if random.uniform(0, 1) < 0.5:
        # vrste
        subSq = random.randrange(3)
        i = random.randrange(3)
        j = random.randrange(3)
        i = 3 * subSq + i
        j = 3 * subSq + j
        tmp = individual.board[i]
        individual.board[i] = individual.board[j]
        individual.board[j] = tmp

        return i, j, True
    else:
        # kolone
        subSq = random.randrange(3)
        i = random.randrange(3)
        j = random.randrange(3)
        i = 3 * subSq + i
        j = 3 * subSq + j
        tmp = individual.board[:][i]
        individual.board[:][i] = individual.board[:][j]
        individual.board[:][j] = tmp

        return i, j, False


def reverse(individual, i, j, isRow):
    if isRow == True:
        tmp = individual.board[i][:]
        individual.board[i][:] = individual.board[j][:]
        individual.board[j][:] = tmp
    else:
        for k in range(9):
            tmp = individual.board[k][i]
            individual.board[k][i] = individual.board[k][j]
            individual.board[k][j] = tmp

def SimulatedAnnealing(solution, maxIter):
    currentValue = fitnessFunction(solution)
    bestValue = currentValue
    i = 1
    while i < maxIter:
        i1, j1, isRow = invert(solution)
        newValue = fitnessFunction(solution)
        if newValue < bestValue:
            bestValue = newValue
        else:
            p = 1.0/i**0.5
            q = random.uniform(0,1)
            if p>q:
                currentValue = newValue
            else:
                reverse(solution, i1, j1, isRow)
        if newValue < bestValue:
            bestValue = newValue
        i += 1
    return bestValue, solution

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

solution = sudokuPuzzle(originalD)
value, sol = SimulatedAnnealing(solution, 150000)
print(value)
print(sol.board)