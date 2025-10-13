## ex 1 - Parsing the system of equations
import copy
import math
import pathlib

def load_system(path: pathlib.Path) -> tuple[list[list[float]], list[float]]:
    A: list[list[float]] = []
    B: list[float] = []
    with open(path, "r") as file:
        for line in file:
            line = line.strip()

            left, right = line.split("=")
            B.append(float(right))
            comps = left.split(" ")

            sign : float = 1.0
            row :list[float] = []
            for comp in comps:
                if comp == '+':
                    sign = 1.0
                elif comp == '-':
                    sign = -1.0
                else:
                    if comp.endswith("x"):
                        nr = comp[:-1]
                        if not nr:
                            nr = 1.0
                        row.append(sign*(float(nr)))
                    elif comp.endswith("y"):
                        nr = comp[:-1]
                        if not nr:
                            nr = 1.0
                        row.append(sign * (float(nr)))
                    elif comp.endswith("z"):
                        nr = comp[:-1]
                        if not nr:
                            nr = 1.0
                        row.append(sign * (float(nr)))
            A.append(row)

    return A,B
# macOS path
A, B = load_system(pathlib.Path("/Users/marius/PycharmProjects/Neural-Networks-Template-2025/Lab02/system.txt"))

# path
#A, B = load_system(pathlib.Path("system.txt"))

print(f"{A=} {B=}")

## ex 2.1 - Determinant of a matrix 3x3

def determinant(matrix: list[list[float]]) -> float:
    a,b,c = matrix[0]
    d,e,f = matrix[1]
    g,h,i = matrix[2]

    return a*e*i + b*f*g + c * d * h - (g*e*c + h*f*a + i*d*b)

print(f"{determinant(A)=}")

## ex 2.2 - Trace
def trace(matrix: list[list[float]]) -> float:
    return matrix[0][0] + matrix[1][1] + matrix[2][2]

print(f"{trace(A)=}")

## ex 2.3 - Vector norm

def norm(vector: list[float]) -> float:
    return math.sqrt(vector[0]*vector[0] + vector[1]*vector[1] + vector[2]*vector[2])

print(f"{norm(B)=}")

## ex 2.4 - Transpose of matrix
def transpose(matrix: list[list[float]]) -> list[list[float]]:
    rows = len(matrix)
    cols = len(matrix[0])

    At : list[list[float]] = []
    for c in range(cols):
        row : list[float] = []
        for r in range(rows):
            row.append(matrix[r][c])
        At.append(row)
    return At

print(f"{transpose(A)=}")

## ex 2.5 - Matrix-vector multiplication
def multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    res : list[float] = []
    if len(matrix[0]) != len(vector):
        raise TypeError("Invalid multiplication")

    for i in range(len(matrix)):
        s : float = 0
        for j in range(len(matrix[i])):
            s = s + matrix[i][j]*vector[j]
        res.append(s)
    return res

print(f"{multiply(A, B)=}")

## ex 3 - Cramer's Rule

def replace_column(matrix: list[list[float]], index: int, col: list[float]) -> None:
    for i in range(len(matrix)):
        matrix[i][index] = col[i]

def solve_cramer(matrix: list[list[float]], vector: list[float]) -> list[float]:
    Ax = copy.deepcopy(matrix)
    Ay = copy.deepcopy(matrix)
    Az = copy.deepcopy(matrix)

    replace_column(Ax,0,vector)
    replace_column(Ay, 1, vector)
    replace_column(Az, 2, vector)

    detA : float = determinant(matrix)

    if abs(detA) < 1e-10:
        raise ValueError(f"The determinant is too close to zero, det = {detA:.5e}")

    return [determinant(Ax)/detA, determinant(Ay)/detA, determinant(Az)/detA]

print(f"{solve_cramer(A, B)=}")

## ex 3 - Solving using Inversion

def minor(matrix: list[list[float]], i: int, j: int) -> list[list[float]]:
    res : list[list[float]] = []
    for ii in range(len(matrix)):
        row : list[float] = []
        for jj in range(len(matrix)):
            if ii != i and jj != j:
                row.append(matrix[ii][jj])
        if len(row) > 0:
            res.append(row)
    return res

def determinant2d(matrix: list[list[float]]) -> float:
    a, b = matrix[0]
    c, d = matrix[1]

    return a*d - b*c

def cofactor(matrix: list[list[float]]) -> list[list[float]]:
    coef : list[list[float]] = []
    for i in range(len(matrix)):
        row : list[float] = []
        for j in range(len(matrix)):
            if (i + j) % 2 == 0:
                row.append(determinant2d(minor(matrix,i,j)))
            else:
                row.append((-1)*determinant2d(minor(matrix,i,j)))
        coef.append(row)
    return coef

def adjoint(matrix: list[list[float]]) -> list[list[float]]:
    return cofactor(transpose(matrix))

def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    inv : list[list[float]] = adjoint(matrix)
    det : float = determinant(matrix)

    if abs(det) < 1e-10:
        raise ValueError(f"The determinant is too close to zero, det = {det:.5e}")

    det = 1.0 / det

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            inv[i][j] = det * inv[i][j]
    return multiply(inv,vector)

print(f"{solve(A, B)=}")