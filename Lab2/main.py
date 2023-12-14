import numpy as np

def gauss_elimination_with_pivoting(A, b):
    n = len(A)
    M = A.copy()
    P = np.eye(n)
    x = np.zeros(n)

    # Прямий хід
    for i in range(n):
        # Вибір головного елемента
        max_index = np.argmax(abs(M[i:, i])) + i
        if i != max_index:
            M[[i, max_index]] = M[[max_index, i]]
            P[[i, max_index]] = P[[max_index, i]]
            b[i], b[max_index] = b[max_index], b[i]

        # Вилучення елементів нижче діагоналі
        for j in range(i+1, n):
            ratio = M[j][i]/M[i][i]
            for k in range(i, n):
                M[j][k] -= ratio * M[i][k]
            b[j] -= ratio * b[i]

    # Зворотній хід
    for i in range(n-1, -1, -1):
        x[i] = b[i]
        for j in range(i+1, n):
            x[i] -= M[i][j] * x[j]
        x[i] /= M[i][i]

    return P, M, x

def jacobi_method(A, b, tolerance=1e-10, max_iterations=1000):
    n = len(A)
    x = np.zeros_like(b)
    D = np.diag(A)
    R = A - np.diagflat(D)

    for iteration in range(max_iterations):
        x_new = (b - np.dot(R, x)) / D
        diff = np.linalg.norm(x_new - x, np.inf)

        if iteration == 0 or iteration == max_iterations - 1:
            print(f"Iteration {iteration+1}, Norm of difference: {diff}")

        if diff < tolerance:
            return x_new
        x = x_new

    return x


def is_strictly_diagonally_dominant(matrix):
    n = matrix.shape[0]

    for i in range(n):
        diagonal_element = abs(matrix[i, i])
        non_diagonal_sum = np.sum(np.abs(matrix[i, :])) - diagonal_element

        if diagonal_element <= non_diagonal_sum:
            return False

    return True

# Оновлення матриці системи до 4x4 для перевірки збіжності методу Якобі
A_4x4 = np.array([[10, 1, 1, 1],
                  [1, 10, 1, 1],
                  [1, 1, 10, 1],
                  [1, 1, 1, 10]], dtype='float64')

# Оновлення вектору вільних членів
b_4x4 = np.array([14, 14, 14, 14], dtype='float64')

# Виконання методу Гауса з вибором головного по рядках
P_4x4, M_4x4, x_gauss_4x4 = gauss_elimination_with_pivoting(A_4x4, b_4x4.copy())
print("Метод Гауса з вибором головного елемента (4x4):")
print("Матриця перестановок P:\n", P_4x4)
print("Модифікована матриця M після прямого ходу:\n", M_4x4)
print("Розв'язок системи:\n", x_gauss_4x4)

# Виконання методу Якобі
x_jacobi_4x4 = jacobi_method(A_4x4, b_4x4)
print("\nМетод Якобі (4x4):")
print("Розв'язок системи:\n", x_jacobi_4x4)

result = is_strictly_diagonally_dominant(A_4x4)
print("Умова строгої діагональної переваги виконується:", result)
