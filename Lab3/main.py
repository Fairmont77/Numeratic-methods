import math
import numpy as np

# Визначення системи нелінійних рівнянь
def system_of_equations(vars):
    x, y = vars
    return [math.sin(x - y) - x*y + 1, x**2 - y**2 - 0.75]

# Визначення матриці Якобі для системи
def jacobian(vars):
    x, y = vars
    return [[math.cos(x - y) - y, -math.cos(x - y) - x], [2*x, -2*y]]

# Функція для обертання матриці
def invert_matrix(matrix):
    det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    if det == 0:
        raise ValueError("Матриця є сингулярною і не може бути обернена.")
    inv_det = 1 / det
    return [[matrix[1][1] * inv_det, -matrix[0][1] * inv_det],
            [-matrix[1][0] * inv_det, matrix[0][0] * inv_det]]

# Модифікований метод Ньютона для розв'язання системи нелінійних рівнянь
def modified_newton_method(x0, y0, max_iter=10000, tol=1e-10):
    x, y = x0, y0
    iterations = []
    for iteration in range(max_iter):
        J = jacobian([x, y])
        J_inv = invert_matrix(J)
        F_val = system_of_equations([x, y])
        delta = np.dot(J_inv, F_val)
        x_new = x - delta[0]
        y_new = y - delta[1]
        iterations.append((x_new, y_new))
        if abs(x_new - x) < tol and abs(y_new - y) < tol:
            return x_new, y_new, iterations
        x, y = x_new, y_new
    raise ValueError("Розв'язок не було знайдено протягом максимальної кількості ітерацій")

# Визначення ітераційних функцій для методу простої ітерації
def phi_x(y):
    return math.sqrt(0.75 + y**2)

def phi_y(x):
    return (1 + math.sin(x)) / x if x != 0 else 0

# Метод простої ітерації для розв'язання системи нелінійних рівнянь
def simple_iteration_method(x0, y0, max_iter=10000, tol=1e-10):
    x, y = x0, y0
    iterations = []
    for iteration in range(max_iter):
        x_new = phi_x(y)
        y_new = phi_y(x)
        iterations.append((x_new, y_new))
        if abs(x_new - x) < tol and abs(y_new - y) < tol:
            return x_new, y_new, iterations
        x, y = x_new, y_new
    raise ValueError("Розв'язок не було знайдено протягом максимальної кількості ітерацій")

# Початкові наближення
x0, y0 = 1.0, 1.0

# Розв'язання методом модифікованого Ньютона
try:
    solution_modified_newton, _, iterations_newton = modified_newton_method(x0, y0)
    print("Метод модифікованого Ньютона")
    print("Розв'язок:", solution_modified_newton)
    print("Кількість ітерацій:", len(iterations_newton))
    for i, (x, y) in enumerate(iterations_newton, 1):
        print(f"Ітерація {i}: x = {x}, y = {y}")
except ValueError as e:
    print(str(e))

# Розв'язання методом простої ітерації
try:
    solution_simple_iteration, _, iterations_simple = simple_iteration_method(x0, y0)
    print("\nМетод простої ітерації")
    print("Розв'язок:", solution_simple_iteration)
    print("Кількість ітерацій:", len(iterations_simple))
    for i, (x, y) in enumerate(iterations_simple, 1):
        print(f"Ітерація {i}: x = {x}, y = {y}")
except ValueError as e:
    print(str(e))
