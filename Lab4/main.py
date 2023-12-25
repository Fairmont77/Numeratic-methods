import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

def f(x):
    return x ** 3 - 3 * x ** 2 - 14 * x - 8 + np.sin(x)

def divided_differences(x_values, y_values):
    n = len(x_values)
    coef = np.zeros([n, n])
    coef[:, 0] = y_values
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x_values[i + j] - x_values[i])
    return coef[0, :]

def newton_interpolation(x_values, y_values, coefs, x):
    n = len(x_values) - 1
    result = coefs[n]
    for i in range(n - 1, -1, -1):
        result = result * (x - x_values[i]) + coefs[i]
    return result

# Визуалізація функції для визначення інтервалу пошуку кореня
x_plot = np.linspace(-6, 6, 400)
y_plot = f(x_plot)
plt.figure(figsize=(12, 6))
plt.plot(x_plot, y_plot, label='Original Function $f(x)$', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

# Обчислення інтерполяційного полінома та знаходження кореня
x_values = np.linspace(-6, 6, 11)
y_values = f(x_values)
coefs = divided_differences(x_values, y_values)

root = brentq(lambda x: newton_interpolation(x_values, y_values, coefs, x), -6, 6)

# Основний графік
y_newton = newton_interpolation(x_values, y_values, coefs, x_plot)
plt.figure(figsize=(12, 6))
plt.plot(x_plot, y_plot, label='Original Function $f(x)$', color='blue')
plt.plot(x_plot, y_newton, label='Newton Interpolation', color='red', linestyle='--')
plt.title('Original Function and Newton Interpolation')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Збільшений графік біля кореня
zoom_range = 0.5  # Збільшити область навколо кореня
x_zoom = np.linspace(root - zoom_range, root + zoom_range, 200)
y_zoom = f(x_zoom)
y_newton_zoom = newton_interpolation(x_values, y_values, coefs, x_zoom)
plt.figure(figsize=(12, 6))
plt.plot(x_zoom, y_zoom, label='Original Function $f(x)$', color='blue')
plt.plot(x_zoom, y_newton_zoom, label='Newton Interpolation', color='red', linestyle='--')
plt.scatter(root, newton_interpolation(x_values, y_values, coefs, root), color='green', marker='o')
plt.title(f'Zoomed In View Near the Root at x = {root:.4f}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()


a, b = 5, 6

# Обчислення інтерполяційного полінома та знаходження кореня
x_values = np.linspace(a, b, 11)
y_values = f(x_values)
coefs = divided_differences(x_values, y_values)
try:
    root = brentq(lambda x: newton_interpolation(x_values, y_values, coefs, x), a, b)
    print(f"Знайдений корінь: {root}")
except ValueError as e:
    print("Не вдалося знайти корінь у заданому інтервалі:", e)
