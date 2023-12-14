import numpy as np

def f(x):
    """Визначаємо функцію для інтегрування."""
    return 1 / (1 + x)

def simpson_integration(f, a, b, n):
    """Виконуємо інтегрування за методом Сімпсона."""
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    S = h / 3 * np.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2])
    return S

def approximate_integral(a, b, tolerance=0.001):
    """Наближено обчислюємо інтеграл з заданою точністю за методом Сімпсона."""
    n = 2
    S_n = simpson_integration(f, a, b, n)

    while True:
        n *= 2
        S_2n = simpson_integration(f, a, b, n)
        error = abs(S_2n - S_n) / 15
        if error < tolerance:
            return S_2n, error, n
        S_n = S_2n

# Межі інтегрування
a, b = -11, -5

# Наближене обчислення інтегралу з заданою точністю
result, error, n_intervals = approximate_integral(a, b)
print(f"Наближене значення інтегралу: {result}, Оцінка похибки: {error}, Кількість інтервалів: {n_intervals}")
