import numpy as np
import matplotlib.pyplot as plt

# Функція для визначення непереревності функції
def continuity(f, a, b, epsilon=1e-4):
    if not callable(f):
        raise ValueError("The provided 'f' is not a callable function.")

    def limit(f, c, delta_x=1e-6):
        left_limit = f(c - delta_x)
        right_limit = f(c + delta_x)
        return left_limit, right_limit


    if a <= b and (a is None or b is None):
        return False


    left_limit_a, right_limit_a = limit(f, a)
    left_limit_b, right_limit_b = limit(f, b)


    if abs(left_limit_a - f(a)) > epsilon or abs(right_limit_a - f(a)) > epsilon or abs(
            left_limit_b - f(b)) > epsilon or abs(right_limit_b - f(b)) > epsilon:
        return False

    return True


# Функція для визначення знакосталості функції
def check_sign_changes(f, a, b, num_points=10000) -> bool:
    x_values = np.linspace(a, b, num_points)
    signs = [np.sign(f(x)) for x in x_values]

    sign_changes = False
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1]:
            sign_changes = True
            break

    return sign_changes


# Функція для визначення мінімуму та максимуму функції на інтервалі
def find_absolute_extrema(f, a, b, num_points=10000):
    x_values = np.linspace(a, b, num_points)
    y_values = [abs(f(x)) for x in x_values]

    absolute_max = max(y_values)
    absolute_min = min(y_values)

    return absolute_max, absolute_min


# Визначимо нашу функцію
def f(x):
    return x ** 2 + np.sin(x) - 12 * x - 0.25


# Визначимо похідну функції
def df(x):
    return 2 * x + np.cos(x) - 12


# Визначимо похідну другого порядку функції
def d2f(x):
    return 2 - np.sin(x)


# Обчислення значень функції на проміжку [-10, 10]
x = np.linspace(-10, 10, 1000)
y = x**2 + np.sin(x) - 12*x - 0.25

# Побудова графіку функції
plt.plot(x, y)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Графік функції f(x) = x^2 + 5sin(x) - 1')
plt.grid(True)
plt.show()


def has_root(f, a, b) -> bool:
    if not callable(f):
        raise ValueError("f must be a callable function.")

    if a >= b:
        raise ValueError("The interval must satisfy 'a < b'.")

    fa = f(a)
    fb = f(b)

    if fa * fb < 0:
        return True
    else:
        return False


# Перевірка зміни знаків функції на кінцях проміжку [-2, 1]
a = -2
b = 0
if has_root(f, a, b):
    print("Зміна знаків функції на кінцях проміжку, корінь існує.")
else:
    print("Функція не має кореня на даному проміжку.")


def dichotomy_method(a, b, eps):
    iteration_count = 0  # Лічильник ітерацій
    if f(a) * f(b) >= 0:
        print("Функція не змінює знак на інтервалі. Виберіть інший інтервал.")
        return None, iteration_count

    while (b - a) / 2 > eps:
        iteration_count += 1  # Збільшуємо лічильник на кожній ітерації
        midpoint = (a + b) / 2
        if f(a) * f(midpoint) < 0:  # Корінь знаходиться в лівій половині
            b = midpoint
        else:  # Корінь знаходиться в правій половині
            a = midpoint

    return (a + b) / 2, iteration_count

# Знаходження кореня з точністю до ε = 10^-4
epsilon = 1e-4
root, iterations = dichotomy_method(a, b, epsilon)
print("Корінь функції:", root)
print("Кількість ітерацій:", iterations)


def modified_newton_method(f, df, ddf, a, b, epsilon):
    x0 = (a + b) / 2  # Початкове наближення
    x = x0

    continuity_df = continuity(df, a, b)
    print("Неперервна похідна:", continuity_df)
    continuity_ddf = continuity(ddf, a, b)
    print("Неперервна друга похідна:", continuity_ddf)
    sign_changes_df = check_sign_changes(df, a, b)
    print("Зміна знаку похідної:", sign_changes_df)
    sign_changes_ddf = check_sign_changes(ddf, a, b)
    print("Зміна знаку другої похідної:", sign_changes_ddf)
    fx0 = f(x0)
    print("f(x0) =", fx0)
    ddfx0 = ddf(x0)
    print("ddf(x0) =", ddfx0)
    has_root_df = has_root(df, a, b)
    print("Наявність кореня в похідної на проміжку [a;b]:", has_root_df)

    # Перевірка достатньої умови збіжності
    if continuity_ddf and not sign_changes_df and not sign_changes_ddf and not has_root_df:
        print("Достатня умова збіжності виконується.")
    else:
        print("Достатня умова збіжності не виконується.")

    iterations = 0  # Лічильник ітерацій

    while True:
        x_next = x - f(x) / df(x0)
        iterations += 1

        print("x_next =", x_next)

        if abs(x_next - x) < epsilon:
            return x_next, iterations

        x = x_next


root_modified_newton, iterations_modified_newton = modified_newton_method(f, df, d2f, a, b, 1e-4)
print("Корінь модифікованим методом Ньютона:", root_modified_newton)
print("Кількість ітерацій:", iterations_modified_newton)

def simple_iteration_method(phi, dphi, a, b, epsilon):
    x0 = (a + b) / 2  # Початкове наближення
    print("x0 =", x0)
    x = x0

    # Перевiримо достатнi умови збiжностi
    q, _ = find_absolute_extrema(dphi, a, b)
    print("q =", q)
    condition_2_left = abs(phi(x0) - x0)
    print("condition_2_left =", condition_2_left)
    delta = max(abs(a - x0), abs(b - x0))
    print("delta =", delta)
    condition_2_right = (1 - q) * delta
    print("condition_2_right =", condition_2_right)

    # Перевірка достатньої умови збіжності
    if q >= 1 or condition_2_left >= condition_2_right:
        print("Достатня умова збіжності не виконується.")
    else:
        print("Достатня умова збіжності виконується.")

    iterations = 0  # Лічильник ітерацій

    while True:
        if iterations > 100000:
            print("Перевищено максимальну кількість ітерацій.")
            return x, iterations

        x_next = phi(x)
        iterations += 1

        if (iterations < 20):
            print("x_next =", x_next)

        if abs(x_next - x) < epsilon:
            return x_next, iterations

        x = x_next


# Виразимо х з функції f(x) = 0
def phi(x):
    x_safe = np.where(x == 0, np.finfo(float).eps, x)
    return (np.sin(x_safe) / x_safe ** 2) + ((12 - np.cos(x_safe)) / x_safe) - 12 / x_safe - (1 / (4 * x_safe ** 2))


# Визначимо похідну функції
def dphi(x):
    return (np.sin(x)/x**2)+((12-np.cos(x))/x)-12/x-(1/(4 * x ** 2))


# Виклик методу простої ітерації
root_simple_iteration, iterations_simple_iteration = simple_iteration_method(phi, dphi, a, b, 1e-4)
print("Корінь методом простої ітерації:", root_simple_iteration)
print("Кількість ітерацій:", iterations_simple_iteration)

def phi2(x):
    under_sqrt = 0.25 - np.sin(x) + 12 * x
    safe_sqrt = np.where(under_sqrt < 0, np.nan, under_sqrt)  # використовуємо np.nan для від'ємних значень
    return (12 - np.cos(x)) / (2 * np.sqrt(safe_sqrt))

# Визначимо похідну функції
def dphi2(x):
    return (12 - np.cos(x))/(2 * np.sqrt(0.25 - np.sin(x) + 12*x))

root_simple_iteration, iterations_simple_iteration = simple_iteration_method(phi2, dphi2, a, b, 1e-4)
print("Корінь методом простої ітерації:", root_simple_iteration)
print("Кількість ітерацій:", iterations_simple_iteration)


