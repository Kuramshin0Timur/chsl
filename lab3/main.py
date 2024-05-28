import numpy as np

# Функция и её производная
def f(x):
    return x**4 - 2*x**3 - 13*x**2 + 14*x - 3

def f_prime(x):
    return 4*x**3 - 6*x**2 - 26*x + 14

# Метод Ньютона
def newton_method(f, f_prime, x0, tol=1e-4, max_iter=1000):
    x = x0
    for i in range(max_iter):
        fx = f(x)
        fpx = f_prime(x)
        if abs(fx) < tol:
            return x, i + 1
        if fpx == 0:
            print("Производная равна нулю. Метод не применим.")
            return None, i + 1
        x = x - fx / fpx
    return x, max_iter

# Варьирование начального приближения
initial_guesses = np.linspace(-5, 5, 100)
roots = []
iterations = []

for x0 in initial_guesses:
    root, iters = newton_method(f, f_prime, x0)
    if root is not None:
        roots.append(root)
        iterations.append(iters)

# Уникальные корни
unique_roots = np.unique(np.round(roots, 4))

print(f"Уникальные корни: {unique_roots}")
print(f"Количество итераций для каждого начального приближения: {iterations}")

# График для визуализации
import matplotlib.pyplot as plt

plt.plot(initial_guesses, roots, 'o', label='Корни')
plt.xlabel('Начальное приближение')
plt.ylabel('Найденный корень')
plt.title('Влияние начального приближения на метод Ньютона')
plt.axhline(y=unique_roots[0], color='r', linestyle='--', label=f'Корень {unique_roots[0]}')
plt.axhline(y=unique_roots[1], color='g', linestyle='--', label=f'Корень {unique_roots[1]}')
plt.axhline(y=unique_roots[2], color='b', linestyle='--', label=f'Корень {unique_roots[2]}')
plt.axhline(y=unique_roots[3], color='y', linestyle='--', label=f'Корень {unique_roots[3]}')
plt.legend()
plt.grid(True)
plt.show()
