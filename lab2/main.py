import os
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Заданные матрица и вектор
A = np.array([[4, 1, 2],
              [3, 5, 1],
              [1, 1, 3]], dtype=float)
b = np.array([4, 7, 3], dtype=float)

# 1. Определение числа обусловленности матрицы
cond_A = np.linalg.cond(A)
print(f"Число обусловленности матрицы A: {cond_A}")

# 2. Решение СЛАУ методами итерации

def simple_iteration(A, b, x0, tol, max_iter=10000):
    n = len(b)
    D = np.diag(np.diag(A))
    L_U = A - D
    D_inv = np.linalg.inv(D)
    B = -D_inv @ L_U
    c = D_inv @ b
    x = x0
    for k in range(max_iter):
        x_new = B @ x + c
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k + 1
        x = x_new
    return x, max_iter

def gauss_seidel(A, b, x0, tol, max_iter=10000):
    n = len(b)
    x = x0
    for k in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k + 1
        x = x_new
    return x, max_iter

def jacobi(A, b, x0, tol, max_iter=10000):
    n = len(b)
    D = np.diag(np.diag(A))
    R = A - D
    x = x0
    for k in range(max_iter):
        x_new = np.dot(np.linalg.inv(D), b - np.dot(R, x))
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k + 1
        x = x_new
    return x, max_iter

# Проверка методов на сходимость
x0 = np.zeros_like(b)

tolerances = [1e-2, 1e-3, 1e-4]

results = {
    'Simple Iteration': [],
    'Gauss-Seidel': [],
    'Jacobi': []
}

for tol in tolerances:
    x_si, k_si = simple_iteration(A, b, x0, tol)
    x_gs, k_gs = gauss_seidel(A, b, x0, tol)
    x_j, k_j = jacobi(A, b, x0, tol)
    
    results['Simple Iteration'].append((tol, x_si, k_si))
    results['Gauss-Seidel'].append((tol, x_gs, k_gs))
    results['Jacobi'].append((tol, x_j, k_j))

for method, result in results.items():
    print(f"\nМетод: {method}")
    for tol, x, k in result:
        print(f"Точность: {tol}, Решение: {x}, Количество итераций: {k}")

# Влияние начального приближения на количество итераций
exact_solution = np.linalg.solve(A, b)
initial_approximations = [np.zeros_like(b), np.ones_like(b), np.full_like(b, 5)]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
methods = [simple_iteration, gauss_seidel, jacobi]
method_names = ['Simple Iteration', 'Gauss-Seidel', 'Jacobi']

for i, tol in enumerate(tolerances):
    for method, method_name in zip(methods, method_names):
        iterations = []
        for x0 in initial_approximations:
            _, k = method(A, b, x0, tol)
            iterations.append(k)
        axes[i].plot(np.linalg.norm(initial_approximations - exact_solution, axis=1), iterations, label=method_name)
    axes[i].set_title(f"Точность {tol}")
    axes[i].set_xlabel("Норма разности начального приближения и точного решения")
    axes[i].set_ylabel("Количество итераций")
    axes[i].legend()

plt.tight_layout()
plt.savefig("iterations_vs_initial_approximation.png")
