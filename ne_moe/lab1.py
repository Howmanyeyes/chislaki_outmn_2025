import numpy as np
from numpy.linalg import norm, solve
import matplotlib.pyplot as plt

l = 1.0
N = 200
num_modes = 5
x = np.linspace(-l, l, N, endpoint=False)
h = x[1] - x[0]
noise = 0.4

L = np.zeros((N, N))
for j in range(N):
    L[j, j] = 2.0
    L[j, (j-1) % N] = -1.0
    L[j, (j+1) % N] = -1.0
L /= h**2

def rayleigh_iteration(L, v0, lam_initial, tol=1e-10, max_iter=1000):
    v = v0 / norm(v0)
    lam = lam_initial

    for _ in range(max_iter):
        v_new = solve(L - lam * np.eye(L.shape[0]), v)
        v_new /= norm(v_new)

        lam_new = (v_new @ (L @ v_new)) / (v_new @ v_new)

        if norm(v_new - v) < tol or norm(v_new + v) < tol:
            return lam_new, v_new

        v = v_new
        lam = lam_new

    return lam, v

def symmetrize(vec):
    vec_rev = vec[::-1]
    even = 0.5*(vec + vec_rev)
    odd  = 0.5*(vec - vec_rev)
    return even/norm(even), odd/norm(odd)

exact_lambdas = [(n*np.pi/l)**2 for n in range(num_modes)]
eigvals = []
eigvecs = []

for n in range(num_modes):
    v0 = np.random.rand(N)
    lam, vec = rayleigh_iteration(L, v0, lam_initial=exact_lambdas[n])
    vec /= norm(vec)
    eigvals.append(lam)
    eigvecs.append(vec)

eig_decomposed = []
for k in range(num_modes):
    if k == 0:
        eig_decomposed.append((eigvecs[k], None))
    else:
        eig_decomposed.append(symmetrize(eigvecs[k]))

for n in range(num_modes):
    if n == 0:
        analytical = np.ones_like(x)
        analytical /= norm(analytical)
        eig_vec = eig_decomposed[n][0] * np.sign(np.dot(eig_decomposed[n][0], analytical))
        error_pct = norm(eig_vec - analytical) / norm(analytical) * 100
        print(f"Собственное значение {n + 1}: {eigvals[n]:.6f}")
        print(f"функция: {eig_vec[:5]} ...")
        print(f"ошибка = {error_pct:.2f}%\n")
        plt.figure(figsize=(8,4))
        plt.plot(x, analytical, 'r--', label='Аналитическая')
        plt.plot(x, eig_vec, 'b', label='Численная')
        plt.title(f"Собственное значение {n + 1}: {eigvals[n]:.4f}")
        plt.xlabel("x")
        plt.ylabel("X(x)")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        analytical_even = np.cos(n * np.pi * x / l)
        analytical_even /= norm(analytical_even)
        num_even = eig_decomposed[n][0] * np.sign(np.dot(eig_decomposed[n][0], analytical_even))
        analytical_odd = np.sin(n * np.pi * x / l)
        analytical_odd /= norm(analytical_odd)
        num_odd = eig_decomposed[n][1] * np.sign(np.dot(eig_decomposed[n][1], analytical_odd))
        error_even_pct = norm(num_even - analytical_even) / norm(analytical_even) * 100
        error_odd_pct  = norm(num_odd  - analytical_odd)  / norm(analytical_odd)  * 100
        print(f"Собственное значение {n + 1}: {eigvals[n]:.6f}")
        print(f"косинусная часть: {num_even[:5]} ...")
        print(f"синусная часть:   {num_odd[:5]} ...")
        print(f"ошибка cos = {error_even_pct:.2f}%, ошибка sin = {error_odd_pct:.2f}%\n")
        plt.figure(figsize=(8,4))
        plt.plot(x, analytical_even, 'r--', label='Аналитическая cos')
        plt.plot(x, num_even, 'b', label='Численная cos')
        plt.plot(x, analytical_odd, 'g--', label='Аналитическая sin')
        plt.plot(x, num_odd, 'm', label='Численная sin')
        plt.title(f"Собственное значение {n + 1}: {eigvals[n]:.4f}")
        plt.xlabel("x")
        plt.ylabel("X(x)")
        plt.legend()
        plt.grid(True)
        plt.show()


# mode_index = 3
# N_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]
# errors = []
# h_list = []
#
# for N_test in N_list:
#     x_test = np.linspace(-l, l, N_test, endpoint=False)
#     h = x_test[1] - x_test[0]
#     h_list.append(h)
#
#     L_test = np.zeros((N_test, N_test))
#     for j in range(N_test):
#         L_test[j, j] = 2.0
#         L_test[j, (j - 1) % N_test] = -1.0
#         L_test[j, (j + 1) % N_test] = -1.0
#     L_test /= h ** 2
#     v0 = np.random.rand(N_test)
#
#     lam_num, _ = rayleigh_iteration(L_test, v0, lam_initial=exact_lambdas[mode_index])
#     error = abs(lam_num - exact_lambdas[mode_index])
#     errors.append(error)
#
# print("N\t h\t\t\tОшибка")
# for i in range(len(N_list)):
#     print(f"{N_list[i]}\t {h_list[i]:.5f}\t {errors[i]:.5e}")
#
# print("\nПорядок аппроксимации между соседними шагами:")
# for i in range(len(errors) - 1):
#     p = np.log(errors[i] / errors[i + 1]) / np.log(h_list[i] / h_list[i + 1])
#     print(f"N={N_list[i]} -> N={N_list[i + 1]}: порядок = {p:.2f}")
#
# plt.figure(figsize=(8,5))
# plt.loglog(h_list, errors, 'o-', label='Ошибка')
# plt.xlabel('Шаг сетки h')
# plt.ylabel('Ошибка')
# plt.title('Порядок аппроксимации')
# plt.grid(True, which="both", ls="--")
# plt.legend()
# plt.show()

# print(exact_lambdas)
# print(eigvecs[0])
# print(noise_lambdas)
mode_index = 5
N_list = [40, 50, 60]
errors = []
h_list = []

for N_test in N_list:
    x_test = np.linspace(-l, l, N_test, endpoint=False)
    h = x_test[1] - x_test[0]
    h_list.append(h)

    L_test = np.zeros((N_test, N_test))
    for j in range(N_test):
        L_test[j, j] = 2.0
        L_test[j, (j - 1) % N_test] = -1.0
        L_test[j, (j + 1) % N_test] = -1.0
    L_test /= h ** 2

    v0 = np.random.rand(N_test)
    lam_num, _ = rayleigh_iteration(L_test, v0, lam_initial=exact_lambdas[mode_index - 1])
    error = abs(lam_num - exact_lambdas[mode_index - 1])
    errors.append(error)

print("N\t h\t\t\tОшибка")
for i in range(len(N_list)):
    print(f"{N_list[i]}\t {h_list[i]:.5f}\t {errors[i]:.5e}")


log_h = np.log(h_list)
log_errors = np.log(errors)

p, logC = np.polyfit(log_h, log_errors, 1)
print(f"\nПорядок аппроксимации = {p:.2f}")

plt.figure(figsize=(8,5))
plt.loglog(h_list, errors, 'o-', label='Ошибка')
h_fit = np.linspace(min(h_list), max(h_list), 100)
error_fit = np.exp(logC) * h_fit**p
plt.loglog(h_fit, error_fit, 'r--', label=f'порядок, p = {p:.2f}')

plt.xlabel('Шаг сетки h')
plt.ylabel('Ошибка')
plt.title('Порядок аппроксимации')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()
