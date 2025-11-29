import numpy as np
from numpy.linalg import norm, solve
import matplotlib.pyplot as plt
from scipy.special import lpmv


# Параметры
a = 1.0
m = 5
U0 = m * (m + 1) * a**2
l = 5.0 / a       
N = 1000          
num_modes = 5     

x = np.linspace(-l, l, N)
h = x[1] - x[0]


# Построение матрицы для -d2/dx2
main_diag = 2 * np.ones(N)
off_diag = -1 * np.ones(N - 1)
L = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
L /= h**2

# Потенциал Пёшль–Теллера
U = U0 * np.tanh(a * x)**2
H = L + np.diag(U)


# Жёсткое наложение граничных условий ψ(-l)=ψ(l)=0
H = H[1:-1, 1:-1]
x_inner = x[1:-1]


# Метод Релея
def rayleigh_iteration(A, v0, lam_initial, tol=1e-10, max_iter=1000):
    v = v0 / norm(v0)
    lam = lam_initial
    for _ in range(max_iter):
        v_new = solve(A - lam * np.eye(A.shape[0]), v)
        v_new /= norm(v_new)
        lam_new = (v_new @ (A @ v_new)) / (v_new @ v_new)
        if norm(v_new - v) < tol or norm(v_new + v) < tol:
            return lam_new, v_new
        v = v_new
        lam = lam_new
    return lam, v


# Поиск собственных состояний
analytical_E = [U0 - (a * (m - k))**2 for k in range(num_modes)]
print("Аналитические уровни энергии:", analytical_E)

eigvals = []
eigvecs = []

for n in range(num_modes):
    v0 = np.random.rand(N - 2)
    lam, vec = rayleigh_iteration(H, v0, lam_initial=analytical_E[n])
    eigvals.append(lam)
    eigvecs.append(vec / norm(vec))

# Сравнение с аналитическим решением
for k in range(num_modes):
    psi_num = eigvecs[k]
    psi_num /= np.max(np.abs(psi_num))
    
    # порядок (m, m - k)
    psi_analytical = lpmv(m - k, m, np.tanh(a * x_inner))
    psi_analytical /= np.max(np.abs(psi_analytical))
    
    # Синхронизация фазы
    sign = np.sign(np.dot(psi_num, psi_analytical))
    psi_num *= sign

    plt.figure(figsize=(8, 4))
    plt.plot(x_inner, psi_analytical, 'r--', label='Аналитическая ψ')
    plt.plot(x_inner, psi_num, 'b', label='Численная ψ')
    plt.title(f"Уровень E{k+1}: E_num={eigvals[k]:.4f},  E_exact={analytical_E[k]:.4f}")
    plt.xlabel("x")
    plt.ylabel("ψ(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

    err = norm(psi_num - psi_analytical) / norm(psi_analytical)
    print(f"Собственное значение {k+1}: E_num={eigvals[k]:.6f}, E_exact={analytical_E[k]:.6f}")
    print(f"Относительная ошибка волновой функции: {err*100:.2f}%\n")
