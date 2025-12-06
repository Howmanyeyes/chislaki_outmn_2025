import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

x_L, x_R = 0, 2
y_B, y_T = 0, 2
lx, ly = x_R - x_L, y_T - y_B
h0 = 0.1
nx, ny = int(np.round(lx / h0)), int(np.round(ly / h0))
hx, hy = lx / nx, ly / ny

koeff = 1.0 # коэффициент теплопроводности
c_rho = 1.0 # c * rho
Q = 1 # мощность пола
q_w = 10 # поток через форточку
l = 0.5 # длина форточки
x1, x2 = x_L + lx / 2 - l / 2, x_L + lx / 2 + l / 2  # координаты форточки по x
T_min, T_max = 20.0, 30.0 # температуры

tau = 0.1 # шаг по времени
nu = 0.5 # Кранк–Николсон
t_max = 200.0 # время анимации

x = np.linspace(x_L + 0.5 * hx, x_R - hx * 0.5, nx)
y = np.linspace(y_B + 0.5 * hy, y_T - hy * 0.5, ny)
X, Y = np.meshgrid(x, y, indexing='ij')

j_top = ny - 1
i_window = np.where((x >= x1) & (x <= x2))[0]
u_mat = T_min * np.ones((nx, ny))
u = u_mat.flatten()
sigma = 0

def RHS(T_2d, nx, ny, Q, c_rho, q_w, i_window, j_top, T_min, T_max, sigma_prev, hy):
    T_avg = np.mean(T_2d)
    sigma_new = sigma_prev
    if T_avg >= T_max:
        sigma_new = 1
    elif T_avg <= T_min:
        sigma_new = 0

    F = np.full(nx * ny, Q / c_rho)
    if sigma_new == 1:
        idx = i_window * ny + j_top
        F[idx] -= q_w / (c_rho * hy)
    return F, sigma_new

def getL_csr(n_x, n_y, h_x, h_y, k=1.0):
    idx = np.arange(n_x * n_y, dtype=int).reshape((n_x, n_y))

    iBB = idx[:, :-1].flatten(); iB = idx[:, 1:].flatten()
    iTT = idx[:, 1:].flatten();  iT = idx[:, :-1].flatten()

    iHL = idx[1:, :].flatten(); iL = idx[:-1, :].flatten()
    iHR = idx[:-1, :].flatten(); iR = idx[1:, :].flatten()

    R1, C1, V1 = iBB, iBB, np.full(iBB.size, +k / h_y**2)
    R2, C2, V2 = iBB, iB,  np.full(iBB.size, -k / h_y**2)
    R3, C3, V3 = iTT, iTT, np.full(iTT.size, +k / h_y**2)
    R4, C4, V4 = iTT, iT,  np.full(iTT.size, -k / h_y**2)
    R5, C5, V5 = iHL, iHL, np.full(iHL.size, +k / h_x**2)
    R6, C6, V6 = iHL, iL,  np.full(iHL.size, -k / h_x**2)
    R7a, C7a, V7a = iHR, iHR, np.full(iHR.size, +k / h_x**2)
    R7b, C7b, V7b = iHR, iR,  np.full(iHR.size, -k / h_x**2)
    R7 = np.concatenate([R7a, R7b])
    C7 = np.concatenate([C7a, C7b])
    V7 = np.concatenate([V7a, V7b])

    row = np.concatenate([R1, R2, R3, R4, R5, R6, R7])
    col = np.concatenate([C1, C2, C3, C4, C5, C6, C7])
    val = np.concatenate([V1, V2, V3, V4, V5, V6, V7])

    return sp.csr_matrix((val, (row, col)), shape=(n_x*n_y, n_x*n_y))

L = getL_csr(nx, ny, hx, hy, koeff)
N = nx * ny
P = sp.eye(N) * c_rho
A = (P + tau * nu * L).tocsc()
B = (P - tau * (1 - nu) * L).tocsr()
solve_A = spla.factorized(A)

nsteps = int(np.ceil(t_max / tau))
frames, times = [], []

frames.append(u_mat.copy())
times.append(0.0)

for n in range(nsteps):
    t_np1 = (n + 1) * tau

    _, sigma_pred = RHS(u_mat, nx, ny, Q, c_rho, q_w, i_window, j_top, T_min, T_max, sigma, hy)

    F_n, _ = RHS(u_mat, nx, ny, Q, c_rho, q_w, i_window, j_top, T_min, T_max, sigma, hy)
    F_np1, sigma_new = RHS(u_mat, nx, ny, Q, c_rho, q_w, i_window, j_top, T_min, T_max, sigma_pred, hy)

    F_mix = (1 - nu) * F_n + nu * F_np1
    rhs = B.dot(u) + tau * F_mix

    u = solve_A(rhs)
    u_mat = u.reshape((nx, ny))
    sigma = sigma_new
    if n % 200 == 0:
        T_avg = np.mean(u_mat)
        print(f"t = {t_np1:6.1f}, T_avg = {T_avg:6.2f}, sigma = {sigma}")

    if (n + 1) % max(1, nsteps // 200) == 0 or n == nsteps - 1:
        frames.append(u_mat.copy())
        times.append(t_np1)

fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(frames[0].T, origin='lower', extent=(x_L, x_R, y_B, y_T),
               cmap='hot', vmin=T_min, vmax=T_max)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Температура')
ax.set_xlabel('x, m')
ax.set_ylabel('y, m')
ax.set_title("t = 0.0")

def update(frame):
    im.set_array(frames[frame].T)
    im.set_clim(vmin=T_min, vmax=T_max)
    T_avg = np.mean(frames[frame])
    ax.set_title(f"t = {times[frame]:.1f} s, T_avg = {T_avg:.1f}")
    return im,

ani = FuncAnimation(fig, update, frames=len(frames), interval=700, blit=False)
plt.show()