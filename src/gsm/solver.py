import numpy as np
import math
from utils import sound_speed, enthalpy, interp_y
from geometry import top_wall, L, h0

def analytical_point_source(r,r_star = 1.0, p_star = 101325.0,
rho_star = 1.0, kappa = 1.4,tol = 1e-10, max_iter = 100):
    """
    Аналитическое решение для точечного источника:
    возвращает (w, M, p, rho) при удалении r ≥ r_star.
    """
    if r < r_star:
        raise ValueError("r < r_star")

    a_star = math.sqrt(kappa * p_star / rho_star)
    def F(w):
        A = max((kappa+1)/2 - (kappa-1)/2*(w*w/a_star**2), 1e-12)
        return 1.0/(A**(1/(kappa-1))*(w/a_star)) - r/r_star

    w_lo = a_star*1.0001
    w_hi = a_star*math.sqrt((kappa+1)/(kappa-1))*0.9999
    f_lo, f_hi = F(w_lo), F(w_hi)
    if f_lo*f_hi>0:
        raise RuntimeError("Корень не лежит в физическом диапазоне")

    # бисекция
    for _ in range(max_iter):
        w_mid = 0.5*(w_lo + w_hi)
        f_mid = F(w_mid)
        if abs(f_mid)<tol:
            w = w_mid
            break
        if f_lo*f_mid<0:
            w_hi = w_mid
        else:
            w_lo, f_lo = w_mid, f_mid
    else:
        raise RuntimeError("Bisection failed")

    A = (kappa+1)/2 - (kappa-1)/2*(w*w/a_star**2)
    a = a_star*math.sqrt(A)
    M = w / a
    rho = rho_star * A**(1/(kappa-1))
    p   = rho * a*a / kappa

    return w, M, p, rho

def compute_F(U, uk, vk, pk, rk, yg, dx, y3, gamma=1.4):
    u3,v3,p3,r3 = U
    p3,r3 = max(p3,1e-8), max(r3,1e-8)
    theta = math.atan2(v3,u3)
    w3 = math.hypot(u3,v3)+1e-8
    a3 = sound_speed(p3,r3,gamma)
    mu = math.asin(max(min(a3/w3,1),-1))

    lam_p = math.tan(theta+mu)
    lam_m = math.tan(theta-mu)
    lam_0 = v3/(u3+1e-8)

    y1 = y3 - lam_p*dx
    y2 = y3 - lam_m*dx
    y4 = y3 - lam_0*dx

    u1 = interp_y(uk, yg, y1)
    p1 = interp_y(pk, yg, y1)
    r1 = interp_y(rk, yg, y1)
    v1 = interp_y(vk, yg, y1)

    u2 = interp_y(uk, yg, y2)
    p2 = interp_y(pk, yg, y2)
    r2 = interp_y(rk, yg, y2)
    v2 = interp_y(vk, yg, y2)

    u4 = interp_y(uk, yg, y4)
    p4 = interp_y(pk, yg, y4)
    r4 = interp_y(rk, yg, y4)
    v4 = interp_y(vk, yg, y4)

    a4 = sound_speed(p4, r4, gamma)

    # F3, F4
    F3 = p3 - p4 - a4*a4*(r3-r4)
    H3 = enthalpy(p3,r3,gamma) + 0.5*w3*w3
    w4 = math.hypot(u4,v4)
    H4 = enthalpy(p4,r4,gamma) + 0.5*w4*w4
    F4 = H3 - H4

    # F1, F2
    a1,w1 = sound_speed(p1,r1,gamma), math.hypot(u1,v1)
    a2,w2 = sound_speed(p2,r2,gamma), math.hypot(u2,v2)

    mu1 = math.asin(max(min(a1/w1,1),-1))
    mu2 = math.asin(max(min(a2/w2,1),-1))
    cot1 = math.cos(mu1)/math.sin(mu1) if abs(mu1)>1e-3 else 0
    cot2 = math.cos(mu2)/math.sin(mu2) if abs(mu2)>1e-3 else 0

    rho_avg1 = 0.5*(r1+r3); w_avg1 = 0.5*(w1+w3)
    rho_avg2 = 0.5*(r2+r3); w_avg2 = 0.5*(w2+w3)

    F1 = theta - math.atan2(v1,u1) + (p3-p1)/(rho_avg1*w_avg1*w_avg1)*cot1
    F2 = theta - math.atan2(v2,u2) - (p3-p2)/(rho_avg2*w_avg2*w_avg2)*cot2

    return np.array([F1, F2, F3, F4])

def run_schm(Ny, Nx=200, x_target=2.0):

    x = np.linspace(0, L, Nx+1)
    y = np.linspace(0, h0, Ny)

    u = np.zeros((Nx+1, Ny))
    v = np.zeros((Nx+1, Ny))
    p = np.zeros((Nx+1, Ny))
    rho = np.zeros((Nx+1, Ny))

    for j in range(Ny):
        xi, yi = x[0] + 1.1, y[j]
        r = math.hypot(xi, yi)
        w_j, M_j, p_j, rho_j = analytical_point_source(r)
        theta = math.atan2(yi, xi)
        u[0, j] = w_j * math.cos(theta)
        v[0, j] = w_j * math.sin(theta)
        p[0, j] = p_j
        rho[0, j] = rho_j

    # шаг по x
    for k in range(Nx):
        dx = x[k+1] - x[k]
        yg = np.linspace(0, top_wall(x[k]), Ny)
        for j in range(Ny):
            U = np.array([u[k,j], v[k,j], p[k,j], rho[k,j]])
            uk, vk, pk, rk = u[k,:], v[k,:], p[k,:], rho[k,:]
            for _ in range(30):
                F = compute_F(U, uk, vk, pk, rk, yg, dx, yg[j])
                if np.linalg.norm(F, np.inf) < 1e-8:
                    break
                J = np.zeros((4,4))
                for i in range(4):
                    dU = np.zeros(4); dU[i] = 1e-8
                    J[:,i] = (compute_F(U+dU, uk, vk, pk, rk, yg, dx, yg[j]) - F)/1e-8
                try:
                    dU = np.linalg.solve(J, -F)
                except:
                    dU = -np.linalg.pinv(J).dot(F)
                U += dU
                U[2:] = np.clip(U[2:], 1e-8, None)
            u[k+1,j], v[k+1,j], p[k+1,j], rho[k+1,j] = U

    w = np.hypot(u, v)
    gamma = 1.4
    M = w / np.sqrt(gamma * p / rho)

    data = {}

    i1 = np.argmin(np.abs(x - x_target))
    w_num, M_num, p_num, rho_num = w[i1, :], M[i1, :], p[i1, :], rho[i1, :]
    w_an, M_an, p_an, rho_an = [], [], [], []
    for j in range(Ny):
        r = math.hypot(x[i1] + 1.1, y[j])
        wa, Ma, pa, ra = analytical_point_source(r)
        w_an.append(wa)
        M_an.append(Ma)
        p_an.append(pa)
        rho_an.append(ra)
    w_an, M_an, p_an, rho_an = map(np.array, (w_an, M_an, p_an, rho_an))
    err_w = 100 * (w_num - w_an) / w_an
    err_M = 100 * (M_num - M_an) / M_an
    data[Ny] = (y, w_num, M_num, w_an, M_an, err_w, err_M)
    return data
