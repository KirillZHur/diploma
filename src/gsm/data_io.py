import numpy as np
import matplotlib.pyplot as plt
from solver import run_schm
from solver import analytical_point_source
import math

def plot_profiles(x_target: float, Ny_list: list):

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    (ax_w, ax_M), (ax_ew, ax_eM) = axes

    for Ny in Ny_list:
        data = run_schm(Ny, x_target=x_target)
        y, w_num, M_num, w_an, M_an, _, _ = data[Ny]
        ax_w.plot(y, w_num, label=f'СХМ Ny={Ny}')
        ax_M.plot(y, M_num, label=f'СХМ Ny={Ny}')

    y0 = data[Ny_list[0]][0]

    ax_w.plot(y0, data[Ny_list[0]][3], 'k--', lw=2, label='Analytic')
    ax_M.plot(y0, data[Ny_list[0]][4], 'k--', lw=2, label='Analytic')

    ax_w.set_title('Распределение скорости в x=2.0 м')
    ax_M.set_title('Распределение числа Маха в x=2.0 м')
    for ax in (ax_w, ax_M):
        ax.set_xlabel('y, m')
        ax.set_ylabel('Value')
        ax.grid(True)
        ax.legend()

    for Ny in Ny_list:
        y, _, _, _, _, err_w, err_M = data[Ny]
        ax_ew.plot(y, err_w, label=f'Error w Ny={Ny}')
        ax_eM.plot(y, err_M, label=f'Error M Ny={Ny}')

    ax_ew.set_title('Speed error (%)')
    ax_eM.set_title('Mach error (%)')
    for ax in (ax_ew, ax_eM):
        ax.set_xlabel('y, m')
        ax.set_ylabel('Error %')
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()
