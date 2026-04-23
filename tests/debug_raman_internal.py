"""检查 r3 与信号频率的匹配"""

import numpy as np
from pynlo.physics.grid import Grid
from pynlo.utility.chi3 import raman_from_dat
from scipy.fft import irfft, rfft

N = 2**16
BANDWIDTH = 2e12
V_CENTER = 193.1e12

grid = Grid.create(n=N, v_center=V_CENTER, bandwidth=BANDWIDTH, v_comoving=V_CENTER)
dt = grid.dt

# 检查 rv_grid 与信号频率的关系
_, raman_r3 = raman_from_dat(
    N, dt, filepath="D:/C-Codes/Optical-Networks/PyNLO/pynlo/utility/RamanGain.dat", raman_frac=0.18, analytic=True
)

print(f"Grid: n={N}, dt={dt * 1e15:.4f} fs, dv={grid.dv / 1e9:.4f} GHz")
print(f"Grid v_grid range: [{grid.v_grid[0] / 1e12:.4f}, {grid.v_grid[-1] / 1e12:.4f}] THz")
print(f"Grid v_grid [0]: {grid.v_grid[0]}, v_grid[-1]: {grid.v_grid[-1]}")

# r3 rv_grid 范围 (raman_from_dat 返回的)
# raman_from_dat 使用 rv_grid = np.arange(n//2 + 1) * dv
rv_grid_from_raman = np.arange(N // 2 + 1) * grid.dv
print(f"\nr3 rv_grid range: [{rv_grid_from_raman[0] / 1e9:.4f}, {rv_grid_from_raman[-1] / 1e9:.4f}] GHz")
print(f"r3 length: {len(raman_r3)}, expected: {N // 2 + 1}")

# 信号频率与拉曼频率的对比
# 我们的信号在 V_CENTER +/- 100 GHz
signal_freqs = [V_CENTER - 100e9, V_CENTER + 100e9]
print(f"\nSignal frequencies: {[f / 1e12 for f in signal_freqs]} THz")

# SRS 需要的频率偏移范围：最多 ~13 THz (拉曼增益谱宽度)
# 而信号间隔只有 200 GHz = 0.2 THz << 13 THz
print("\n拉曼频率偏移范围: ~0-1000 GHz")
print("信号频率间隔: 200 GHz")
print("所以 SRS 效应应该是可见的，因为 200 GHz < 1000 GHz")

# 测试：手动计算拉曼卷积
t = grid.t_grid
signal1 = np.exp(1j * 2 * np.pi * (V_CENTER - 100e9) * t)
signal2 = np.exp(1j * 2 * np.pi * (V_CENTER + 100e9) * t)
combined = signal1 + signal2

I_total = np.abs(combined) ** 2

# 直接卷积测试
I_rv = rfft(I_total) * dt
I_raman_rv = raman_r3 * I_rv
I_raman = irfft(I_raman_rv, n=N) / dt

diff = np.max(np.abs(I_raman - I_total))
print("\n拉曼卷积测试:")
print(f"  Original I_max: {np.max(I_total):.6e}")
print(f"  Raman I_max: {np.max(I_raman):.6e}")
print(f"  Difference: {diff:.6e}")
print(f"  Relative diff: {diff / np.max(I_total) * 100:.4f}%")

if diff < 1e-10:
    print("  WARNING: 拉曼卷积没有效果！")
else:
    print("  拉曼卷积正常工作！")
