import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import fsolve
import csv
import os

from fastslippy.utilities.math_util import MathUtil

# --- Numerical parameters ---
# lx = 40e3; % nature fault length
lx = 0.2

# --- Material parameters ---
Vp = 0 * 1e-9
P = 40e6  # normal effective stress
rho = 2400
cs = 1645
G = rho * cs * cs
eta = G / 2 / cs

# --- Friction parameters ---
mu0 = 0.594
V0 = 1e-6
L = 2.5e-4  # characteristic slip distance
a = 0.012
b = 0.010
t_load = 2000

# --- Initial condition ---
tauqs = 0.0
U = 0.0
V = 1e-10
# theta = L / Vp * 1e10;
# tau0 = P * a * asinh(V / (2 * V0) * exp((mu0 + b * log(V0 / Vp)) / a)) + eta * V;
# tau0 = P * a * asinh(V / (2 * V0) * exp((mu0 + b * log(V0 * theta / L)) / a)) + eta * V;
tau0 = P * 0.3 + eta * V
tau = tauqs + tau0 - eta * V
theta = L / V0 * np.exp(a / b * np.log(2 * V0 / V * np.sinh(tau / a / P)) - mu0 / b)
t = 0.0
dt = 1e0
yr = 365 * 24 * 60 * 60
Vw = 1e30

nt = 330000
Um = np.zeros(nt)
Vm = np.zeros(nt)
taum = np.zeros(nt)
tauqsm = np.zeros(nt)
thetam = np.zeros(nt)
dtm = np.zeros(nt)
tm = np.zeros(nt)

# FF = np.zeros((nt, 6)) # Unused in loop optimization mapping
flag = 0
colordot = 'r.'

# Variable to track it over loops
it_final = 0

# --- First loop (t_load) ---
for it in range(1, t_load + 1):
    idx = it - 1  # 0-based indexing for Python arrays
    
    V = V0 * np.exp((tauqs + tau0 - mu0 * P) / (a * P)) * (theta * V0 / L) ** (-b / a)
    dt = min(max(0.1324 * L / V, 1e-150), 1.2 * dt)
    dt = min(1e20, dt)
    
    if t + dt > t_load:
        dt = t_load - t
        flag = 1
        
    t = t + dt
    theta = theta + dt * (1 - V * theta / L)
    tauqs = tauqs + dt * G / lx * (Vp / 2 - V)
    U = U + dt * V
    
    Um[idx] = U
    Vm[idx] = V
    taum[idx] = tauqs + tau0
    tauqsm[idx] = tauqs
    thetam[idx] = theta
    dtm[idx] = dt
    tm[idx] = t
        
    if flag == 1:
        it_final = it
        break
else:
    it_final = t_load

# --- Second loop (up to nt) ---
Vp = 1e-9
lo = 1e-40
for it in range(it_final + 1, nt + 1):
    idx = it - 1
    
    # Define implicit objective function for fzero (fsolve in Python)
    def objective_v(VV):
        return (P * a * np.arcsinh(VV / (2 * V0) * np.exp((mu0 + b * np.log(V0 * theta / L)) / a)) 
                / (1 + L / Vw / theta) + eta * VV - (tauqs + tau0))
                
    # Using fsolve with the current V value as the initial guess
    #V = fsolve(objective_v, V)[0]
    hi = V * 2

    x, fx, flag = MathUtil.bisection(
                    objective_v,
                    lo,
                    hi,
                    target=0.0,
                    tolX=0.0, #1e-14,
                    tolFun=5,
                    maxiter=100)
    
    if flag > 0:
        V = x

    V = np.maximum(V, 1e-40)
    
    # 1-based index matching for the condition steps
    if 15000 < it < 15010:
        Vp = 0 * 1e-9
    if 15010 < it < 18000:
        Vp = 1e-9
    # 30s
    if 18000 < it < 18030:
        Vp = 0 * 1e-9
    if 18030 < it < 21000:
        Vp = 1e-9
    # 100s
    if 21000 < it < 21100:
        Vp = 0 * 1e-9
    if 21100 < it < 24000:
        Vp = 1e-9
    # 300s
    if 24000 < it < 24300:
        Vp = 0 * 1e-9
    if 24300 < it < 27000:
        Vp = 1e-9
    # 1000s
    if 27000 < it < 27100:
        Vp = 0 * 1e-9
    if 27100 < it < 30000:
        Vp = 1e-9
    # 3000s
    if 30000 < it < 30300:
        Vp = 0 * 1e-9
    if 30300 < it < 33000:
        Vp = 1e-9

    t = t + dt
    theta = theta + dt * (1 - V * theta / L)
    tauqs = tauqs + dt * G / lx * (Vp / 2 - V)
    U = U + dt * V
    
    Um[idx] = U
    Vm[idx] = V
    taum[idx] = tauqs + tau0 - eta * V
    tauqsm[idx] = tauqs
    thetam[idx] = theta
    dtm[idx] = dt
    tm[idx] = t

matplotlib.use("TkAgg")

mu = taum / P
plt.figure(figsize=(8, 4))

x, y = [], []
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'UU-SS-01.csv')
with open(file_path, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        x.append(float(row[0]))
        y.append(float(row[1]))
plt.plot(x, y, '-', color='orange', linewidth=3)


plt.plot(Um * 1000, mu, 'k-', linewidth=1.5)
plt.xlabel('Slip displacement, U [mm]')
plt.ylabel(r'Friction coefficient, $\mu$')
#plt.title('Friction vs Slip Displacement')
plt.xlim([1.0, 4.5])
plt.ylim([0.5, 0.7])
plt.tight_layout()
plt.show()