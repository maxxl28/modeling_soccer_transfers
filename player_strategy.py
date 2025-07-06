# Purpose: create a mathamatical model to simulate soccer 
# tranfers from the Top 5 European Leagues to the Saudi 
# League player perspective

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Define the replicator dynamics equation
def replicator_dynamics(x, a0, d0, b, pGrow, mGrow):
    a = a0 + pGrow * x  # P's payoff when facing P
    d = d0 + mGrow * (1 - x)  # M's payoff when facing M
    fP = a * x + b * (1 - x)  # Avg. payoff for Prestige players
    fM = b * x + d * (1 - x)  # Avg. payoff for Money players
    return x * (1 - x) * (fP - fM), fP, fM, a, d

# Modify
params = {
    'a0': 2.5,       
    'd0': 2.0,       
    'b': 1.4,        
    'pGrow': 1.0,    
    'mGrow': 5.0,    
    'x0': 0.6,       
    't_end': 10      
}

# Solve the ODE
def solve_ode(params):
    t = np.linspace(0, params['t_end'], 1000)
    x = np.zeros(len(t))
    fP, fM, a, d = np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t))
    x[0] = params['x0']
    dt = t[1] - t[0]
    for i in range(1, len(t)):
        dx, fP[i-1], fM[i-1], a[i-1], d[i-1] = replicator_dynamics(x[i-1], 
            **{k: params[k] for k in ['a0', 'd0', 'b', 'pGrow', 'mGrow']})
        x[i] = x[i-1] + dx * dt
    # Calculate final values
    _, fP[-1], fM[-1], a[-1], d[-1] = replicator_dynamics(x[-1], 
        **{k: params[k] for k in ['a0', 'd0', 'b', 'pGrow', 'mGrow']})
    return t, x, fP, fM, a, d

# Set up side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
plt.subplots_adjust(bottom=0.3)

# Initial plot with selected scenario
t, x, fP, fM, a, d = solve_ode(params)

# Population plot
ax1.plot(t, x, 'b-', label='Prestige Players (P)')
ax1.plot(t, 1 - x, 'r-', label='Money Players (M)')
ax1.set_xlabel('Time (Season)')
ax1.set_ylabel('Population Fraction')
ax1.set_title('Player Distribution Over Time')
ax1.legend()
ax1.grid(True)

# Payoff plot
ax2.plot(t, fP, 'b-', label='Avg. Prestige Payoff')
ax2.plot(t, fM, 'r-', label='Avg. Money Payoff')
ax2.plot(t, a, 'b--', label='PvP Payoff')
ax2.plot(t, d, 'r--', label='MvM Payoff')
ax2.set_xlabel('Time (Season)')
ax2.set_ylabel('Payoff Value')
ax2.set_title('Payoff Dynamics')
ax2.legend()
ax2.grid(True)

# Sliders 
axcolor = 'lightgoldenrodyellow'
slider_params = [
    ('a0', 'PvP Base Payoff', 0.1, 5.0, params['a0'], 0.05),
    ('d0', 'MvM Base Payoff', 0.1, 5.0, params['d0'], 0.05),
    ('b', 'Cross-Payoff', 0.1, 5.0, params['b'], 0.05),
    ('pGrow', 'Prestige Growth', 0.1,10.0, params['pGrow'], 0.05),
    ('mGrow', 'Money Growth', 0.1, 10.0, params['mGrow'], 0.05),
    ('x0', 'Initial Prestige %', 0.0, 1.0, params['x0'], 0.05),
    ('t_end', 'Time Range', 1, 50, params['t_end'], 1)
]

sliders = {}
for i, (name, label, vmin, vmax, valinit, valstep) in enumerate(slider_params):
    ax = plt.axes([0.25, 0.2 - i*0.025, 0.65, 0.02], facecolor=axcolor)
    sliders[name] = Slider(ax, label, vmin, vmax, valinit=valinit, valstep=valstep)

# Update function
def update(val):
    for name in params:
        params[name] = sliders[name].val
    t, x, fP, fM, a, d = solve_ode(params)
    
    ax1.clear()
    ax1.plot(t, x, 'b-', label='Prestige Players (P)')
    ax1.plot(t, 1 - x, 'r-', label='Money Players (M)')
    ax1.set_xlabel('Time (Season)')
    ax1.set_ylabel('Population Fraction')
    ax1.set_title('Player Distribution Over Time')
    ax1.legend()
    ax1.grid(True)
    
    ax2.clear()
    ax2.plot(t, fP, 'b-', label='Avg. Prestige Payoff')
    ax2.plot(t, fM, 'r-', label='Avg. Money Payoff')
    ax2.plot(t, a, 'b--', label='PvP Payoff')
    ax2.plot(t, d, 'r--', label='MvM Payoff')
    ax2.set_xlabel('Time (Season)')
    ax2.set_ylabel('Payoff Value')
    ax2.set_title('Payoff Dynamics')
    ax2.legend()
    ax2.grid(True)
    
    fig.canvas.draw_idle()

for slider in sliders.values():
    slider.on_changed(update)

plt.show()