# Purpose: create a mathamatical model to simulate soccer 
# tranfers from the Top 5 European Leagues to the Saudi 
# League club perspective

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Payoff matrix [Saudi, Europe]


payoffs = {
    ('Youth', 'Youth'): (4, 4),    # (Saudi, Europe)
    ('Youth', 'Star'): (2, 5),
    ('Star', 'Youth'): (5, 2),
    ('Star', 'Star'): (1, 1)    
}



def replicator_dynamics(x, y):
    """x = prob Saudi chooses Youth, y = prob Europe chooses Youth"""
    # Saudi payoffs
    saudi_youth_payoff = y * payoffs[('Youth', 'Youth')][0] + (1-y) * payoffs[('Youth', 'Star')][0]
    saudi_star_payoff = y * payoffs[('Star', 'Youth')][0] + (1-y) * payoffs[('Star', 'Star')][0]
    saudi_avg = x * saudi_youth_payoff + (1-x) * saudi_star_payoff
    
    # Europe payoffs
    euro_youth_payoff = x * payoffs[('Youth', 'Youth')][1] + (1-x) * payoffs[('Star', 'Youth')][1]
    euro_star_payoff = x * payoffs[('Youth', 'Star')][1] + (1-x) * payoffs[('Star', 'Star')][1]
    euro_avg = y * euro_youth_payoff + (1-y) * euro_star_payoff
    
    dx = x * (1 - x) * (saudi_youth_payoff - saudi_avg)
    dy = y * (1 - y) * (euro_youth_payoff - euro_avg)
    
    return dx, dy

# Simulation parameters
params = {
    'x0': 0.5,  # Initial prob Saudi chooses Youth
    'y0': 0.5,  # Initial prob Europe chooses Youth
    't_end': 10
}

def simulate(params):
    t = np.linspace(0, params['t_end'], 1000)
    x = np.zeros(len(t))
    y = np.zeros(len(t))
    x[0], y[0] = params['x0'], params['y0']
    dt = t[1] - t[0]
    
    for i in range(1, len(t)):
        dx, dy = replicator_dynamics(x[i-1], y[i-1])
        x[i] = np.clip(x[i-1] + dx * dt, 0, 1)
        y[i] = np.clip(y[i-1] + dy * dt, 0, 1)
    
    return t, x, y

# Set up plots: 3 subplots now
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
plt.subplots_adjust(bottom=0.25)

# Initial simulation
t, x, y = simulate(params)

# Plot Saudi marginal strategies
ax1.plot(t, x, 'g-', label='Saudi Youth Dev Probability')
ax1.plot(t, 1-x, 'm-', label='Saudi Superstar Probability')
ax1.set_title('Saudi Club Strategy Evolution')
ax1.set_xlabel('Years')
ax1.legend()

# Plot Europe marginal strategies
ax2.plot(t, y, 'b-', label='Europe Youth Dev Probability')
ax2.plot(t, 1-y, 'c-', label='Europe Superstar Probability')
ax2.set_title('European Club Strategy Evolution')
ax2.set_xlabel('Years')
ax2.legend()

# Plot joint population proportions of all 4 strategy pairs
ax3.plot(t, x*y, 'g-', label='Saudi Youth & Europe Youth')
ax3.plot(t, x*(1 - y), 'm-', label='Saudi Youth & Europe Star')
ax3.plot(t, (1 - x)*y, 'b-', label='Saudi Star & Europe Youth')
ax3.plot(t, (1 - x)*(1 - y), 'c-', label='Saudi Star & Europe Star')
ax3.set_title('Joint Strategy Populations Over Time')
ax3.set_xlabel('Years')
ax3.legend()

# Sliders
axcolor = 'lightgoldenrodyellow'
sliders = {
    'x0': Slider(plt.axes([0.25, 0.15, 0.65, 0.02], facecolor=axcolor),
                 label='Initial Saudi Youth %', valmin=0, valmax=1, valinit=0.5, valstep=.01),
    'y0': Slider(plt.axes([0.25, 0.10, 0.65, 0.02], facecolor=axcolor),
                 label='Initial Europe Youth %', valmin=0, valmax=1, valinit=0.5, valstep=.01),
    't_end': Slider(plt.axes([0.25, 0.05, 0.65, 0.02], facecolor=axcolor),
                    label='Time Horizon', valmin=1, valmax=40, valinit=10)
}

def update(val):
    for name in params:
        params[name] = sliders[name].val
    t, x, y = simulate(params)
    
    # Update Saudi plot
    ax1.clear()
    ax1.plot(t, x, 'g-', label='Saudi Youth Dev')
    ax1.plot(t, 1-x, 'm-', label='Saudi Superstars')
    ax1.set_title('Saudi Club Strategy Evolution')
    ax1.set_xlabel('Years')
    ax1.legend()
    
    # Update Europe plot
    ax2.clear()
    ax2.plot(t, y, 'b-', label='Europe Youth Dev')
    ax2.plot(t, 1-y, 'c-', label='Europe Superstars')
    ax2.set_title('European Club Strategy Evolution')
    ax2.set_xlabel('Years')
    ax2.legend()
    
    # Update joint population plot
    ax3.clear()
    ax3.plot(t, x*y, 'g-', label='Saudi Youth & Europe Youth')
    ax3.plot(t, x*(1 - y), 'm-', label='Saudi Youth & Europe Star')
    ax3.plot(t, (1 - x)*y, 'b-', label='Saudi Star & Europe Youth')
    ax3.plot(t, (1 - x)*(1 - y), 'c-', label='Saudi Star & Europe Star')
    ax3.set_title('Joint Strategy Populations Over Time')
    ax3.set_xlabel('Years')
    ax3.legend()
    
    fig.canvas.draw_idle()

for slider in sliders.values():
    slider.on_changed(update)

plt.show()
