import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from interp import LinHist, interp

# Initial parameters
loc1_init, scale1_init = 0, 1
loc2_init, scale2_init = 2, 1.5
bins_init = 30
n_samples = 100000
Range = (-10, 10)

# Generate data
data1 = np.random.normal(loc1_init, scale1_init, n_samples)
data2 = np.random.normal(loc2_init, scale2_init, n_samples)
h1, e1 = np.histogram(data1, bins=bins_init, range=Range)
h2, e2 = np.histogram(data2, bins=bins_init, range=Range)
e1 = e1[:-1] + 0.5 * (e1[1] - e1[0])  # Center the bins
e2 = e2[:-1] + 0.5 * (e2[1] - e2[0])  # Center the bins
H1 = LinHist(e1, h1)
H2 = LinHist(e2, h2)
HI = interp(H1, H2, 0.5)

# Figure and axes
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.35)
hist1 = ax.hist(data1, bins=bins_init, alpha=0.6, label="Dist 1")
hist2 = ax.hist(data2, bins=bins_init, alpha=0.6, label="Dist 2")
ax.legend()

# Slider axes
axcolor = 'lightgoldenrodyellow'
ax_alpha = plt.axes([0.1, 0.25, 0.65, 0.03], facecolor=axcolor)
ax_bins  = plt.axes([0.1, 0.2, 0.65, 0.03], facecolor=axcolor)
ax_loc1  = plt.axes([0.1, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_scale1= plt.axes([0.1, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_loc2  = plt.axes([0.1, 0.05, 0.65, 0.03], facecolor=axcolor)
ax_scale2= plt.axes([0.1, 0.0, 0.65, 0.03], facecolor=axcolor)

# Sliders
s_bins   = Slider(ax_bins, 'Bins', 5, 100, valinit=bins_init, valstep=1)
s_loc1   = Slider(ax_loc1, 'Loc1', -5, 5, valinit=loc1_init)
s_scale1 = Slider(ax_scale1, 'Scale1', 0, 3, valinit=scale1_init)
s_loc2   = Slider(ax_loc2, 'Loc2', -5, 5, valinit=loc2_init)
s_scale2 = Slider(ax_scale2, 'Scale2', 0, 3, valinit=scale2_init)
s_alpha = Slider(ax_alpha, 'Alpha', 0, 1, valinit=0.5)

# Update function
def update(val):
    global HI
    HI = interp(H1, H2, s_alpha.val)
    ax.clear()
    bins = int(s_bins.val)

    ax.step(H1.x, H1.y, where='mid', label='H1', color='k')
    ax.step(H2.x, H2.y, where='mid', label='H2', color='k')
    ax.step(HI.x, HI.y, where='mid', label='H2', color='r')

    ax.legend()
    fig.canvas.draw_idle()


def update_data(val):
    """Update the data based on slider values."""
    print("Updating data with new parameters...")
    global data1, data2, h1, h2, e1, e2, H1, H2
    loc1, scale1 = s_loc1.val, s_scale1.val
    loc2, scale2 = s_loc2.val, s_scale2.val
    
    # Generate new data
    data1 = np.random.normal(loc1, scale1, n_samples)
    data2 = np.random.normal(loc2, scale2, n_samples)

    h1, e1 = np.histogram(data1, bins=s_bins.val, range=Range)
    h2, e2 = np.histogram(data2, bins=s_bins.val, range=Range)
    e1 = e1[:-1] + 0.5 * (e1[1] - e1[0])  # Center the bins
    e2 = e2[:-1] + 0.5 * (e2[1] - e2[0])  # Center the bins
    H1 = LinHist(e1, h1)
    H2 = LinHist(e2, h2)
    
    update(val)

# Connect sliders
s_loc1.on_changed(update_data)
s_scale1.on_changed(update_data)
s_loc2.on_changed(update_data)
s_scale2.on_changed(update_data)
s_bins.on_changed(update_data)

s_alpha.on_changed(update)

plt.show()

