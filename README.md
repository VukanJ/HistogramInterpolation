## Linear interpolation of 1D histograms
Method:
  *A.L Read, "Linear interpolation of histograms", Nucl. Instrum. Methods Phys. Res. A425 (1999) 357*

![Example](example.pdf)

```python
from interp import interp, LinHist

# Fill example histograms
h1, e1 = np.histogram(np.random.normal(loc=-1.3, scale=0.15, size=100000), bins=40, range=(-3, 3))
h2, e2 = np.histogram(np.random.normal(loc=0.7, scale=0.65, size=130000), bins=40, range=(-3, 3))
bwidth = 6 / 40

# Convert to LinHist class instances
H1 = LinHist(e1[:-1] + 0.5 * bwidth, h1)
H2 = LinHist(e2[:-1] + 0.5 * bwidth, h2)

alpha = 0.6 # interpolation parameter
Hinterp = interp(H1, H2, alpha=alpha)

# plot
plt.step(H1.x, H1.y, where="mid", label="$H_1$")
plt.step(H2.x, H2.y, where="mid", label="$H_2$")
plt.step(Hinterp.x, Hinterp.y, where="mid", label="Interpolated")
plt.xlabel("$x$")
plt.ylabel("Counts")
plt.legend(loc=best)
plt.show()
```
