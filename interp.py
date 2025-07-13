import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class LinHist:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        # assert all([v1 == v2 for v1, v2 in zip(sorted(x), x)])
        self.x = x
        self.y = y
        self.y = np.nan_to_num(self.y, nan=0)

    def cdf(self) -> "LinHist":
        cdf = np.cumsum(self.y)
        return LinHist(self.x, cdf / cdf[-1])

    def __len__(self) -> int:
        return len(self.x)

    def eval(self, X) -> np.ndarray:
        return np.interp(X, self.x, self.y)

    def area(self) -> float:
        return np.sum((self.y[1:] + self.y[:-1]) * (self.x[1:] - self.x[:-1])) * 0.5


def interp(h1: LinHist, h2: LinHist, alpha: float):
    assert len(h1) == len(h2)
    assert len(h1.x) == len(h2.x)

    cdf1 = h1.cdf()
    cdf2 = h2.cdf()

    # Check whether cdf1 or cdf2 has more y values

    xspace = cdf1.x

    x1 = h1.x
    x2 = LinHist(cdf2.y, cdf2.x).eval(cdf1.y)

    f1_x1 = LinHist(xspace, h1.y).eval(x1)
    f2_x2 = LinHist(xspace, h2.y).eval(x2)

    f = f1_x1 * f2_x2
    D = (1.0 - alpha) * f2_x2 + alpha * f1_x1
    D[D == 0] = 1
    f /= D
    return LinHist(x1 * (1-alpha) + alpha* x2, f)

# def interp(h1: LinHist, h2: LinHist, alpha: float):
#     assert len(h1) == len(h2)
#     assert len(h1.x) == len(h2.x)
#
#     cdf1 = h1.cdf()
#     cdf2 = h2.cdf()
#
#     yspace = np.linspace(0, 1, 100)
#     xspace = np.linspace(h1.x[0], h1.x[-1], 100)
#
#     x1 = LinHist(cdf1.y, cdf1.x).eval(yspace)
#     x2 = LinHist(cdf2.y, cdf2.x).eval(yspace)
#
#     f1_x1 = LinHist(xspace, h1.eval(xspace)).eval(x1)
#     f2_x2 = LinHist(xspace, h2.eval(xspace)).eval(x2)
#
#     f = f1_x1 * f2_x2
#     D = (1.0 - alpha) * f2_x2 + alpha * f1_x1
#     D[D == 0] = 1
#     f /= D
#     return LinHist(x1 * (1-alpha) + alpha* x2, f)


data1 = np.random.normal(loc=-1.3, scale=0.05, size=100000)
data2 = np.random.normal(loc=0.7, scale=0.6, size=100000)

hrange = (-3, 3)
nbins = 40
h1, e1 = np.histogram(data1, bins=nbins, range=hrange)
h2, e2 = np.histogram(data2, bins=nbins, range=hrange)
h1 = np.array(h1, dtype=np.float32)
h2 = np.array(h2, dtype=np.float32)
bwidth = (hrange[1] - hrange[0]) / nbins
e1 = e1[:-1] + bwidth
e2 = e2[:-1] + bwidth

H1 = LinHist(e1, h1)
H2 = LinHist(e2, h2)
xhighres = np.linspace(*hrange, 1000)

HI = interp(H1, H2, 0.01)

plt.subplot(2, 1, 1)
plt.step(H1.x, H1.y, where="mid")
plt.plot(xhighres, H1.eval(xhighres), linewidth=0.5, color="k")
plt.step(H2.x, H2.y, where="mid")
plt.plot(xhighres, H2.eval(xhighres), linewidth=0.5, color="k")
plt.step(H1.x, HI.eval(H1.x), where="mid")
print(HI.eval(H1.x))
plt.plot(xhighres, HI.eval(xhighres), linewidth=0.5, color="k")
plt.ylabel("Counts")


plt.subplot(2, 1, 2)
cdf1 = H1.cdf()
cdf2 = H2.cdf()
plt.step(H1.x, y=cdf1.y, where="mid")
plt.plot(xhighres, cdf1.eval(xhighres), linewidth=0.5, color="k")
plt.step(H2.x, y=cdf2.y, where="mid")
plt.plot(xhighres, cdf2.eval(xhighres), linewidth=0.5, color="k")
HICDF = HI.cdf()
plt.step(HICDF.x, y=HICDF.y, where="mid")
plt.plot(xhighres, HICDF.eval(xhighres), linewidth=0.5, color="k")
plt.ylabel("CDF")
plt.xlabel("x")
plt.savefig("result.pdf")
