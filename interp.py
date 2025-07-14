import numpy as np


class LinHist:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    def cdf(self) -> "LinHist":
        cdf = np.cumsum(self.y)
        return LinHist(self.x, cdf / cdf[-1])

    def __len__(self) -> int:
        return len(self.x)

    def eval(self, X) -> np.ndarray:
        return np.interp(X, self.x, self.y)


def interp_fast(h1: LinHist, h2: LinHist, alpha: float) -> LinHist:
    assert len(h1) == len(h2)
    assert len(h1.x) == len(h2.x)

    cdf1 = h1.cdf()
    cdf2 = h2.cdf()

    # Check whether cdf1 or cdf2 has more evenly distributed y values
    # Use that histogram to obtain x1
    quality_cdf1 = ((cdf1.y[1:] - cdf1.y[:-1])**2).sum()
    quality_cdf2 = ((cdf2.y[1:] - cdf2.y[:-1])**2).sum()
    if quality_cdf1 > quality_cdf2:
        h1, h2 = h2, h1
        cdf1, cdf2 = cdf2, cdf1
        alpha = 1 - alpha

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

def interp_high_precision(h1: LinHist, h2: LinHist, alpha: float, precision=100) -> LinHist:
    assert len(h1) == len(h2)
    assert len(h1.x) == len(h2.x)

    cdf1 = h1.cdf()
    cdf2 = h2.cdf()

    yspace = np.linspace(0, 1, precision)
    xspace = np.linspace(h1.x[0], h1.x[-1], precision)

    x1 = LinHist(cdf1.y, cdf1.x).eval(yspace)
    x2 = LinHist(cdf2.y, cdf2.x).eval(yspace)

    f1_x1 = LinHist(xspace, h1.eval(xspace)).eval(x1)
    f2_x2 = LinHist(xspace, h2.eval(xspace)).eval(x2)

    f = f1_x1 * f2_x2
    D = (1.0 - alpha) * f2_x2 + alpha * f1_x1
    D[D == 0] = 1
    f /= D
    return LinHist(x1 * (1-alpha) + alpha* x2, f)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    data1 = np.random.normal(loc=-1.3, scale=0.25, size=100000)
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

    alpha = 0.71
    HI = interp_fast(H1, H2, alpha)

    plt.subplot(2, 1, 1)
    plt.step(H1.x, H1.y, where="mid", label="$H_1$")
    # plt.plot(xhighres, H1.eval(xhighres), linewidth=0.5, color="k")
    plt.step(H2.x, H2.y, where="mid", label="$H_2$")
    # plt.plot(xhighres, H2.eval(xhighres), linewidth=0.5, color="k")
    plt.step(H1.x, HI.eval(H1.x), color="k", where="mid", label=rf"$H_\text{{Interp}}(\alpha={alpha})$")
    plt.legend(loc="upper right")
    # plt.plot(xhighres, HI.eval(xhighres), linewidth=0.5, color="k")
    plt.ylabel("Counts")


    plt.subplot(2, 1, 2)
    cdf1 = H1.cdf()
    cdf2 = H2.cdf()
    plt.step(H1.x, y=cdf1.y, where="mid", label="$CDF_1$")
    plt.plot(xhighres, cdf1.eval(xhighres), linestyle='--', linewidth=0.5, color="k")
    plt.step(H2.x, y=cdf2.y, where="mid", label="$CDF_2$")
    plt.plot(xhighres, cdf2.eval(xhighres), linestyle='--', linewidth=0.5, color="k")
    HICDF = HI.cdf()
    plt.step(HICDF.x, y=HICDF.y, where="mid", color="k", label=rf"$CDF_\text{{Interp}}(\alpha={alpha})$")
    plt.plot(xhighres, HICDF.eval(xhighres), linestyle='--', linewidth=0.5, color="k", label="Linearized CDF")
    plt.legend(loc="lower right")
    plt.ylabel("CDF")
    plt.xlabel("x")
    plt.savefig("result.pdf")
