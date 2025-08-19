import numpy as np


class LinHist:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    def cdf(self) -> "LinHist":
        cdf = np.cumsum(self.y)
        return LinHist(self.x, cdf / cdf[-1])

    def inverse(self) -> "LinHist":
        return LinHist(self.y, self.x)

    def derivative(self) -> "LinHist":
        dy = np.diff(self.y, prepend=0)
        dx = np.diff(self.x, prepend=0)
        return LinHist(self.x, dy / dx)

    def area(self) -> float:
        return np.sum(self.y * np.diff(self.x, prepend=0))

    def __len__(self) -> int:
        return len(self.x)

    def eval(self, X) -> np.ndarray:
        return np.interp(X, self.x, self.y)


def interp(h1: LinHist, h2: LinHist, alpha: float) -> LinHist:
    assert len(h1) == len(h2)
    assert len(h1.x) == len(h2.x)

    cdf1 = h1.cdf()
    cdf2 = h2.cdf()

    area1 = h1.area()
    area2 = h2.area()
    interp_area = (1 - alpha) * area1 + alpha * area2

    ally = np.sort(np.unique(np.concatenate((np.linspace(0, 1, 100), cdf1.y, cdf2.y))))

    # if len(ally) < 10:
    #     ally = np.linspace(0, 1, 101)

    x1 = cdf1.inverse().eval(ally)
    x2 = cdf2.inverse().eval(ally)
    xmid = x1 * (1 - alpha) + alpha * x2

    h3y = LinHist(xmid, ally * interp_area).eval(h1.x)
    h3 = LinHist(h1.x, h3y).derivative()

    return LinHist(h1.x, h3.eval(h1.x))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    N = 100000
    data1 = np.random.normal(loc=-1.3, scale=0.002, size=N)
    data2 = np.random.normal(loc=0.7, scale=0.002, size=int(1.3*N))

    hrange = (-3, 3)
    nbins = 40
    h1, e1 = np.histogram(data1, bins=nbins, range=hrange)
    h2, e2 = np.histogram(data2, bins=nbins, range=hrange)
    h1 = np.array(h1, dtype=np.float32)
    h2 = np.array(h2, dtype=np.float32)
    bwidth = (hrange[1] - hrange[0]) / nbins
    e1 = e1[:-1] + 0.5 * bwidth
    e2 = e2[:-1] + 0.5 * bwidth

    H1 = LinHist(e1, h1)
    H2 = LinHist(e2, h2)
    xhighres = np.linspace(*hrange, 1000)

    alpha = 0.5
    HI = interp(H1, H2, alpha)

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
