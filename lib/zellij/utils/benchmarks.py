# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-03T22:38:46+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)

import numpy as np
from abc import ABC, abstractmethod


def himmelblau(x):
    x_ar = np.array(x)
    return np.sum(x_ar ** 4 - 16 * x_ar ** 2 + 5 * x_ar) * (1 / len(x_ar))


class Benchmark(ABC):
    def __init__(
        self,
        lower,
        upper,
        optimum,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        super(Benchmark, self).__init__()
        self.lower = lower
        self.upper = upper
        self.optimum = optimum
        self.shift = shift
        self.rotate = rotate
        self.shuffle = shuffle
        self.bias = bias

    @property
    def shift(self):
        return self._shift

    @shift.setter
    def shift(self, shift):
        self._shift = shift

    @property
    def rotate(self):
        return self._rotate

    @rotate.setter
    def rotate(self, rotate):
        self._rotate = rotate

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        self._shuffle = shuffle

    def _shift_point(self, y):
        if self.shift is None:
            res = np.array(y, dtype=float)
        else:
            res = np.array(y, dtype=float) - self.shift[: len(y)]

        if isinstance(res, np.ndarray):
            return res
        else:
            return np.array([res])

    def _rotate_point(self, y):
        if self.rotate is None:
            res = y
        else:
            res = np.dot(self.rotate[: len(y), : len(y)], y)

        if isinstance(res, np.ndarray):
            return res
        else:
            return np.array([res])

    def transform(self, y):
        return self._rotate_point(self._shift_point(y))

    @abstractmethod
    def __call__(self, y):
        pass


class Sphere(Benchmark):
    def __init__(
        self,
        lower=-100.0,
        upper=100.0,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        super(Sphere, self).__init__(
            lower, upper, optimum, shift, rotate, shuffle, bias
        )

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        return np.sum(np.square(z)) + self.bias


class Schwefel_problem(Benchmark):
    def __init__(
        self,
        lower=-100.0,
        upper=100.0,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        super(Schwefel_problem, self).__init__(
            lower, upper, optimum, shift, rotate, shuffle, bias
        )

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        return np.max(np.abs(z)) + self.bias


class Rosenbrock(Benchmark):
    def __init__(
        self,
        lower=-100,
        upper=100,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        super(Rosenbrock, self).__init__(
            lower, upper, optimum, shift, rotate, shuffle, bias
        )

    def __call__(self, y):
        z = self.transform(y) + 1
        if self.shuffle is not None:
            z = z[self.shuffle]
        return (
            np.sum((z[:-1] - 1) ** 2 + 100 * (z[:-1] ** 2 - z[1:]) ** 2)
            + self.bias
        )


class Rastrigin(Benchmark):
    def __init__(
        self,
        lower=-5,
        upper=5,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        super(Rastrigin, self).__init__(
            lower, upper, optimum, shift, rotate, shuffle, bias
        )

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        return np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z) + 10) + self.bias


class Griewank(Benchmark):
    def __init__(
        self,
        lower=-600,
        upper=600,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        super(Griewank, self).__init__(
            lower, upper, optimum, shift, rotate, shuffle, bias
        )

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        return (
            np.sum(z ** 2 / 4000)
            - np.prod(np.cos(z / np.sqrt(np.arange(1, len(y) + 1))))
            + 1
            + self.bias
        )


class Ackley(Benchmark):
    def __init__(
        self,
        lower=-32,
        upper=32,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        super(Ackley, self).__init__(
            lower, upper, optimum, shift, rotate, shuffle, bias
        )

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        return (
            -20 * np.exp(-0.2 * np.sqrt(np.mean(z ** 2)))
            - np.exp(np.mean(np.cos(2 * np.pi * z)))
            + 20
            + np.exp(1.0)
            + self.bias
        )


class Schwefel_problem_2_22(Benchmark):
    def __init__(
        self,
        lower=-10,
        upper=10,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        super(Schwefel_problem_2_22, self).__init__(
            lower, upper, optimum, shift, rotate, shuffle, bias
        )

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        z = np.abs(z)
        return np.sum(z) + np.prod(z) + self.bias


class Schwefel_problem_1_2(Benchmark):
    def __init__(
        self,
        lower=-65.536,
        upper=65.536,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        super(Schwefel_problem_1_2, self).__init__(
            lower, upper, optimum, shift, rotate, shuffle, bias
        )

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        sum = 0
        for i in range(len(z)):
            sum += np.sum(z[:i]) ** 2

        return sum + self.bias


class F10(Benchmark):
    def __init__(
        self,
        lower=-100,
        upper=100,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        super(F10, self).__init__(
            lower, upper, optimum, shift, rotate, shuffle, bias
        )

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        F = (z[:-1] ** 2 + z[1:] ** 2) ** 0.25 * np.sin(
            50 * (z[:-1] ** 2 + z[1:] ** 2) ** 0.1
        ) ** 2 + 1
        return (
            np.sum(F)
            + (z[-1] ** 2 + z[0] ** 2) ** 0.25
            * np.sin(50 * (z[-1] ** 2 + z[0] ** 2) ** 0.1) ** 2
            + 1
            + self.bias
        )


class Bohachevsky(Benchmark):
    def __init__(
        self,
        lower=-15,
        upper=15,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        super(Bohachevsky, self).__init__(
            lower, upper, optimum, shift, rotate, shuffle, bias
        )

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        res = (
            np.sum(
                z[:-1] ** 2
                + 2 * z[1:] ** 2
                - 0.3 * np.cos(3 * np.pi * z[:-1])
                - 0.4 * np.cos(4 * np.pi * z[1:])
                + 0.7
            )
            + self.bias
        )

        return res


class Schaffer(Benchmark):
    def __init__(
        self,
        lower=-100,
        upper=100,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        super(Schaffer, self).__init__(
            lower, upper, optimum, shift, rotate, shuffle, bias
        )

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        F = (z[:-1] ** 2 + z[1:] ** 2) ** 0.25 * np.sin(
            50 * (z[:-1] ** 2 + z[1:] ** 2) ** 0.1
        ) ** 2 + 1
        return np.sum(F) + self.bias


class Styblinsky_tang(Benchmark):
    def __init__(
        self,
        lower=-5,
        upper=5,
        optimum=-39.16599,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        super(Styblinsky_tang, self).__init__(
            lower, upper, optimum, shift, rotate, shuffle, bias
        )

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        return np.sum(z ** 4 - 16 * z ** 2 + 5 * z) / 2 + self.bias


class Alpine(Benchmark):
    def __init__(
        self,
        lower=-10,
        upper=10,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        super(Alpine, self).__init__(
            lower, upper, optimum, shift, rotate, shuffle, bias
        )

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        return np.sum(np.absolute(z * np.sin(z) + 0.1 * z)) + self.bias


# CEC2020
# realnum 1
class Cigar(Benchmark):
    def __init__(
        self,
        lower=-100,
        upper=100,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        super(Cigar, self).__init__(
            lower, upper, optimum, shift, rotate, shuffle, bias
        )

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        return z[0] ** 2 + (10 ** 6) * np.sum(z[1:] ** 2) + self.bias


class Happycat(Benchmark):
    def __init__(
        self,
        alpha=1 / 8,
        lower=-2,
        upper=2,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        super(Happycat, self).__init__(
            lower, upper, optimum, shift, rotate, shuffle, bias
        )
        self.alpha = alpha

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        znorm = np.sum(z ** 2)
        return (
            ((znorm - len(z)) ** 2) ** self.alpha
            + (0.5 * znorm + np.sum(z)) / len(z)
            + 0.5
            + self.bias
        )


class Levy(Benchmark):
    def __init__(
        self,
        lower=-100,
        upper=100,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        super(Levy, self).__init__(
            lower, upper, optimum, shift, rotate, shuffle, bias
        )

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        z = 1 + (z - 1) / 4
        return (
            np.sin(np.pi * y[0]) ** 2
            + np.sum(
                (y[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * y[:-1] + 1) ** 2)
            )
            + (y[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * y[-1]) ** 2)
            + self.bias
        )


class Brown(Benchmark):
    def __init__(
        self,
        lower=-1,
        upper=4,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        super(Brown, self).__init__(
            lower, upper, optimum, shift, rotate, shuffle, bias
        )

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        return np.sum(
            (z[:-1] ** 2) ** (z[1:] ** 2 + 1)
            + (z[1:] ** 2) ** (z[:-1] ** 2 + 1)
            + self.bias
        )


class High_conditioned_elliptic(Benchmark):
    def __init__(
        self,
        lower=-1,
        upper=4,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        super(High_conditioned_elliptic, self).__init__(
            lower, upper, optimum, shift, rotate, shuffle, bias
        )

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        return np.sum(
            (10 ** 6) ** (np.arange(0, len(z)) / (len(z) - 1)) * z ** 2
            + self.bias
        )


class HGBat(Benchmark):
    def __init__(
        self,
        alpha=1 / 8,
        lower=-2,
        upper=2,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        super(HGBat, self).__init__(
            lower, upper, optimum, shift, rotate, shuffle, bias
        )
        self.alpha = alpha

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        znorm = np.sum(z ** 2)
        zsum = np.sum(z)
        return (
            np.abs(znorm ** 2 - zsum ** 2) ** self.alpha
            + (0.5 * znorm + zsum) / len(z)
            + 0.5
            + self.bias
        )


class Discus(Benchmark):
    def __init__(
        self,
        lower=-5,
        upper=5,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        super(Discus, self).__init__(
            lower, upper, optimum, shift, rotate, shuffle, bias
        )

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        return 10 ** 6 * z[0] ** 2 + np.sum(z[1:] ** 2) + self.bias


class Lunacek(Benchmark):
    def __init__(
        self,
        lower=-5.12,
        upper=5.12,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        super(Lunacek, self).__init__(
            lower, upper, optimum, shift, rotate, shuffle, bias
        )

    def __call__(self, y):
        z = self._rotate_point(self._shift_point(y))
        if self.shuffle is not None:
            z = z[self.shuffle]
        d = 1.0
        s = 1.0 - 1.0 / (2.0 * np.sqrt(len(z) + 20.0) - 8.2)
        mu0 = 2.5
        mu1 = -np.sqrt((mu0 ** 2 - d) / s)
        return (
            np.min(
                [
                    np.sum((z - mu0) ** 2),
                    d * len(z) + s * np.sum((z - mu1) ** 2),
                ]
            )
            + 10 * np.sum(1 - np.cos(2 * np.pi * (z - mu0)))
            + self.bias
        )


# CEC2020
# realnum 3
class Lunacek_bi_rastrigin(Benchmark):
    def __init__(
        self,
        lower=-5.12,
        upper=5.12,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        super(Lunacek_bi_rastrigin, self).__init__(
            lower, upper, optimum, shift, rotate, shuffle, bias
        )

    def __call__(self, y):
        z = 0.1 * self._shift_point(y)
        if self.shuffle is not None:
            z = z[self.shuffle]

        d = 1.0
        s = 1.0 - 1.0 / (2.0 * np.sqrt(len(z) + 20.0) - 8.2)
        mu0 = 2.5
        mu1 = -np.sqrt((mu0 ** 2 - d) / s)

        z = 2 * z * np.sign(self.shift[: len(z)]) + mu0
        z = self._rotate_point(z)

        return (
            np.min(
                [
                    np.sum((z - mu0) ** 2),
                    d * len(z) + s * np.sum((z - mu1) ** 2),
                ]
            )
            + 10 * np.sum(1 - np.cos(2 * np.pi * (z - mu0)))
            + self.bias
        )


# CEC2020
# realnum 2
class Modified_schwefel(Benchmark):
    def __init__(
        self,
        lower=-100,
        upper=100,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        super(Modified_schwefel, self).__init__(
            lower, upper, optimum, shift, rotate, shuffle, bias
        )

    def __call__(self, y):
        z = (
            self._rotate_point(10 * self._shift_point(y))
            + 4.209687462275036e002
        )
        if self.shuffle is not None:
            z = z[self.shuffle]
        g = np.zeros(len(z), dtype=float)
        zabs = np.abs(z)

        mask1 = zabs <= 500
        g[mask1] = z[mask1] * np.sin(np.sqrt(zabs[mask1]))

        mask2 = z > 500
        zmod = z[mask2] % 500
        g[mask2] = (500 - zmod) * np.sin(np.sqrt(np.abs(500 - zmod))) - (
            (z[mask2] - 500) ** 2
        ) / (10000 * len(z))

        mask3 = z < -500
        zabsmod = zabs[mask3] % 500
        g[mask3] = (zabsmod - 500) * np.sin(np.sqrt(np.abs(zabsmod - 500))) - (
            (z[mask3] + 500) ** 2
        ) / (10000 * len(z))

        return 418.9829 * len(z) - np.sum(g) + self.bias


class Expanded_schaffer(Benchmark):
    def __init__(
        self,
        lower=-100,
        upper=100,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        super(Expanded_schaffer, self).__init__(
            lower, upper, optimum, shift, rotate, shuffle, bias
        )

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        xpy = z[:-1] ** 2 + z[1:] ** 2
        xpy_last = z[-1] ** 2 + z[0] ** 2
        F = 0.5 + (np.sin(np.sqrt(xpy) - 0.5) ** 2) / (1 + 0.001 * xpy) ** 2
        last = (
            0.5
            + (np.sin(np.sqrt(xpy_last) - 0.5) ** 2)
            / (1 + 0.001 * xpy_last) ** 2
        )
        return np.sum(F) + last + self.bias


# CEC2020
# realnum 7
class Expanded_rosenbrock_griewangk(Benchmark):
    def __init__(
        self,
        lower=-100,
        upper=100,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        super(Expanded_rosenbrock_griewangk, self).__init__(
            lower, upper, optimum, shift, rotate, shuffle, bias
        )

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        tmp1 = z[:-1] ** 2 - z[1:]
        tmp2 = z[:-1] - 1.0
        temp = 100 * tmp1 ** 2 + tmp2 ** 2

        lasttmp1 = z[-1] ** 2 - z[0]
        lasttmp2 = z[-1] - 1.0
        lasttemp = 100 * lasttmp1 ** 2 + lasttmp2 ** 2
        return (
            np.sum(temp ** 2 / 4000 - np.cos(temp) + 1.0)
            + lasttemp ** 2 / 4000
            - np.cos(lasttemp)
            + 1.0
            + self.bias
        )


class Weierstrass(Benchmark):
    def __init__(
        self,
        a=0.5,
        b=3,
        kmax=20,
        lower=-5,
        upper=5,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        super(Weierstrass, self).__init__(
            lower, upper, optimum, shift, rotate, shuffle, bias
        )
        k = np.arange(0, kmax)
        self.ak = self.a ** k
        self.bk = self.b ** k

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        sum1 = np.sum(
            self.ak[:, np.newaxis]
            * np.cos(np.outer(2 * np.pi * self.bk, (z + 0.5)))
        )
        sum2 = len(z) * np.sum(self.ak * np.cos(2 * np.pi * self.bk * 0.5))
        return sum1 - sum2 + self.bias


##############
# GECCO 2022 #
##############

# hf01 case 11 realnum 4
class H1(Benchmark):
    def __init__(
        self,
        lower=-100,
        upper=100,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        self.g1 = Modified_schwefel()
        self.g2 = Rastrigin()
        self.g3 = High_conditioned_elliptic()
        super(H1, self).__init__(
            lower, upper, optimum, shift, rotate, shuffle, bias
        )
        self.p = np.cumsum([0.3, 0.3, 0.4])

    @Benchmark.shift.setter
    def shift(self, shift):
        self.g1.shift = shift
        self.g2.shift = shift
        self.g3.shift = shift
        self._shift = None

    @Benchmark.rotate.setter
    def rotate(self, rotate):
        self.g1.rotate = rotate
        self.g2.rotate = rotate
        self.g3.rotate = rotate
        self._rotate = None

    @Benchmark.shuffle.setter
    def shuffle(self, shuffle):
        self.g1.shuffle = None
        self.g2.shuffle = None
        self.g3.shuffle = None
        self._shuffle = shuffle

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        idx1, idx2, idx3 = np.ceil(self.p * len(z)).astype(int)
        z1, z2, z3 = z[0:idx1], z[idx1:idx2], z[idx2:idx3]

        if not isinstance(z1, np.ndarray):
            z1 = np.array([z1])
        if not isinstance(z2, np.ndarray):
            z2 = np.array([z2])
        if not isinstance(z3, np.ndarray):
            z3 = np.array([z3])
        return self.g1(z1) + self.g2(z2) + self.g3(z3) + self.bias


# hf06 case 16 realnum 16
class H2(Benchmark):
    def __init__(
        self,
        lower=-100,
        upper=100,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        self.g1 = Expanded_schaffer()
        self.g2 = HGBat(alpha=0.5)
        self.g3 = Rosenbrock()
        self.g4 = Modified_schwefel()
        super(H2, self).__init__(
            lower, upper, optimum, shift, rotate, shuffle, bias
        )
        self.p = np.cumsum([0.2, 0.2, 0.3, 0.3])

    @Benchmark.shift.setter
    def shift(self, shift):
        self.g1.shift = shift
        self.g2.shift = shift
        self.g3.shift = shift
        self.g4.shift = shift
        self._shift = None

    @Benchmark.rotate.setter
    def rotate(self, rotate):
        self.g1.rotate = rotate
        self.g2.rotate = rotate
        self.g3.rotate = rotate
        self.g4.rotate = rotate
        self._rotate = None

    @Benchmark.shuffle.setter
    def shuffle(self, shuffle):
        self.g1.shuffle = None
        self.g2.shuffle = None
        self.g3.shuffle = None
        self.g4.shuffle = None
        self._shuffle = shuffle

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        idx1, idx2, idx3, idx4 = np.ceil(self.p * len(z)).astype(int)
        z1, z2, z3, z4 = z[0:idx1], z[idx1:idx2], z[idx2:idx3], z[idx3:idx4]

        if not isinstance(z1, np.ndarray):
            z1 = np.array([z1])
        if not isinstance(z2, np.ndarray):
            z2 = np.array([z2])
        if not isinstance(z3, np.ndarray):
            z3 = np.array([z3])
        if not isinstance(z4, np.ndarray):
            z4 = np.array([z4])

        return self.g1(z1) + self.g2(z2) + self.g3(z3) + self.g4(z4) + self.bias


# hf05 case 15 realnum 6
class H3(Benchmark):
    def __init__(
        self,
        lower=-100,
        upper=100,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        self.g1 = Expanded_schaffer()
        self.g2 = HGBat(alpha=0.5)
        self.g3 = Rosenbrock()
        self.g4 = Modified_schwefel()
        self.g5 = High_conditioned_elliptic()
        super(H3, self).__init__(
            lower, upper, optimum, shift, rotate, shuffle, bias
        )
        self.p = np.cumsum([0.1, 0.2, 0.2, 0.2, 0.3])

    @Benchmark.shift.setter
    def shift(self, shift):
        self.g1.shift = shift
        self.g2.shift = shift
        self.g3.shift = shift
        self.g4.shift = shift
        self.g5.shift = shift
        self._shift = None

    @Benchmark.rotate.setter
    def rotate(self, rotate):
        self.g1.rotate = rotate
        self.g2.rotate = rotate
        self.g3.rotate = rotate
        self.g4.rotate = rotate
        self.g5.rotate = rotate
        self._rotate = None

    @Benchmark.shuffle.setter
    def shuffle(self, shuffle):
        self.g1.shuffle = None
        self.g2.shuffle = None
        self.g3.shuffle = None
        self.g4.shuffle = None
        self.g5.shuffle = None
        self._shuffle = shuffle

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        idx1, idx2, idx3, idx4, idx5 = np.ceil(self.p * len(z)).astype(int)
        z1, z2, z3, z4, z5 = (
            z[0:idx1],
            z[idx1:idx2],
            z[idx2:idx3],
            z[idx3:idx4],
            z[idx4:idx5],
        )
        if not isinstance(z1, np.ndarray):
            z1 = np.array([z1])
        if not isinstance(z2, np.ndarray):
            z2 = np.array([z2])
        if not isinstance(z3, np.ndarray):
            z3 = np.array([z3])
        if not isinstance(z4, np.ndarray):
            z4 = np.array([z4])
        if not isinstance(z5, np.ndarray):
            z5 = np.array([z5])

        return (
            self.g1(z1)
            + self.g2(z2)
            + self.g3(z3)
            + self.g4(z4)
            + self.g5(z5)
            + self.bias
        )


# cf02 case 22 realnum 22
class C1(Benchmark):
    def __init__(
        self,
        lower=-100,
        upper=100,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        self.g1 = Rastrigin(shift=None, rotate=None, shuffle=None)
        self.g2 = Griewank(shift=None, rotate=None, shuffle=None)
        self.g3 = Modified_schwefel(shift=None, rotate=None, shuffle=None)
        super(C1, self).__init__(
            lower, upper, optimum, shift, rotate, shuffle, bias
        )

    @Benchmark.shift.setter
    def shift(self, shift):
        if shift is None:
            shift = [None] * 3
        self.g1.shift = shift[0]
        self.g2.shift = shift[1]
        self.g3.shift = shift[2]
        self._shift = None

    @Benchmark.rotate.setter
    def rotate(self, rotate):
        if rotate is None:
            rotate = [None] * 3
        self.g1.rotate = rotate[0]
        self.g2.rotate = rotate[1]
        self.g3.rotate = rotate[2]
        self._rotate = None

    @Benchmark.shuffle.setter
    def shuffle(self, shuffle):
        if shuffle is None:
            shuffle = [None] * 3
        self.g1.shuffle = shuffle[0]
        self.g2.shuffle = shuffle[1]
        self.g3.shuffle = shuffle[2]
        self._shuffle = None

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]

        st1 = np.sum(self.g1.transform(z) ** 2)
        w1 = 1 / np.sqrt(st1) * np.exp(-st1 / (2 * len(z) * 10 ** 2))

        st2 = np.sum(self.g2.transform(z) ** 2)
        w2 = 1 / np.sqrt(st2) * np.exp(-st2 / (2 * len(z) * 20 ** 2))

        st3 = np.sum(self.g3.transform(z) ** 2)
        w3 = 1 / np.sqrt(st3) * np.exp(-st3 / (2 * len(z) * 30 ** 2))

        sw = w1 + w2 + w3
        if sw != 0:
            w1, w2, w3 = w1 / sw, w2 / sw, w3 / sw
        else:
            w1, w2, w3 = 1, 1, 1

        return (
            w1 * (1 * self.g1(z))
            + w2 * (10 * self.g2(z) + 100)
            + w3 * (1 * self.g3(z) + 200)
            + self.bias
        )


# cf04 case 24 realnum 24
class C2(Benchmark):
    def __init__(
        self,
        lower=-100,
        upper=100,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):
        self.g1 = Ackley(shift=None, rotate=None, shuffle=None)
        self.g2 = High_conditioned_elliptic(
            shift=None, rotate=None, shuffle=None
        )
        self.g3 = Griewank(shift=None, rotate=None, shuffle=None)
        self.g4 = Rastrigin(shift=None, rotate=None, shuffle=None)
        super(C2, self).__init__(lower, upper, optimum, None, None, None, bias)

    @Benchmark.shift.setter
    def shift(self, shift):
        if shift is None:
            shift = [None] * 4
        self.g1.shift = shift[0]
        self.g2.shift = shift[1]
        self.g3.shift = shift[2]
        self.g4.shift = shift[3]
        self._shift = None

    @Benchmark.rotate.setter
    def rotate(self, rotate):
        if rotate is None:
            rotate = [None] * 4
        self.g1.rotate = rotate[0]
        self.g2.rotate = rotate[1]
        self.g3.rotate = rotate[2]
        self.g4.rotate = rotate[3]
        self._rotate = None

    @Benchmark.shuffle.setter
    def shuffle(self, shuffle):
        if shuffle is None:
            shuffle = [None] * 4
        self.g1.shuffle = shuffle[0]
        self.g2.shuffle = shuffle[1]
        self.g3.shuffle = shuffle[2]
        self.g4.shuffle = shuffle[3]
        self._shuffle = None

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]

        st1 = np.sum(self.g1.transform(z) ** 2)
        w1 = 1 / np.sqrt(st1) * np.exp(-st1 / (2 * len(z) * 10 ** 2))

        st2 = np.sum(self.g2.transform(z) ** 2)
        w2 = 1 / np.sqrt(st2) * np.exp(-st2 / (2 * len(z) * 20 ** 2))

        st3 = np.sum(self.g3.transform(z) ** 2)
        w3 = 1 / np.sqrt(st3) * np.exp(-st3 / (2 * len(z) * 30 ** 2))

        st4 = np.sum(self.g4.transform(z) ** 2)
        w4 = 1 / np.sqrt(st4) * np.exp(-st4 / (2 * len(z) * 40 ** 2))

        sw = w1 + w2 + w3 + w4

        if sw != 0:
            w1, w2, w3, w4 = w1 / sw, w2 / sw, w3 / sw, w4 / sw
        else:
            w1, w2, w3, w4 = 1, 1, 1, 1

        return (
            w1 * (10 * self.g1(z))
            + w2 * (1e-6 * self.g2(z) + 100)
            + w3 * (10 * self.g3(z) + 200)
            + w4 * (1 * self.g4(z) + 300)
            + self.bias
        )


# cf05 case 25 realnum 25
class C3(Benchmark):
    def __init__(
        self,
        lower=-100,
        upper=100,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):

        self.g1 = Rastrigin(shift=None, rotate=None, shuffle=None)
        self.g2 = Happycat(alpha=0.25, shift=None, rotate=None, shuffle=None)
        self.g3 = Ackley(shift=None, rotate=None, shuffle=None)
        self.g4 = Discus(shift=None, rotate=None, shuffle=None)
        self.g5 = Rosenbrock()
        super(C3, self).__init__(lower, upper, optimum, None, None, None, bias)

    @Benchmark.shift.setter
    def shift(self, shift):
        if shift is None:
            shift = [None] * 5
        self.g1.shift = shift[0]
        self.g2.shift = shift[1]
        self.g3.shift = shift[2]
        self.g4.shift = shift[3]
        self.g5.shift = shift[4]
        self._shift = None

    @Benchmark.rotate.setter
    def rotate(self, rotate):
        if rotate is None:
            rotate = [None] * 5
        self.g1.rotate = rotate[0]
        self.g2.rotate = rotate[1]
        self.g3.rotate = rotate[2]
        self.g4.rotate = rotate[3]
        self.g5.rotate = rotate[4]
        self._rotate = None

    @Benchmark.shuffle.setter
    def shuffle(self, shuffle):
        if shuffle is None:
            shuffle = [None] * 5
        self.g1.shuffle = shuffle[0]
        self.g2.shuffle = shuffle[1]
        self.g3.shuffle = shuffle[2]
        self.g4.shuffle = shuffle[3]
        self.g5.shuffle = shuffle[4]
        self._shuffle = None

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]

        st1 = np.sum(self.g1.transform(z) ** 2)
        w1 = 1 / np.sqrt(st1) * np.exp(-st1 / (2 * len(z) * 10 ** 2))

        st2 = np.sum(self.g2.transform(z) ** 2)
        w2 = 1 / np.sqrt(st2) * np.exp(-st2 / (2 * len(z) * 20 ** 2))

        st3 = np.sum(self.g3.transform(z) ** 2)
        w3 = 1 / np.sqrt(st3) * np.exp(-st3 / (2 * len(z) * 30 ** 2))

        st4 = np.sum(self.g4.transform(z) ** 2)
        w4 = 1 / np.sqrt(st4) * np.exp(-st4 / (2 * len(z) * 40 ** 2))

        st5 = np.sum(self.g5.transform(z) ** 2)
        w5 = 1 / np.sqrt(st5) * np.exp(-st5 / (2 * len(z) * 50 ** 2))

        sw = w1 + w2 + w3 + w4 + w5
        if sw != 0:
            w1, w2, w3, w4, w5 = w1 / sw, w2 / sw, w3 / sw, w4 / sw, w5 / sw
        else:
            w1, w2, w3, w4 = 1, 1, 1, 1, 1

        return (
            w1 * (10 * self.g1(z))
            + w2 * (1 * self.g2(z) + 100)
            + w3 * (10 * self.g3(z) + 200)
            + w4 * (1e-6 * self.g4(z) + 300)
            + w5 * (1 * self.g5(z) + 400)
            + self.bias
        )


##############
# SOCCO 2011 #
##############


class CF9F1_25(Benchmark):
    def __init__(
        self,
        lower=-100,
        upper=100,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):

        self.g1 = F10(shift=None, rotate=None, shuffle=None)
        self.g2 = Sphere(shift=None, rotate=None, shuffle=None)
        super(CF9F1_25, self).__init__(
            lower, upper, optimum, None, None, None, bias
        )

        self.p = 0.25

    @Benchmark.shift.setter
    def shift(self, shift):
        if shift is None:
            shift = [None] * 2
        self.g1.shift = shift[0]
        self.g2.shift = shift[1]
        self._shift = None

    @Benchmark.rotate.setter
    def rotate(self, rotate):
        if rotate is None:
            rotate = [None] * 2
        self.g1.rotate = rotate[0]
        self.g2.rotate = rotate[1]
        self._rotate = None

    @Benchmark.shuffle.setter
    def shuffle(self, shuffle):
        if shuffle is None:
            shuffle = [None] * 2
        self.g1.shuffle = shuffle[0]
        self.g2.shuffle = shuffle[1]
        self._shuffle = None

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        idx1 = np.ceil(self.p * len(z)).astype(int)
        z1, z2 = z[0:idx1], z[idx1:]
        return self.g1(z1) + self.g2(z2) + self.bias


class CF9F3_25(Benchmark):
    def __init__(
        self,
        lower=-100,
        upper=100,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):

        self.g1 = F10(shift=None, rotate=None, shuffle=None)
        self.g2 = Rosenbrock(shift=None, rotate=None, shuffle=None)
        super(CF9F3_25, self).__init__(
            lower, upper, optimum, None, None, None, bias
        )

        self.p = 0.25

    @Benchmark.shift.setter
    def shift(self, shift):
        if shift is None:
            shift = [None] * 2
        self.g1.shift = shift[0]
        self.g2.shift = shift[1]
        self._shift = None

    @Benchmark.rotate.setter
    def rotate(self, rotate):
        if rotate is None:
            rotate = [None] * 2
        self.g1.rotate = rotate[0]
        self.g2.rotate = rotate[1]
        self._rotate = None

    @Benchmark.shuffle.setter
    def shuffle(self, shuffle):
        if shuffle is None:
            shuffle = [None] * 2
        self.g1.shuffle = shuffle[0]
        self.g2.shuffle = shuffle[1]
        self._shuffle = None

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        idx1 = np.ceil(self.p * len(z)).astype(int)
        z1, z2 = z[0:idx1], z[idx1:]
        return self.g1(z1) + self.g2(z2) + self.bias


class CF9F4_25(Benchmark):
    def __init__(
        self,
        lower=-100,
        upper=100,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):

        self.g1 = F10(shift=None, rotate=None, shuffle=None)
        self.g2 = Rastrigin(shift=None, rotate=None, shuffle=None)
        super(CF9F4_25, self).__init__(
            lower, upper, optimum, None, None, None, bias
        )

        self.p = 0.25

    @Benchmark.shift.setter
    def shift(self, shift):
        if shift is None:
            shift = [None] * 2
        self.g1.shift = shift[0]
        self.g2.shift = shift[1]
        self._shift = None

    @Benchmark.rotate.setter
    def rotate(self, rotate):
        if rotate is None:
            rotate = [None] * 2
        self.g1.rotate = rotate[0]
        self.g2.rotate = rotate[1]
        self._rotate = None

    @Benchmark.shuffle.setter
    def shuffle(self, shuffle):
        if shuffle is None:
            shuffle = [None] * 2
        self.g1.shuffle = shuffle[0]
        self.g2.shuffle = shuffle[1]
        self._shuffle = None

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        idx1 = np.ceil(self.p * len(z)).astype(int)
        z1, z2 = z[0:idx1], z[idx1:]
        return self.g1(z1) + self.g2(z2) + self.bias


class CF10F7_25(Benchmark):
    def __init__(
        self,
        lower=-100,
        upper=100,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):

        self.g1 = Bohachevsky(shift=None, rotate=None, shuffle=None)
        self.g2 = Schwefel_problem_2_22(shift=None, rotate=None, shuffle=None)
        super(CF10F7_25, self).__init__(
            lower, upper, optimum, None, None, None, bias
        )

        self.p = 0.25

    @Benchmark.shift.setter
    def shift(self, shift):
        if shift is None:
            shift = [None] * 2
        self.g1.shift = shift[0]
        self.g2.shift = shift[1]
        self._shift = None

    @Benchmark.rotate.setter
    def rotate(self, rotate):
        if rotate is None:
            rotate = [None] * 2
        self.g1.rotate = rotate[0]
        self.g2.rotate = rotate[1]
        self._rotate = None

    @Benchmark.shuffle.setter
    def shuffle(self, shuffle):
        if shuffle is None:
            shuffle = [None] * 2
        self.g1.shuffle = shuffle[0]
        self.g2.shuffle = shuffle[1]
        self._shuffle = None

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        idx1 = np.ceil(self.p * len(z)).astype(int)
        z1, z2 = z[0:idx1], z[idx1:]
        return self.g1(z1) + self.g2(z2) + self.bias


class CF9F1_75(Benchmark):
    def __init__(
        self,
        lower=-100,
        upper=100,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):

        self.g1 = F10(shift=None, rotate=None, shuffle=None)
        self.g2 = Sphere(shift=None, rotate=None, shuffle=None)
        super(CF9F1_75, self).__init__(
            lower, upper, optimum, None, None, None, bias
        )

        self.p = 0.75

    @Benchmark.shift.setter
    def shift(self, shift):
        if shift is None:
            shift = [None] * 2
        self.g1.shift = shift[0]
        self.g2.shift = shift[1]
        self._shift = None

    @Benchmark.rotate.setter
    def rotate(self, rotate):
        if rotate is None:
            rotate = [None] * 2
        self.g1.rotate = rotate[0]
        self.g2.rotate = rotate[1]
        self._rotate = None

    @Benchmark.shuffle.setter
    def shuffle(self, shuffle):
        if shuffle is None:
            shuffle = [None] * 2
        self.g1.shuffle = shuffle[0]
        self.g2.shuffle = shuffle[1]
        self._shuffle = None

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        idx1 = np.ceil(self.p * len(z)).astype(int)
        z1, z2 = z[0:idx1], z[idx1:]
        return self.g1(z1) + self.g2(z2) + self.bias


class CF9F3_75(Benchmark):
    def __init__(
        self,
        lower=-100,
        upper=100,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):

        self.g1 = F10(shift=None, rotate=None, shuffle=None)
        self.g2 = Rosenbrock(shift=None, rotate=None, shuffle=None)
        super(CF9F3_75, self).__init__(
            lower, upper, optimum, None, None, None, bias
        )

        self.p = 0.75

    @Benchmark.shift.setter
    def shift(self, shift):
        if shift is None:
            shift = [None] * 2
        self.g1.shift = shift[0]
        self.g2.shift = shift[1]
        self._shift = None

    @Benchmark.rotate.setter
    def rotate(self, rotate):
        if rotate is None:
            rotate = [None] * 2
        self.g1.rotate = rotate[0]
        self.g2.rotate = rotate[1]
        self._rotate = None

    @Benchmark.shuffle.setter
    def shuffle(self, shuffle):
        if shuffle is None:
            shuffle = [None] * 2
        self.g1.shuffle = shuffle[0]
        self.g2.shuffle = shuffle[1]
        self._shuffle = None

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        idx1 = np.ceil(self.p * len(z)).astype(int)
        z1, z2 = z[0:idx1], z[idx1:]
        return self.g1(z1) + self.g2(z2) + self.bias


class CF9F4_75(Benchmark):
    def __init__(
        self,
        lower=-100,
        upper=100,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):

        self.g1 = F10(shift=None, rotate=None, shuffle=None)
        self.g2 = Rastrigin(shift=None, rotate=None, shuffle=None)
        super(CF9F4_75, self).__init__(
            lower, upper, optimum, None, None, None, bias
        )

        self.p = 0.75

    @Benchmark.shift.setter
    def shift(self, shift):
        if shift is None:
            shift = [None] * 2
        self.g1.shift = shift[0]
        self.g2.shift = shift[1]
        self._shift = None

    @Benchmark.rotate.setter
    def rotate(self, rotate):
        if rotate is None:
            rotate = [None] * 2
        self.g1.rotate = rotate[0]
        self.g2.rotate = rotate[1]
        self._rotate = None

    @Benchmark.shuffle.setter
    def shuffle(self, shuffle):
        if shuffle is None:
            shuffle = [None] * 2
        self.g1.shuffle = shuffle[0]
        self.g2.shuffle = shuffle[1]
        self._shuffle = None

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        idx1 = np.ceil(self.p * len(z)).astype(int)
        z1, z2 = z[0:idx1], z[idx1:]
        return self.g1(z1) + self.g2(z2) + self.bias


class CF10F7_75(Benchmark):
    def __init__(
        self,
        lower=-100,
        upper=100,
        optimum=0,
        shift=None,
        rotate=None,
        shuffle=None,
        bias=0,
    ):

        self.g1 = Bohachevsky(shift=None, rotate=None, shuffle=None)
        self.g2 = Schwefel_problem_2_22(shift=None, rotate=None, shuffle=None)
        super(CF10F7_75, self).__init__(
            lower, upper, optimum, None, None, None, bias
        )

        self.p = 0.75

    @Benchmark.shift.setter
    def shift(self, shift):
        if shift is None:
            shift = [None] * 2
        self.g1.shift = shift[0]
        self.g2.shift = shift[1]
        self._shift = None

    @Benchmark.rotate.setter
    def rotate(self, rotate):
        if rotate is None:
            rotate = [None] * 2
        self.g1.rotate = rotate[0]
        self.g2.rotate = rotate[1]
        self._rotate = None

    @Benchmark.shuffle.setter
    def shuffle(self, shuffle):
        if shuffle is None:
            shuffle = [None] * 2
        self.g1.shuffle = shuffle[0]
        self.g2.shuffle = shuffle[1]
        self._shuffle = None

    def __call__(self, y):
        z = self.transform(y)
        if self.shuffle is not None:
            z = z[self.shuffle]
        idx1 = np.ceil(self.p * len(z)).astype(int)
        z1, z2 = z[0:idx1], z[idx1:]
        return self.g1(z1) + self.g2(z2) + self.bias
