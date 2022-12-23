# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-03T22:37:34+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import numpy as np
from abc import abstractmethod
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

import logging

logger = logging.getLogger("zellij.spokedart")


def randomMuller(n, m, surface=False):
    u = np.random.normal(0, 1, (n, m))
    d = np.linalg.norm(u, axis=1, keepdims=True)
    p = u / d

    if surface:
        return p
    else:
        return np.random.random(n) ** (1 / m) * p


def interSegmentCircle(A, B, C, R):

    dist = np.linalg.norm(B - A)

    BmA = B - A
    CmA = C - A

    a = np.sum(BmA ** 2)
    b = -2 * np.sum(BmA * CmA)
    c = np.sum(CmA ** 2) - R ** 2

    delta = b ** 2 - 4 * a * c

    intersect = np.empty((0, len(C)))

    if delta < 0:
        p1, p2 = (None, None)
    elif delta > 0:
        p1, p2 = (
            A + ((-b + np.sqrt(delta)) / (2 * a)) * BmA,
            A + ((-b - np.sqrt(delta)) / (2 * a)) * BmA,
        )

        Ap1, p1B = np.linalg.norm(p1 - A), np.linalg.norm(B - p1)
        Ap2, p2B = np.linalg.norm(p2 - A), np.linalg.norm(B - p2)
        if dist == Ap1 + p1B:
            intersect = np.append(intersect, [p1], axis=0)
        if dist == Ap2 + p2B:
            intersect = np.append(intersect, [p2], axis=0)

        # intersect = np.append(intersect, [p1, p2], axis=0)

    else:
        p1 = A + (-b / (2 * a)) * BmA
        dot = np.dot(BmA, BmA)
        dotc1 = np.dot(BmA, p1 - A)
        if dotc1 >= 0 and dotc1 <= dot:
            intersect = np.append(intersect, [p1], axis=0)

    return intersect


def interLineHyperplane(A, B, X, Y):
    """interLineHyperplane(A, B, X, Y)

    Compute the intersection between line AB and the plane separating X and Y.

    Parameters
    ----------
    A : type
        First point from line
    B : type
        Second point from line
    X : type
        First point under plane
    Y : type
        Second point above plane

    Returns
    -------
    bool
        if true the intersection is found, else no
    np.ndarray
        point intersection

    """

    if np.all(X == Y):
        return False, -1
    else:
        # line vector
        BmA = B - A

        # Normal vector for bisecting plane (plane at the middle of X and Y)
        YmX = Y - X
        v = YmX / np.linalg.norm(YmX)
        xmid = (X + Y) / 2

        if np.dot(BmA, v) == 0:
            return False, -1
        else:
            d = np.dot(xmid - A, v) / np.dot(BmA, v)
            intersection = A + BmA * d

            # C = A + (B-A)*t, use parametric and symetric equation of a line: (c0-a0)/(b0-a0) = (c1-a1)/(b1-a1)... = t, be caraful when (bi -ai) = 0, t doesn't exists for dim i.
            maxidx = np.argmax(
                BmA != 0
            )  # argmax stops at the first occurence of True. At least one value from BmA must be != 0, otherwise, A==B which is not possible.
            t = (intersection[maxidx] - A[maxidx]) / BmA[maxidx]

            # not on the half line
            if t < 0:
                return False, -1
            else:
                return True, intersection


class HalfLine(object):
    def __init__(self, A, B):
        self.A = A
        self.B = B

        self.v = B - A

    def point(self, d):
        return self.A + self.v * d


class Hyperplane(object):
    def __init__(self, X, Y):

        self.cellX = X
        self.cellY = Y

        self.X = X.seed
        self.Y = Y.seed

        assert not np.all(self.X == self.Y), logger.error(
            f"Can't build a bisecting plane between two identical points, got {X.seed} and {Y.seed}"
        )

        YmX = self.Y - self.X

        self.xmid = (self.X + self.Y) / 2
        self.v = YmX / np.linalg.norm(YmX)

    def intersection(self, line):
        # line is on the plane /!\ impossible in our case
        # if np.dot(line.v, self.v) == 0:
        #     return False, -1
        # else:

        d = np.dot(self.xmid - line.A, self.v) / np.dot(line.v, self.v)

        # not on the half line
        if d < 0:
            return False, -1
        else:
            return True, line.point(d)


class SpokeDart(object):
    def __init__(self, dim, seeds, r, reject=12):

        ##############
        # PARAMETERS #
        ##############

        self.seeds = seeds
        self.radius = r
        self.m = reject
        self.dim = dim

    def run(self, max):

        s = np.random.random(self.dim)

        S = np.empty((0, self.dim))
        P = np.array([s])
        N = []

        while len(P) > 0 and len(S) < max:

            s = P[0]
            P = P[1:]

            if len(S) > 0:
                N = list(self.collect_neighbors(s, S))

            reject = 0
            while reject < self.m and len(S) < max:
                a, b = self.randomSpoke(s)

                if len(N) > 0:
                    covered, a, b = self.trimSpoke(a, b, S[N])
                else:
                    covered = False

                if covered:
                    reject += 1
                else:
                    u = b - a

                    ulength = np.linalg.norm(u)
                    if ulength > 0:
                        rlength = np.random.uniform(0, ulength)
                        v = u * rlength / ulength

                        sp = a + v

                    else:
                        sp = a

                    if np.any(sp > np.array([1] * self.dim)) or np.any(
                        sp < np.array([0] * self.dim)
                    ):
                        reject += 1
                    else:
                        S = np.append(S, [sp[:]], axis=0)
                        N.append(len(S) - 1)
                        P = np.append(P, [sp[:]], axis=0)

                        reject = 0

        return S

    def collect_neighbors(self, s, samples):
        dist = cdist([s], samples)[0]
        mask = dist < 3 * self.radius
        return mask.nonzero()[0]

    def randomSpoke(self, s):
        p = randomMuller(1, self.dim, surface=True)[0]
        pclose = p * self.radius + s
        pfar = p * 2 * self.radius + s
        return pclose, pfar

    def trimSpoke(self, a, b, neighbors):
        trims = np.empty((0, self.dim))
        covered = False
        i = 0
        while i < len(neighbors) and not covered:
            n = neighbors[i]
            i += 1

            if np.linalg.norm(n - a) <= self.radius:
                covered = True
            else:
                res = interSegmentCircle(a, b, n, self.radius)
                trims = np.append(trims, res, axis=0)

        if covered:
            return covered, -1, -1
        elif len(trims) > 0:
            dist = np.linalg.norm(trims - a, axis=1)
            minidx = np.argmin(dist)
            return covered, a, trims[minidx]
        else:
            return covered, a, b
