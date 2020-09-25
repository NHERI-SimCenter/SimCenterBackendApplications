from cmath import pi, exp, sqrt
import numpy as np
import sys

class gauss1D:

    def __init__(self, Ly, Ny, sigma = 1.0, d = 1.0):
        # overall length in x-direction
        self.Lx = 1
        # overall length in y-direction
        self.Ly = Ly
        # Number of wave number increments in x-direction
        self.Nx = 1
        # Number of wave number increments in y-direction
        self.Ny = Ny
        # Standard deviation for spectral density function
        self.sigma = sigma
        # correlation decay coefficient
        self.d = d

        # wave number increment in x and y direction
        self.dkx = 2 * pi / self.Lx
        self.dky = 2 * pi / self.Ly
        # number of position increments in x and y direction
        self.Mx = 1
        self.My = int(2 * Ny)
        # position increment in x and y direction
        self.dx = self.Lx / self.Mx
        self.dy = self.Ly / self.My
        # upper limit to wave number in x and y direction
        self.kxu = self.Nx * self.dkx
        self.kyu = self.Ny * self.dky

    def calculate(self):
        # matrix of random phase angles
        phi = 2 * pi * np.random.rand(self.Mx, self.My)
        psi = 2 * pi * np.random.rand(self.Mx, self.My)
        self.f = np.zeros([self.Mx, self.My])
        f1 = f3 = np.zeros(self.Mx, dtype=complex)
        f2 = f4 = np.zeros(self.My, dtype=complex)
        part1 = part2 = np.zeros(self.Mx)

        for pp in range(0, self.Mx):
            xp = pp * self.dx
            for qq in range(0, self.My):
                yq = qq * self.dy
                for kk in range(0, self.Mx):
                    kxk = kk * self.dkx
                    f1[kk] = exp(1j * kxk * xp)
                    for ll in range(0, self.My):
                        kyl = ll * self.dky
                        kappa = sqrt(kxk ** 2 + kyl ** 2)
                        Sgg = self.sigma ** 2 * self.d ** 2 * exp(-self.d ** 2 *
                        abs(kappa) ** 2 / 4.0) / 4.0 / pi
                        Akl = sqrt(2 * Sgg * self.dkx * self.dky)
                        f2[ll] = Akl * exp(1j * phi[kk, ll]) * exp(1j * kyl * yq)
                    f2sum = np.sum(f2)
                    part1[kk] = np.real(sqrt(2) * np.sum(f2sum * f1[kk]))

                for kk in range(0, self.Mx):
                    kxk = kk * self.dkx
                    f3[kk] = exp(1j * kxk * xp)
                    for ll in range(0, self.My):
                        kyl = ll * self.dky
                        kappa = sqrt(kxk ** 2 + kyl ** 2)
                        Sgg = self.sigma ** 2 * self.d ** 2 * exp(-self.d ** 2 *
                        abs(kappa) ** 2 / 4.0) / 4.0 / pi
                        Akl = sqrt(2 * Sgg * self.dkx * self.dky)
                        f4[ll] = Akl * exp(1j * psi[kk, ll]) * exp(-1j * kyl * yq)
                    f4sum = np.sum(f4)
                    part2[kk] = np.real(sqrt(2) * np.sum(f4sum * f3[kk]))

                self.f[pp, qq] = part1.sum() + part2.sum()


def printField(self):
    print(self.f)

if __name__ == "__main__":
    Ly = 6.0
    Ny = 6.0
    sigma = 1.0
    d = 1.0
    a = gauss1D(Ly, Ny, sigma, d)
    a.calculate()
    F = a.f.reshape((-1,1))
    Y = np.linspace(0, a.Ly, a.My)




