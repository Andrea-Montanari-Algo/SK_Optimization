import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits import mplot3d
import scipy.integrate as int
import sys
import time

FAC_XMAX = 8.0
NX0 = 1000


def uf(iq, xm, dph, xmax, nx, l, r):
    u = np.zeros(nx)
    dx = float(xmax) / float(nx)
    pl = r * (xm >= xmax)
    mi = l * (xm <= -xmax)
    cent = np.logical_and((xm > -xmax), (xm < xmax))
    i1 = ((xm / dx).astype('int')) + nx
    i1 = np.minimum(i1, 2 * nx - 1)
    i1 = np.maximum(i1, 1)
    s = (np.sign(xm)).astype('int')
    zs = (s == 0).astype('int')
    s = s + zs
    i2 = i1 + s
    i2 = np.minimum(i2, 2 * nx)
    i2 = np.maximum(i2, 0)
    u = (-(xm - (i2 - nx) * dx) * dph[iq, i1] + (xm - (i1 - nx) * dx) * dph[iq, i2]) / (dx * s)
    u = cent * u + pl + mi
    return u


def amp(A, mu, ph, dph, ddph, xmax, nx):
    tol = 0.000000001
    n = A.shape[0]
    [q, m] = mu
    nq = len(q)
    dq = q[1] - q[0]
    sdq = np.sqrt(dq)
    sn = np.sqrt(float(n))
    b = 0.
    u = np.zeros((n, nq + 2))
    g = np.zeros((n, nq + 2))
    x = np.zeros((n, nq + 2))
    u[:, 2] = np.random.normal(0, 1, n)
    g[:, 1] = np.ones(n)
    z = np.zeros((n, nq))
    for iq in range(nq - 1):
        sigma = np.sign(z[:, iq])
        energy0 = np.dot(sigma, np.dot(A, sigma)) / (2. * n)
        b = np.sum(g[:, iq + 1]) / n
        u[:, iq + 3] = np.dot(A, g[:, iq + 1] * u[:, iq + 2]) - b * g[:, iq] * u[:, iq + 1]
        u[:, iq + 3] = u[:, iq + 3] - np.dot(u[:, iq + 3], u[:, iq + 2]) * u[:, iq + 2] / n - np.dot(u[:, iq + 3],
                                                                                                     u[:, iq + 1]) * u[
                                                                                                                     :,
                                                                                                                     iq + 1] / n
        u[:, iq + 3] = sn * u[:, iq + 3] / np.linalg.norm(u[:, iq + 3])
        #     print(iq,', norm2(u) = ',np.linalg.norm(u[:,iq+3])**2/n,', norm2(z) = ',np.linalg.norm(z[:,iq])**2/n,', energy = ',energy0)
        x[:, iq + 2] = x[:, iq + 1] + m[iq] * uf(iq, x[:, iq + 1], dph, xmax, nx, -1, 1) * dq + sdq * u[:, iq + 2]
        v = uf(iq, x[:, iq + 2], ddph, xmax, nx, 0, 0)
        normv = np.linalg.norm(v)
        g[:, iq + 2] = sn * v / (normv + tol)
        z[:, iq + 1] = z[:, iq] + u[:, iq + 2] * g[:, iq + 1] * sdq
    return x, z


def rmatrix(n, lam):
    av = lam / (2. * n)
    sd = 1 / np.sqrt(2. * n)
    Z = np.random.normal(av, sd, n * n)
    Z = np.reshape(Z, (n, n))
    Z = Z + np.conj(Z.T)
    return Z


def read_mu(filename):
    f = open(filename, "r")
    qv = []
    mv = []
    for line in f:
        if (line[0] != "#"):
            [q, m] = np.fromstring(line, dtype=float, sep=' ')
            qv.append(q)
            mv.append(m)
            print(q,m)
    qv = np.array(qv)
    mv = np.array(mv)
    return [qv, mv]


def interpolate_mu_old(mu, ns):
    [qv, mv] = mu
    nq = np.size(qv)
    qs = 0.
    ms = 0.
    qq = np.zeros(nq * ns)
    mm = np.zeros(nq * ns)
    dq = qv[1] - qv[0]
    for i in range(nq):
        for j in range(ns):
            ii = i * ns + j
            qq[ii] = qs
            mm[ii] = ms
            qs = qs + dq / float(ns)
            ms = ms + mv[i] / float(ns)
    return [qq, mm]


def interpolate_mu(mu, ns):
    [qv, mv] = mu
    nq = np.size(qv)
    qs = 0.
    ms = 0.
    qq = np.zeros(nq * ns)
    mm = np.zeros(nq * ns)
    dq = qv[1] - qv[0]
    for i in range(nq):
        ms = ms + mv[i]
        for j in range(ns):
            ii = i * ns + j
            qq[ii] = qs
            mm[ii] = ms
            qs = qs + dq / float(ns)
    return [qq, mm]


def initKernel(dt, x, x0):
    nnx = np.size(x)
    nnx0 = np.size(x0)
    G = np.zeros((nnx, nnx0))
    for i in range(0, nnx):
        for j in range(0, nnx0):
            G[i, j] = np.exp(-(x[i] - x0[j]) ** 2 / (2 * dt))
    s = np.dot(G, np.ones(nnx0))
    G = np.dot(np.diag(1. / s), G)
    return G


def phi(mu, xmax, nx):
    mtol = 0.00001
    x = np.linspace(-xmax, xmax, 2 * nx + 1)
    dx = xmax / nx
    [qv, ms] = mu
    nq = np.size(qv)
    phi = np.ones((nq, 2 * nx + 1))
    u = np.zeros((nq, 2 * nx + 1))  # dphi/dx
    w = np.zeros((nq, 2 * nx + 1))  # d^2phi/dx^2
    xabs = np.abs(x)
    phi[nq - 1, :] = xabs
    u[nq - 1, :] = np.sign(x)
    w[nq - 1, nx] = 2. / dx

    dq = qv[1] - qv[0]
    sdev = 6 * np.sqrt(dq)
    nx0 = nx + np.int(sdev / (float(xmax) / float(nx)))
    x0max = nx0 * (xmax / nx)
    x0 = np.linspace(-x0max, x0max, 2 * nx0 + 1)

    phi0 = np.zeros(2 * nx0 + 1)
    u0 = np.sign(x0)
    w0 = np.zeros(2 * nx0 + 1)
    z0 = np.ones(2 * nx0 + 1)
    toadd = np.zeros(2 * nx0 + 1)
    toadd[0:nx0 - nx] = np.abs(x0[0:nx0 - nx] - x0[nx0 - nx])
    toadd[nx0 + nx + 1:2 * nx0 + 1] = np.abs(x0[nx0 + nx + 1:2 * nx0 + 1] - x0[nx0 + nx])
    G = initKernel(dq, x, x0)

    for iq in range(nq - 2, -1, -1):
        m = ms[iq]
        phi0 = 0. * phi0
        phi0[nx0 - nx:nx0 + nx + 1] = phi[iq + 1, :] - phi[iq + 1, 0]
        phi0 = phi0 + toadd + phi[iq + 1, 0]
        u0[nx0 - nx:nx0 + nx + 1] = u[iq + 1, :]
        w0[nx0 - nx:nx0 + nx + 1] = w[iq + 1, :]
        z0 = np.exp(m * phi0)
        z1 = np.dot(G, z0)
        print(iq,m,np.sum(G))
        if (m > mtol):
            phi[iq, :] = (1. / m) * np.log(z1)
            u[iq, :] = np.dot(G, u0 * z0) / z1
            w[iq, :] = np.dot(G, w0 * z0) / z1 + m * (np.dot(G, u0 * u0 * z0) / z1) - m * u[iq, :] ** 2
        else:
            phi[iq, :] = np.dot(G, phi0)
            u[iq, :] = np.dot(G, u0)
            w[iq, :] = np.dot(G, w0)
    return phi, u, w


def energy(A, z):
    s = np.sign(z)
    n = A.shape[0]
    return (np.dot(s, np.dot(A, s)) / (2. * n))


def energy_stats(nint, n, nsample, input_file, output_file):
    [qv, mv] = read_mu(input_file)
    [q, m] = interpolate_mu([qv, mv], nint)
    nq = len(q)
    qea = q[nq - 1]
    xmax = FAC_XMAX
    nx = np.array(NX0)
    ph, dph, ddph = phi([q, m], xmax, nx)
    en = np.zeros(nsample)
    for js in range(nsample):
        A = rmatrix(n, 0.)
        [x, z] = amp(A, [q, m], ph, dph, ddph, xmax, nx)
        en[js] = energy(A, z[:, nq - 1])
        print(js, en[js], np.mean(en[0:js + 1]))

    emean = np.mean(en)
    ese = np.std(en) / np.sqrt(nsample)
    f = open(output_file, "w+")
    f.write('#\n')
    f.write('# n = {:d}, nq = {:d}, nint = {:d}\n'.format(n, nq, nint))
    f.write('# xmax = {:4.3f}, nx0 = {:d}\n'.format(FAC_XMAX, NX0))
    f.write('# inputfile = ' + input_file + "\n")
    f.write('#\n')
    f.write('# E = {:11.8f},  STD_ERR = {:11.8f}  \n'.format(emean, ese))
    f.write('#\n')
    f.write('#        q                mu(q)       \n')
    f.write('#\n')
    np.savetxt(f, en, fmt='%14.10f', delimiter='        ', newline='\n')
    f.close()


def plotXZ(nint, n, filename, ns):
    [qv, mv] = read_mu(filename)
    [q, m] = interpolate_mu([qv, mv], nint)
    nq = len(q)
    qea = q[nq - 1]
    xmax = FAC_XMAX
    nx = np.array(NX0)
    ph, dph, ddph = phi([q, m], xmax, nx)
    A = rmatrix(n, 0.)
    [x, z] = amp(A, [q, m], ph, dph, ddph, xmax, nx)
    plt.ion()
    plt.xlabel(r'$t$', fontsize=18)
    plt.ylabel(r'$z_{i,t}$', fontsize=18)
    plt.axis([0, 1, -1.1, 1.1])
    plt.plot([0, 1], [1, 1], '--', color="black")
    plt.plot([0, 1], [-1, -1], '--', color="black")
    plt.plot([0, 1], [0, 0], '--', color="black")

    zmax = 0.
    for js in range(ns):
        zzmax = np.max(np.abs(z[js, :]))
        zmax = np.max((zmax, zzmax))

    for js in range(ns):
        plt.plot(q, z[js, :] / zmax, "-", color="red", linewidth=2)
    plt.savefig('sample_n1E4_traj_z.pdf')
    plt.close()

    plt.ion()
    plt.xlabel(r'$t$', fontsize=18)
    plt.ylabel(r'$x_{i,t}$', fontsize=18)
    plt.axis([0, 1, -3, 3])
    # plt.plot([0,1],[1,1],'--',color="black")
    # plt.plot([0,1],[-1,-1],'--',color="black")
    plt.plot([0, 1], [0, 0], '--', color="black")
    for js in range(ns):
        plt.plot(q, x[js, 0:nq], "-", color="blue", linewidth=2)
    plt.savefig('sample_n1E4_traj_x.pdf')
    plt.close()

    print('---->', energy(A, z[:, nq - 1]))


def histX(nint, n, qt, input_file, output_file):
    [qv, mv] = read_mu(input_file)
    [q, m] = interpolate_mu([qv, mv], nint)
    nq = len(q)
    qea = q[nq - 1]
    xmax = FAC_XMAX
    nx = np.array(NX0)
    ph, dph, ddph = phi([q, m], xmax, nx)
    A = rmatrix(n, 0.)
    [x, z] = amp(A, [q, m], ph, dph, ddph, xmax, nx)

    dq = q[1] - q[0]
    iq = np.int(qt / dq)
    if (iq > nq - 1):
        xp = x[:, nq - 1]
    else:
        xp = x[:, iq]

    plt.hist(xp)
    f = open(output_file, "w+")
    f.write('#\n')
    f.write('# n = {:d}, nq = {:d}, nint = {:d}\n'.format(n, nq, nint))
    f.write('# xmax = {:4.3f}, nx0 = {:d}\n'.format(FAC_XMAX, NX0))
    f.write('# inputfile = ' + input_file + "\n")
    f.write('#\n')
    f.write('# E = {:11.8f}  \n'.format(energy(A, z[:, nq - 1])))
    f.write('#\n')
    f.write('#        x      \n')
    f.write('#\n')
    np.savetxt(f, xp, fmt='%14.10f', delimiter='        ', newline='\n')
    f.close()


def testFree(nint, n, filename):
    [qv, mv] = read_mu(filename)
    [q, m] = interpolate_mu([qv, mv], nint)
    nq = len(q)
    qea = q[nq - 1]
    xmax = FAC_XMAX
    nx = np.array(NX0)
    ph, dph, ddph = phi([q, m], xmax, nx)
    A = rmatrix(n, 0.)
    [x, z] = amp(A, [q, m], ph, dph, ddph, xmax, nx)
    xx = x[:, nq - 2]
    print(energy(A, z))


def test():
    [beta, qv, mv] = read_mu("md_beta4_nq40_it20000.txt")
    [q, m] = interpolate_mu([qv, mv], 100)
    xmax = FAC_XMAX * beta
    nx = np.array(NX0)
    ph, dph, ddph = phi(beta, [q, m], xmax, nx)

    x = np.linspace(-xmax, xmax, 2 * nx + 1)
    Q, X = np.meshgrid(q, x)
    Q = np.transpose(Q)
    X = np.transpose(X)

    nq = len(q)
    plt.plot(x, ph[0, :], "-", color="red", linewidth=2)
    plt.plot(x, ph[nq - 1, :], "--", color="red", linewidth=2)

    plt.plot(x, dph[0, :], "-", color="blue", linewidth=2)
    plt.plot(x, dph[nq - 1, :], "--", color="blue", linewidth=2)

    plt.plot(x, ddph[0, :], "-", color="green", linewidth=2)
    plt.plot(x, ddph[nq - 1, :], "--", color="green", linewidth=2)

