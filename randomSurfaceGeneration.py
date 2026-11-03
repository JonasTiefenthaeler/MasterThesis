import numpy as np
import vtk
import os
from scipy.special import gamma
import pickle
import pyvista as pv
from functions import *



## NOTE: The function fractal_surf_generator is basically the Matlab code from the paper https://doi.org/10.1016/j.triboint.2018.11.020 ported to python
def fractal_surf_generator(PS, Zh, M, N, max_it, min_speed, min_err):

    PS = PS.copy()

    PS[0, 0] = 0.0
    PS = PS / np.sqrt(np.sum(PS**2)) * (M*N)

   
    Zh_mean = Zh.mean()
    Zh = Zh - Zh_mean
    Sq = np.sqrt(np.mean(Zh**2))
    Zh = Zh/Sq

    Zh_order = np.sort(Zh.ravel())

    n0, bin_edges = np.histogram(Zh_order, bins='scott', density=True)
    #print(n0)
   
    err = {
        "H":  [1.0],
        "PS": [1.0],
        "PS0":[1.0]
    }

    Zh = Zh.reshape(M, N)
    zh = np.fft.fft2(Zh)

    while True:

        phase = np.angle(zh)
        Zs = np.real(np.fft.ifft2(PS * np.exp(1j * phase)))

        n_hist, _ = np.histogram(Zs.ravel(), bins = bin_edges, density = True)
        bin_sizes = np.diff(bin_edges)
        err_H = np.sum(np.abs(n_hist - n0) * bin_sizes)
        err["H"].append(err_H)

        Zh_flat = Zs.ravel()
        ind = np.argsort(Zh_flat)
        Zh_flat[ind] = Zh_order
        Zh = Zh_flat.reshape(M, N)
        zh = np.fft.fft2(Zh)

        # Power spectrum error
        Zh_abs = np.abs(zh)
        mask_pos = PS > 0
        mask_zero = PS == 0

        err_PS = np.sqrt(
            np.mean((1 - Zh_abs[mask_pos] / PS[mask_pos])**2)
        )
        err_PS0 = (
            np.sqrt(np.mean(Zh_abs[mask_zero]**2))
            / np.mean(PS[mask_pos])
        )

        err["PS"].append(err_PS)
        err["PS0"].append(err_PS0)

        # --- Stopping conditions ---
        if len(err["H"]) >= max_it:
            break

        if abs(err["H"][-1] - err["H"][-2]) / err["H"][-1] < min_speed:
            break

        if err["H"][-1] < min_err or (
            err_PS < min_err and err_PS0 < min_err
        ):
            break

    # --- Rescale outputs ---
    Zh = Zh * Sq + Zh_mean
    Zs = Zs * Sq + Zh_mean

    return Zh, Zs, err


def save_heightfield_vtk(Z,M,N, filename='orig_surface.vts' ):
    endpointx = 8*np.pi
    endpointy = 2*np.pi
    x = np.linspace(0,(endpointx-(endpointx/M)),M)
    y = np.linspace(0,(endpointy-(endpointy/N)),N)
    X, Y = np.meshgrid(x,y,indexing='ij')
    grid=pv.StructuredGrid(X,Y,Z)
    grid.save(filename)

def generate_fourier_coefs(Z, M, N):
    f = np.fft.fft2(Z)
    A = np.real(f)/(M*N)
    B = np.imag(f)/(M*N)

    return A, B

## Generate random distribution

def generate_randomDist(k, lam, randomSeed,M,N):
    rng = np.random.default_rng(int(randomSeed))
    ranDist = lam * rng.weibull(k, size=(M,N))

    return ranDist


# ## Generate Power Spectrum

def generate_power_Spectrum(Sq, beta,M,N):
    q_lin_x = np.fft.fftfreq(N)
    q_lin_y = np.fft.fftfreq(M)
    qx, qy = np.meshgrid(q_lin_x, q_lin_y)


    Cq = ((Sq**2) * beta) / (2*np.pi*((beta**2 + qx**2 + qy**2)**(3/2)))
    return Cq



def save_SurfaceGenOutput(randomSeed, M, N, k, lam, beta, A, B,
                          nelemZ=None, nelemth=None, polyOrder=None,
                          spectrum_param_name='beta', rel_width=None, filename='SurfaceGenOutput.txt'):
    with open(filename, 'w') as f:
        f.write('###########################\n')
        f.write('Random seed:\n')
        f.write(repr(randomSeed) + '\n')
        f.write('M:\n')
        f.write(repr(M)+'\n')
        f.write('N:\n')
        f.write(repr(N)+'\n')
        f.write('shape parameter k: \n')
        f.write(repr(k)+'\n')
        f.write('scaling parameter lambda: \n')
        f.write(repr(lam)+'\n')
        f.write(f'{spectrum_param_name}: \n')
        f.write(repr(beta)+'\n')
        if rel_width is not None:
            f.write('rel_width: \n')
            f.write(repr(rel_width)+'\n')
        if nelemZ is not None:
            f.write('nelemZ: \n')
            f.write(repr(nelemZ)+'\n')
        if nelemth is not None:
            f.write('nelemth: \n')
            f.write(repr(nelemth)+'\n')
        if polyOrder is not None:
            f.write('polyOrder: \n')
            f.write(repr(polyOrder)+'\n')
        f.write('###########################\n')
        f.write('A:\n')
        np.savetxt(f,A)
        f.write('###########################\n')
        f.write('B:\n')
        np.savetxt(f,B)



