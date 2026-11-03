import os
import numpy as np
from scipy.optimize import minimize
from functions import *
from randomSurfaceGeneration import *
import mpi4py

dir = 'dir/dir/dir'
casename = 'CaseName'

os.chdir(dir)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

## NOTE: Define the various surface parameters, the surface should be optimized for

scaling_factor = 1.0

target_R                = 1.0
target_avg_height       = 0.03082 / scaling_factor
target_rms_height       = 0.03888/scaling_factor
target_skewness         = -0.51858
target_max_peaktotrough = 0.30647 / scaling_factor
target_es               = 0.22742
target_meanpeaktovalley = 0.20585 / scaling_factor

targets = np.array([
    target_R, target_avg_height, target_rms_height, target_skewness,
    target_max_peaktotrough, target_es, target_meanpeaktovalley,
])
METRIC_NAMES = [
    'avg_R', 'avg_height', 'rms_height', 'skewness',
    'max_peak_to_trough', 'eff_slope', 'mean_peak_to_valley',
]
# weight per metric (0 disables); errors are normalised by the target value.
weights = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

#Define mesh settings
polyOrder       = 7             # Targeted polynomial order
PTS_PER_ELEMENT = 15            # How many mesh points should be in the mesh per fourer mode. Not used right now, M and N could be changed dynamically 
nelemZ          = 80            # Elements in lenth
nelemth         = 40            # elemtns in azimuthal direction
# highest resolved mode is M/2, so PTS_PER_ELEMENT points across the smallest
# streamwise feature => M = 2*(nelemZ*polyOrder)/PTS_PER_ELEMENT, floored to even.
M               = 58            # Inclueded fourier modes in z-direction
N               = 28            # Included fourier modes in azimuthal direction
targetRetau    = 315            # Approximation of Re_tau of the rough case
rmax           = 1.0            # maximum radius of smooth pipe
stretching     = 1.05           # value for mesh refinement (gemoetrical progression) near the wall of generated pointcloud. 
Lz             = 8. * np.pi     # Domain size in length
Lth            = 2. * np.pi     # Domain size in azimuthal directionn
r_crit         = 0.7            # Maximum radius at which the mesh is shifted to rough      
reynoldsnumber = 5700           # Re_b of case

# parameters for surface generation (initial guess for the optimisation)
k = 22.82204            # shape factor of random distribution
lam = 0.48457           # scale factor of random distribution
Sq = 1.32568            # broadband-spectrum amplitude
beta = 11.18727         # broadband-spectrum decay 

Seed = 10000            # fixed seed, necessary since the difference between each random seed can be greater than the difference of altered optimization values

# surf generator settings. Generator stops, if one of the three is reached
max_iters = 10000       # max iterations of surface generator
min_speed = 0.0         # minimum difference between the last two interations of surface generator
min_err = 1e-4          # Minimum error between Zh and Zs

xyz, Nstruct, X, Y, Z = user_defined_interpolation_points(
    nelemZ, nelemth, polyOrder, targetRetau, rmax, stretching,
    reynoldsnumber=reynoldsnumber)
XYZ_base = np.stack([X[2:], Y[2:], Z[2:]], axis=-1)

## NOTE: Generates the surface from the input parameters k, lamda, Sq and beta. Then returns its seven surface metrics
def evaluate_surface(k, lam, Sq, beta, save_vtk=False, filename = 'PointsMappedToRough.vts'):
    Ps = generate_power_Spectrum(Sq=Sq, beta=beta, M=M, N=N)
    RanDist = generate_randomDist(k=k, lam=lam, randomSeed=Seed, M=M, N=N)
    Zh, Zs, err = fractal_surf_generator(
        PS=Ps, Zh=RanDist, M=M, N=N, max_it=max_iters,
        min_speed=min_speed, min_err=min_err)
    Zh = Zh - np.mean(Zh)
    A, B = generate_fourier_coefs(Z=Zh, M=M, N=N)

    XYZ = map_smooth_to_rough(
        XYZ=XYZ_base.copy(), A=A, B=B, nModM=M, nModN=N,
        Lz=Lz, Lth=Lth, pipe_R=rmax, r_crit=r_crit, save_vtk=save_vtk, filename = filename)

    rms_height = calculate_rms_height(XYZ)
    return np.array([
        calculate_mean_radius(XYZ),
        calculate_avg_roughness_height(XYZ),
        rms_height,
        calculate_skewness(XYZ, rms_height),
        calculate_max_peakToValley_height(XYZ),
        calculate_Effective_full_slope(XYZ, Lz, Lth),
        calculate_mean_peak_to_valley_height(XYZ, nTiles=5),
    ])


## NOTE: Calculates the loss of the current optimization step - hence a value for the difference between the current state and the target
def loss(params):
    """Normalised weighted squared error of the metrics against the targets."""
    k, lam, Sq, beta = np.abs(params)   # keep parameters positive
    calcs = evaluate_surface(k, lam, Sq, beta)
    safe_targets = np.where(np.abs(targets) > 1e-12, targets, 1.0)
    return float(np.sum(weights * ((calcs - targets) / safe_targets) ** 2))


# ── Optimise surface parameters to best match the target metrics ──────────────
if rank == 0:
    print('Optimising surface parameters (k, lam, Sq, beta) ...')
    print('-------------------------------------------')
    print('Targets: ')
    print('target avg height: ', target_avg_height)
    print('target rms height: ', target_rms_height)
    print('target skewness: ', target_skewness)
    print('target es: ', target_es)
    print('target mean ptv: ', target_meanpeaktovalley)
    print('-------------------------------------------')
    print('parameters')
    print('M:', M)
    print('N: ', N)
    print('poly Order: ', polyOrder)
    print('PTS per roughness feature: ', PTS_PER_ELEMENT)
    print('nelemZ: ',nelemZ)
    print('nelemTh: ', nelemth)
res = minimize(loss, x0=[k, lam, Sq, beta], method='Nelder-Mead',
               options={'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True, 'maxiter': 500 }) ## With the options, it can be controlled, how long the optmization runs
k, lam, Sq, beta = np.abs(res.x)

if rank == 0:
    print(f'\nOptimised parameters: k={k:.5f}  lam={lam:.5f}  '
          f'Sq={Sq:.5f}  beta={beta:.5f}  (loss={res.fun:.6f})')

# Final surface with the optimised parameters (writes PointsMappedToRough.vts)
calcs = evaluate_surface(k, lam, Sq, beta, save_vtk=True, filename=casename+'.vts')
if rank == 0:
    print('\nMetrics at optimised parameters:')
    for name, c, t in zip(METRIC_NAMES, calcs, targets):
        print(f'  {name:25s}  calc={c:+.5f}  target={t:+.5f}  diff={c-t:+.5f}')

# Save the optimised surface (rank 0 only, to avoid concurrent writes under MPI)
if rank == 0:
    Ps = generate_power_Spectrum(Sq=Sq, beta=beta, M=M, N=N)
    RanDist = generate_randomDist(k=k, lam=lam, randomSeed=Seed, M=M, N=N)
    Zh, Zs, err = fractal_surf_generator(PS=Ps, Zh=RanDist, M=M, N=N, max_it=max_iters, min_speed=min_speed, min_err=min_err)
    Zh = Zh - np.mean(Zh)
    A, B = generate_fourier_coefs(Z=Zh, M=M, N=N)

    save_SurfaceGenOutput(randomSeed=Seed, M=M, N=N, k=k, lam=lam, beta=beta, A=A, B=B,
                          spectrum_param_name='beta', filename='SurfaceGenOutput_'+casename+'.txt')
