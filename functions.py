## NOTE: MOST functions for all the various python scripts are defined here. Some functions are defined within the specific scripts, however the differentiation is not done due to a specific reason.

from mpi4py import MPI
import numpy as np
import pyvista as pv
import subprocess as sp
import re
import math
import os
import json
import warnings

comm = MPI.COMM_WORLD

##################################################
# Functions for deforming the Meshes from smooth to rough or reversely
##################################################

## Remaps a neko mesh, which is a sinusoidal pipe to a smooth pipe
def map_sin_to_smooth( dir, fname_mesh_orig, fname_mesh_deform, pipe_R, r_crit, kplus,Re_tau, lengthZ,nmodZ ):

    from pysemtools.io.ppymech.neksuite import pynekwrite, pynekread
    from pysemtools.datatypes.msh import Mesh
    from pysemtools.datatypes.field import FieldRegistry

    ###########################################################################################
    # Get mpi info
    comm = MPI.COMM_WORLD

    ###########################################################################################
    # intialize the mesh and some fields
    msh = Mesh(comm, create_connectivity=False)
    fld = FieldRegistry(comm)

    ###########################################################################################
    # read mesh 
    file_path = dir + "/" + fname_mesh_orig
    pynekread(file_path, comm, msh=msh, fld = fld, data_dtype=np.single)
      
    ###########################################################################################
    lamZ = lengthZ / nmodZ
    lamTH = lamZ

    mesh_r  = np.sqrt(msh.x**2 + msh.y**2)
    mesh_th = np.arctan2(msh.y,msh.x)
    

    surf = kplus/Re_tau * np.cos(2*np.pi*msh.z/lamZ) * np.cos(2*np.pi*mesh_th*pipe_R/lamTH)
    mesh_change = mesh_r[ mesh_r>r_crit ]
    surf_change = surf[ mesh_r>r_crit ]
    mesh_r_changed = 1.0 / (1.0+surf_change/(pipe_R-r_crit)) * (mesh_change + r_crit/(pipe_R-r_crit) * surf_change)
    mesh_r[ mesh_r>r_crit ] = mesh_r_changed

    msh.x = mesh_r * np.cos(mesh_th)
    msh.y = mesh_r * np.sin(mesh_th)

    ###########################################################################################
    full_fname_mesh_deform = dir + "/" + fname_mesh_deform
    pynekwrite(full_fname_mesh_deform, comm, msh=msh, fld=fld, wdsz=4, write_mesh=True)

# Maps a pointcloud of a smooth pipe generated in python to a sinusoidal pipe
def map_smooth_to_sin(XYZ, Re_tau, Lz, nmodZ, kplus, pipe_R, r_crit):
    X = XYZ[...,0]
    Y = XYZ[...,1]
    Z = XYZ[...,2]

    R, T = cartesian_to_polar(X, Y)

    lamZ = Lz / nmodZ
    lamTH = lamZ

    surf_y = kplus/Re_tau * np.cos(2.0*np.pi*Z/lamZ) * np.cos(2.0*np.pi*T*1.0/lamTH)
    rr = R +(R-r_crit)/(pipe_R-r_crit)*surf_y
    R=rr

    X, Y = polar_to_cartesian(R,T)
    XYZ[...,0] = X
    XYZ[...,1] = Y
    XYZ[...,2] = Z
    grid = pv.StructuredGrid(X, Y, Z)
    grid.save("PointsMappedToSin.vts")
    return XYZ

# Maps rough neko mesh to smooth pipe
def map_rough_to_smooth_neko(dir, comm, fname_mesh_rough, fname_mesh_smooth, Lz, Lth, A, B, amplitude, nModM, nModN, pipe_R,r_crit):
    from pysemtools.io.ppymech.neksuite import pynekwrite, pynekread
    from pysemtools.datatypes.msh import Mesh
    from pysemtools.datatypes.field import FieldRegistry

    if comm.Get_rank() == 0:
        print('Remapping rough surface to smooth')
    msh = Mesh(comm, create_connectivity=False)
    fld = FieldRegistry(comm)

    file_path_rough = dir + "/" + fname_mesh_rough
    pynekread(file_path_rough, comm, msh = msh, fld = fld, data_dtype=np.single)

    mesh_r = np.sqrt(msh.x**2 + msh.y**2)
    mesh_th = np.arctan2(msh.y, msh.x)
    mesh_z = msh.z

    halfZ  = int(np.ceil(nModM/2))
    halfTh = int(np.ceil(nModN/2))

    u = np.where(np.arange(nModM) <= halfZ,  np.arange(nModM), np.arange(nModM) - nModM)
    v = np.where(np.arange(nModN) <= halfTh, np.arange(nModN), np.arange(nModN) - nModN)

    # Nek5000 mesh is not on a tensor-product (r,θ,z) grid, so work with all N points flattened.
    mesh_shape = mesh_z.shape
    z_flat  = mesh_z.ravel()   # (N,)
    th_flat = mesh_th.ravel()  # (N,)

    cos_beta = np.cos(2.0*np.pi * v[:, np.newaxis] * th_flat[np.newaxis, :] / Lth)  # (nModN, N)
    sin_beta = np.sin(2.0*np.pi * v[:, np.newaxis] * th_flat[np.newaxis, :] / Lth)

    FC = A @ cos_beta - B @ sin_beta  # (nModM, N)
    FS = A @ sin_beta + B @ cos_beta
    del cos_beta, sin_beta

    cos_alpha = np.cos(2.0*np.pi * u[:, np.newaxis] * z_flat[np.newaxis, :] / Lz)   # (nModM, N)
    sin_alpha = np.sin(2.0*np.pi * u[:, np.newaxis] * z_flat[np.newaxis, :] / Lz)

    rm = ((cos_alpha * FC - sin_alpha * FS).sum(axis=0)).reshape(mesh_shape)
    rm = rm / amplitude   # scale roughness inside the deformation (matches scaling_factor in the Neko .f90); do NOT shrink the whole radius
    mesh_change = mesh_r[mesh_r>r_crit]
    surf_change = rm[mesh_r>r_crit]
    mesh_r_change = 1.0 / (1.0 + surf_change/(pipe_R-r_crit)) * (mesh_change + r_crit/(pipe_R - r_crit)*surf_change)
    mesh_r[mesh_r > r_crit] =  mesh_r_change
    max_R = np.max(mesh_r)
    msh.x = mesh_r * np.cos(mesh_th)
    msh.y = mesh_r * np.sin(mesh_th)
    
    file_path_smooth = dir + "/" + fname_mesh_smooth
    pynekwrite(file_path_smooth, comm, msh = msh, fld = fld, wdsz = 4, write_mesh=True)
    return max_R

def inverseSurfaceMapping_concentricPoints(XYZ, nModM, nModN, A, B, pipe_R, r_crit, Lz, Lth, amplitude = 1.0):
    import pysemtools.interpolation.utils as interp_utils
    #print('start inverse mpaping interpolation points')
    X, Y, Z = extract_X_Y_Z(XYZ)
    R, T = cartesian_to_polar(X, Y)

    halfM = int(np.ceil(nModM/2))
    halfN = int(np.ceil(nModN/2))

    u = np.where(np.arange(nModM) <= halfM, np.arange(nModM), np.arange(nModM) - nModM)
    v = np.where(np.arange(nModN) <= halfN, np.arange(nModN), np.arange(nModN) - nModN)

    # X, Y, Z have shape (nr, ntheta, nz): Z varies only along axis 2, T only along axis 1.
    z_1d = Z[0, 0, :]   # (nz,)
    t_1d = T[0, :, 0]   # (ntheta,)

    cos_z = np.cos(2.0 * np.pi * u[:, np.newaxis] * z_1d[np.newaxis, :] / Lz)   # (nModM, nz)
    sin_z = np.sin(2.0 * np.pi * u[:, np.newaxis] * z_1d[np.newaxis, :] / Lz)
    cos_t = np.cos(2.0 * np.pi * v[:, np.newaxis] * t_1d[np.newaxis, :] / Lth)  # (nModN, ntheta)
    sin_t = np.sin(2.0 * np.pi * v[:, np.newaxis] * t_1d[np.newaxis, :] / Lth)

    P = A @ cos_t - B @ sin_t    # (nModM, ntheta)
    Q = -A @ sin_t - B @ cos_t  # (nModM, ntheta)

    rm_2d = P.T @ cos_z + Q.T @ sin_z  # (ntheta, nz)
    rm = rm_2d[np.newaxis, :, :]       # (1, ntheta, nz) → broadcasts to (nr, ntheta, nz)
    rm = rm / amplitude   # scale roughness inside the deformation (matches scaling_factor in the Neko .f90); do NOT shrink the whole radius
    rr = 1.0 / (1.0 + rm/(pipe_R-r_crit)) * (R + r_crit/(pipe_R-r_crit)*rm)
    R[R>r_crit] = rr[R > r_crit]
    #print('Avg radius inverse: ', np.mean(R[0,...]))
    X, Y = polar_to_cartesian(R,T)
    # XYZ[...,0] = X
    # XYZ[...,1] = Y
    # XYZ[...,2] = Z
    grid = pv.StructuredGrid(X, Y, Z)
    grid.save("interpolationpoints_mapped_inverse_rough.vts")
    #print('saved inverse mapped points')
    Nstruct = [R.shape[0], R.shape[1], R.shape[2]]
    X_out, Y_out, Z_out = X.copy(), Y.copy(), Z.copy()  # (nr, ntheta, nz) — returned to caller
    X = np.transpose(X, axes=(2,0,1))
    Y = np.transpose(Y, axes=(2,0,1))
    Z = np.transpose(Z, axes=(2,0,1))
    xyz = interp_utils.transform_from_array_to_list(Nstruct[0], Nstruct[1], Nstruct[2], [X, Y, Z])
    #print('Finished inverse mapping interpolation points')
    return xyz, Nstruct, X_out, Y_out, Z_out

# Maps smooth pointcloud to rough pipe 
def map_smooth_to_rough(XYZ, A, B,nModM, nModN, Lz, Lth,  pipe_R, r_crit, amplitude = 1.0, save_vtk=True, filename = 'PointsMappedToRough.vts'):
    X, Y, Z = extract_X_Y_Z(XYZ)

    R, T = cartesian_to_polar(X, Y)
    halfM = math.ceil(nModM/2)
    halfN = math.ceil(nModN/2)

    u = np.where(np.arange(nModM) <= halfM, np.arange(nModM), np.arange(nModM) - nModM)
    v = np.where(np.arange(nModN) <= halfN, np.arange(nModN), np.arange(nModN) - nModN)

    # XYZ is (nr, ntheta, nz, 3): Z varies only along last axis, T only along second axis.
    # rm is therefore independent of the radial index and can be computed on the 2D (ntheta, nz) grid.
    z_1d = Z[0, 0, :]   # (nz,)
    t_1d = T[0, :, 0]   # (ntheta,)

    # cos/sin of the separable phase components
    cos_z = np.cos(2.0 * np.pi * u[:, np.newaxis] * z_1d[np.newaxis, :] / Lz)   # (nModM, nz)
    sin_z = np.sin(2.0 * np.pi * u[:, np.newaxis] * z_1d[np.newaxis, :] / Lz)
    cos_t = np.cos(2.0 * np.pi * v[:, np.newaxis] * t_1d[np.newaxis, :] / Lth)  # (nModN, ntheta)
    sin_t = np.sin(2.0 * np.pi * v[:, np.newaxis] * t_1d[np.newaxis, :] / Lth)

    # Sum over n using angle-addition identity: cos(a+b)=cos_a*cos_b - sin_a*sin_b
    # P[m, theta] = sum_n A[m,n]*cos_t[n] - B[m,n]*sin_t[n]
    # Q[m, theta] = -sum_n A[m,n]*sin_t[n] + B[m,n]*cos_t[n]  (from -sin(a+b) coefficient)
    P = A @ cos_t - B @ sin_t    # (nModM, nModN) @ (nModN, ntheta) = (nModM, ntheta)
    Q = -A @ sin_t - B @ cos_t  # (nModM, ntheta)

    # Sum over m: rm_2d[theta, z] = sum_m P[m]*cos_z[m] + Q[m]*sin_z[m]
    rm_2d = P.T @ cos_z + Q.T @ sin_z  # (ntheta, nz)

    rm = rm_2d[np.newaxis, :, :]  # broadcast over radial dimension → (nr, ntheta, nz)
    
    rr = R + ((((R-r_crit)/(pipe_R-r_crit))*rm)/amplitude)
    R[R>r_crit] = rr[R > r_crit]

    
    X, Y = polar_to_cartesian(R,T)
    XYZ[...,0] = X
    XYZ[...,1] = Y
    XYZ[...,2] = Z
    if save_vtk and comm.Get_rank() == 0:
        grid = pv.StructuredGrid(X, Y, Z)
        grid.save(filename)
        print('Avg radius rough: ', np.mean(R[0,...]))
    return XYZ

# Calculates the yplus values for each point
def calculate_yplus(XYZ, delta_nu, r_max):
    X, Y, Z = extract_X_Y_Z(XYZ)
    R, T = cartesian_to_polar(X, Y)
    y = r_max -R
    r = np.mean(y,axis=(1,2))
    yplus = r/delta_nu

    return yplus, r
##################################################
# Functions for defining interpolation points
##################################################
#convert polar to cartesian coords
def polar_to_cartesian(R,T): 
    X = R * np.cos(T)
    Y = R * np.sin(T)

    return X,Y

# Connverts cartesian coords to polar coords
def cartesian_to_polar(X, Y):
    R = np.sqrt(X**2 + Y**2)
    T = np.arctan2(Y, X)

    return R, T

# create pointclout with sinusoidal pipe shape
def create_sin_surface(r, theta, z, Re_tau, Lz, nmodZ, kplus, r_crit): 
    lamZ = Lz / nmodZ
    lamTH = lamZ
    
    surf_y = kplus/Re_tau* np.cos(2.0*np.pi*z/lamZ)  * np.cos(2.0*np.pi*theta*1.0/lamTH)
    rr = r + (r - r_crit)/(1.0-r_crit)*surf_y
    R = rr
    return R, theta, z

# Maps smooth to rough surface, as done in neko
def create_rough_surface(r, theta, z, amplitude):
    A = np.loadtxt("../A.dat")
    B = np.loadtxt("../B.dat")
    rowsA, colsA  = A.shape
    rowsB, colsB = B.shape
    lz = z[-1] - z[0]
    lth = theta[-1] - theta[0]

    for r in r:
        k = 0.
        rm = 0.
        for m in range(rowsA):
            if (m <= np.ceil(len(z)/2)):
                u = m-1
            else:
                u = m - len(z)
            for n in range(rowsB):
                if (n <= np.ceil(len(theta)/2)):
                    v = n-1
                else:
                    v = n-len(theta)

                phase = 2.0*np.pi*(((u*z)/lz) + (v * theta)/lth)
                k = A[m,n]*np.cos(phase) - B[m,n]*np.sin(phase)
                rm = rm + amplitude*k
                r += rm[0]

    return r, theta, z

# define mesh refinement at wall when generating interpolation pointcloud. Done as in the PySEMTools examples
def geometric_stretching(dx0, stretching, xmax):        
    dx_base = dx0 * (stretching**np.linspace(0,1000, 1001))
    x_base = np.cumsum( dx_base)
    N_inrange = np.sum(x_base<=xmax)
    x_base = x_base[0:N_inrange]
    x_base = x_base/np.max(x_base)
    x = np.zeros(N_inrange+1)
    x[0] = 0.0
    x[1:N_inrange+1] = x_base
    #x *= xmax
    return x

def merge_wall_and_main_points(xyz_wall, xyz_main, Nstruct_wall, Nstruct_main):
    """Concatenate wall and main interpolation points along the radial axis.

    Each block comes from transform_from_array_to_list, i.e. it is the C-order
    flatten of a per-coordinate array of shape (nz, nr_block, ntheta), so nz is
    the slowest-varying index. They must be merged along the radial axis *before*
    re-flattening so that the combined flat ordering matches the
    [nr_w+nr_m, ntheta, nz] reshape done in read_interpolated_stat_hdf5_fields.
    A plain np.vstack instead concatenates whole z-blocks and scrambles
    wall/main data across all radial slots.
    """
    nr_w, nth, nz = Nstruct_wall          # [nr_wall, ntheta, nz]
    nr_m = Nstruct_main[0]                 # Nstruct_main = [nr_main, ntheta, nz]
    ncol = xyz_wall.shape[1]
    wall_4d = xyz_wall.reshape(nz, nr_w, nth, ncol)
    main_4d = xyz_main.reshape(nz, nr_m, nth, ncol)
    merged = np.concatenate([wall_4d, main_4d], axis=1)   # (nz, nr_w+nr_m, ntheta, ncol)
    return merged.reshape(-1, ncol)

# Generation of interpolation pointcould. Mostly according to PySEMTools examples
def user_defined_interpolation_points(nelemZ, nelemTh, polyOrder, targetRetau, rmax, stretching, reynoldsnumber, roughSurface, Lth, Lz):
    import pysemtools.interpolation.utils as interp_utils
       
    theta_bbox = [0.0, Lth]    # define coords of plane
    ntheta = nelemTh*polyOrder                   # define number of points in azimuthal direction
    z_bbox = [0.0, Lz]
    nz = nelemZ*polyOrder                         # define number of points in z-direction

    theta = np.linspace(theta_bbox[0], theta_bbox[1], ntheta+1)#, endpoint = False)     # add one point to exclude since peridic
    theta = .5 * (theta[0:ntheta] + theta[1:])                      # exclude last points since periodic
    z = np.linspace(z_bbox[0], z_bbox[1], nz+1)
    z = .5 * (z[0:nz] + z[1:])

    dr0 = 0.1/targetRetau

    r_base = geometric_stretching(dr0, stretching,rmax)
    if roughSurface == False:
        r_base[0] = 0.1/targetRetau
    r_base = rmax-r_base
    nr = r_base.size
    nz = z.size

    R,T,Z = np.meshgrid(r_base, theta, z, indexing='ij') # build coords in polar coordinate system
    
    wall_pos = 1-(0.1/reynoldsnumber)
    r_wall = [wall_pos, 0.999*wall_pos] 
    theta_wall = theta
    z_wall = z

    R_wall, T_wall, Z_wall = np.meshgrid(r_wall, theta_wall, z_wall, indexing = 'ij')   
    #[nr, ntheta, nz]
    
    R = np.concatenate((R_wall, R))
    T = np.concatenate((T_wall, T))
    Z = np.concatenate((Z_wall, Z))

    X,Y = polar_to_cartesian(R,T)
    
    Nstruct = [X.shape[0], X.shape[1], X.shape[2]]
    if comm.Get_rank() == 0:
        print("Nstruct: ", Nstruct)
    X_T = np.transpose(X, axes=(2, 0, 1))
    Y_T = np.transpose(Y, axes=(2, 0, 1))
    Z_T = np.transpose(Z, axes=(2, 0, 1))
    xyz = interp_utils.transform_from_array_to_list(Nstruct[0], Nstruct[1], Nstruct[2], [X_T, Y_T, Z_T])

    return xyz, Nstruct, X, Y, Z
##################################################
# Averaging functions
##################################################
def av_func_interp(x):
     return x 
  
def _not_found_mask(x):
    # A point is "not found" (written as zeros by interpolation) only when ALL
    # its components are zero.  Individual zero components (e.g. R_uv at the
    # centreline, or R_uw/R_vw throughout a pipe) are legitimate and must not
    # be masked.  For scalar fields (ndim<=3) fall back to exact-zero masking.
    if x.ndim >= 4:
        return np.all(x == 0, axis=-1, keepdims=True)
    return x == 0

def av_func_z(x):
    mask = _not_found_mask(x)
    x = np.where(mask, np.nan, x)
    x = np.nanmean(x, axis = 2, keepdims = False)
    x = switch_nan_to_zero(x)
    return x

def av_func_th(x):
    mask = _not_found_mask(x)
    x = np.where(mask, np.nan, x)
    x = np.nanmean(x, axis = 1, keepdims = False)
    x = switch_nan_to_zero(x)
    return x

def avg_from_3D_to_1D(x, n_sigma=50.0, abs_max=None, return_counts=True):
    """Average a (nr, ntheta, nz[, ncomp]) field down to (nr[, ncomp]),
    ignoring every interpolation 'not-found' point and every extrapolation
    overshoot.

    Two failure modes are filtered before the theta/z mean is taken:

    * 'not-found' points, which the interpolator writes as all-zeros, are
      masked by ``_not_found_mask`` (see its note on legitimate zero
      components).
    * Points that land marginally *outside* the spectral-element mesh are not
      written as zeros but get the element polynomial *extrapolated* beyond its
      [-1, 1] range, which overshoots by many orders of magnitude (|u| up to
      O(1e4) against O(1) physically) and can make the normal Reynolds stresses
      negative.  Being nonzero, they slip past the not-found mask.  A ring holds
      ntheta*nz ~ 1.5e6 points, so a *single* surviving 1e4 spike still shifts
      that ring's mean by O(0.1) in wall units - enough to put a visible wiggle
      into U+(r) while leaving the spectra untouched.  This is the same defect
      the spectra path already removes with a hard |u| > 5 clip; here the cut
      has to be scale-free, because this function is also handed pressure,
      velocity gradients, Reynolds stresses and every budget term, whose
      physical magnitudes differ by orders of magnitude.

    Points are therefore rejected as any value more than ``n_sigma`` robust
    standard deviations (1.4826*MAD) from the median of their own radial ring.
    The cut is applied per component and only where the ring has a finite
    spread (MAD > 0), so genuinely constant components (e.g. R_uw in a pipe)
    are never touched.  The default is deliberately loose: the overshoots sit
    at O(1e4) sigma, while the most intermittent genuine quantity (dissipation)
    stays inside ~20 sigma, so 50 removes the pathology with a wide margin and
    no risk to the physics.  Tighten it towards 10 to also catch milder
    overshoots, at some risk of clipping real extreme events.  ``abs_max``
    additionally imposes a hard magnitude bound for callers that do know the
    physical scale of the field (the spectra path's |u| > 5, say).

    Unlike the spectra path, rejected points are simply dropped rather than
    gap-filled: a hole biases an FFT by injecting a discontinuity, but leaves
    an arithmetic mean unbiased, so filling would add an assumption for nothing.

    The mean is taken over the theta and z axes together so that every
    remaining valid point gets equal weight.  A two-step z-then-theta nanmean
    instead re-detects zeros on the already-reduced array (masking genuine zero
    components) and weights each theta-line equally regardless of how many valid
    points it actually contained.

    With ``return_counts`` the per-radius number of retained and rejected points
    is returned alongside the mean, as ``(out, n_valid, n_rejected)``.  Both are
    worth plotting against r: a jump in either is what a postprocessing artefact
    in U+(r) looks like at its source.

    One radius is processed at a time.  The full array reaches ~8 GB in float64
    at Re_b = 19000, and nanmedian would copy it several times over.
    """
    x = np.asarray(x)
    # if x.ndim < 3:
    # # a 2D array here is almost always an already-averaged (nr, ncomp) field
    # # being passed back in; without this the reshape below silently averages
    # # across the component axis instead and returns quiet nonsense
    #     raise ValueError("avg_from_3D_to_1D expects (nr, ntheta, nz[, ncomp]), "
    #                     "got shape %r - has this field already been averaged?"
    #                     % (x.shape,))
    nr = x.shape[0]
    comp_shape = x.shape[3:]             # () scalar, (ncomp,) vector, (3, 3) tensor
    shape = (nr,) + comp_shape

    out        = np.zeros(shape, dtype=float)
    n_valid    = np.zeros(shape, dtype=np.int64)
    n_rejected = np.zeros(shape, dtype=np.int64)

    for i in range(nr):
        # keep the leading axis so _not_found_mask still sees the original rank
        chunk = np.asarray(x[i:i + 1], dtype=float)
        chunk = np.where(_not_found_mask(chunk), np.nan, chunk)
        # collapse theta and z into one axis, leave any component axes alone
        ring = chunk.reshape((1, -1) + comp_shape)

        # a ring fully inside the roughness is all-NaN, which both nanmedian and
        # nanmean warn about; it legitimately averages to 0, as it did before
        with np.errstate(invalid='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            med   = np.nanmedian(ring, axis=1, keepdims=True)
            sigma = 1.4826 * np.nanmedian(np.abs(ring - med), axis=1, keepdims=True)
            bad   = (np.abs(ring - med) > n_sigma * sigma) & (sigma > 0)
            if abs_max is not None:
                bad |= np.abs(ring) > abs_max
            ring = np.where(bad, np.nan, ring)
            mean_i = np.nanmean(ring, axis=1)

        out[i]        = mean_i[0]
        n_rejected[i] = bad.sum(axis=1)[0]
        n_valid[i]    = np.sum(~np.isnan(ring), axis=1)[0]
    
    out = switch_nan_to_zero(out)
    return (out, n_valid, n_rejected) if return_counts else out


def switch_zero_to_nan(x):
    x = np.where(x == 0, np.nan, x)
    return x

def switch_nan_to_zero(x):
    x = np.nan_to_num(x, nan=0.0)
    return x
##################################################
# Functions for calculating wall friction
##################################################

# Calculates the wall normal vecotr for each point of the wall
def wall_normal_vector_numerical( XYZ ):
    XYZ_first_off_wall = XYZ[0:1,...] #extracts the first points in r

    dz  = XYZ_first_off_wall[0,0,1,2]  - XYZ_first_off_wall[0,0,0,2] # computes the distance between the first two points in Z
    Lz  = XYZ_first_off_wall[0,0,-1,2] - XYZ_first_off_wall[0,0,0,2] + dz # computes the total length in Z

    tmp1 = np.copy(XYZ_first_off_wall[:,:,-1:-2:-1,:]) #Takes the last entry in Z as a slice (keeps dimenstions)

    tmp1[:,:,:,2] =  tmp1[:,:,:,2] - Lz # Subtracts the length in Z from the Z-coord of the last point


    tmp2 = np.copy(XYZ_first_off_wall[:,:,0:1,:]) #Take a slice of the first points in Z
 
    tmp2[:,:,:,2] =  tmp2[:,:,:,2] + Lz # add Lz to the first Z coord of all points

    XYZ_first_off_wall_added = np.concatenate( (tmp1 , XYZ_first_off_wall , tmp2 ) , axis=2 ) # "stacks" tmp1, XYZ_first_off_wall and tmp2 array 

    v1_first_off_wall = np.roll(XYZ_first_off_wall,shift=-1,axis=1) - np.roll(XYZ_first_off_wall,shift=1,axis=1) # Central difference of the points
    v2_first_off_wall = np.roll(XYZ_first_off_wall_added,shift=-1,axis=2) - np.roll(XYZ_first_off_wall_added,shift=1,axis=2) # Central difference of the points
    v2_first_off_wall = v2_first_off_wall[:,:,1:-1,:] #removes first and last entry of v2_first_off_wall, because of periodic


    v1_first_off_wall = .5 * v1_first_off_wall
    v2_first_off_wall = .5 * v2_first_off_wall

    vn_mag = np.linalg.norm(  np.cross( v2_first_off_wall , v1_first_off_wall , axis=3 ) , axis=3 , keepdims=True  ) #compute wall normal vector


    v1_mag = np.linalg.norm( v1_first_off_wall , axis=3 , keepdims=True ) 
    v2_mag = np.linalg.norm( v2_first_off_wall , axis=3 , keepdims=True )

    v1_first_off_wall = v1_first_off_wall / v1_mag
    v2_first_off_wall = v2_first_off_wall / v2_mag

    vn_first_off_wall = np.cross( v2_first_off_wall , v1_first_off_wall , axis=3 ) #compute wall normals at the wall

    return vn_first_off_wall, vn_mag, XYZ_first_off_wall

# calculates the friction at each point of the wall
def calculate_friction(XYZ, Rij, dUidXj, P, Rer):
    XYZ_red     = XYZ[:,0:Rij.shape[1],0:Rij.shape[2],:] #Selects slice of XYZ, doesnt to anything, since XYZ and RIJ have the same shape in dim 1 and 2
    Nxyz_red    = XYZ_red.shape

    dz          = XYZ_red[0,0,1,2] - XYZ_red[0,0,0,2] #XYZ_red[0,0,1,2] - XYZ_red[0,0,0,2] # computes the distance between the first two points in z
    Lz          = np.max( XYZ_red[...,2] ) - np.min( XYZ_red[...,2] ) + dz # computes the total length in Z
    
    vn_first_off_wall, vn_mag, XYZ_first_off_wall = wall_normal_vector_numerical(XYZ)

    dUidXj_3x3  = np.reshape( dUidXj , shape=(*dUidXj.shape[0:3], 3, 3) , order='F') # rearrange dUidXj tensor such that there is a 3x3 matrix for every point
    Sij         = 0.5*(dUidXj_3x3 + np.permute_dims(dUidXj_3x3, (0,1,2,4,3))) #compute strain tensor
    vn_first_off_wall = np.expand_dims(vn_first_off_wall, axis = 4)

    Sij_n               = np.matmul( Sij , vn_first_off_wall ) # project strain tensor on normal vector
    Sij_n               = np.squeeze(Sij_n) #reduce tensor from 5D to 4D array 
    vn_first_off_wall   = np.squeeze( vn_first_off_wall ) #reduce tensor from 5D to 4D array

    Sij_n_red           = Sij_n#av_func_th(Sij_n)

    vn_mag_red          = vn_mag[:,:, 0:Rij.shape[2]]

    fluid_nu        = 1/Rer 

    tauWall         = 2*fluid_nu * Sij_n
    tauWall_red     = 2*fluid_nu * Sij_n_red

    tauWall_field   = tauWall_red[0,...,2] # 2D field of wall friction (theta, z)
    tauWall_red_z   = np.nanmean(tauWall_field, axis = 0, keepdims = True)

    P_wall          = P[0,...] - np.mean(P[-1,...])
  
    Pwall_force     = - P_wall * vn_first_off_wall[0:Rij.shape[2],:]
    presDrag_field  = Pwall_force[...,2] # 2D field of pressure drag (theta, z)

    vn_mag_red  = np.squeeze(vn_mag_red)

    ds = vn_mag_red
    P_wall_forceZ = np.nanmean(presDrag_field, axis = 0, keepdims = True)

    drag_normalization_coef = (1) / (2*np.pi * Lz)#(nmodTh) / (2*np.pi * Lz)

    mean_tauW   = np.sum(ds * tauWall_red_z, axis=(0,1))* drag_normalization_coef
    mean_presDrag = np.sum(ds * P_wall_forceZ, axis=(0,1)) * drag_normalization_coef
    mean_force_wall = mean_tauW + mean_presDrag

    return mean_force_wall, fluid_nu, mean_tauW, mean_presDrag, tauWall_field, presDrag_field

# Writes the complete array of the wall friction to txt file
def write_tauWall_field_to_txt(filepath, tauWall_field):
    """Write the 2D (theta, z) wall friction field to a plain text file."""
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)

    np.savetxt(filepath, tauWall_field, fmt='%22.6e')
##################################################
# Functions for flipping the data into cylindrical coord
##################################################
def define_cylindrical_coord_system_vectors(XYZ):
    v1 = np.zeros_like(XYZ)         # Build unit verctor in direction 1 along z
    v1[...,2] = 1.0

    tmp = np.zeros_like(XYZ[-1,...])
    tmp[...,2] = XYZ[0,...,2]

    v2 =  tmp-XYZ[0,...] # # Build unit vector in direction 2 along the wall normal, based on dimension one of XYZ corresponding ro r
    v2 = v2 / np.sqrt(np.sum(v2**2, axis = -1, keepdims = True))
    v2 = v2 * np.ones_like(v1)

    v3 = np.cross(v1, v2, axis = 3)  # Buildvector in right-handed orthogonal coordinate system, which is a cross product of v1 and v2

    return v1, v2, v3

def convert_2Dtensor_from_9c_to_3x3(T2_9c):
    T2_3x3 = T2_9c.reshape(T2_9c.shape[0:-1]+(3,3))
    return T2_3x3

def convert_2DTensor_from_3x3_to_9c(T2_3x3):
    T2_9c = T2_3x3.reshape(T2_3x3.shape[:-2]+(9,))
    return T2_9c

def rotate_9c2D_tensor(T2,Qtrans,Q):
    from pysemtools.postprocessing.statistics.tensor_manipulations import rotate_2D_tensor
    T2_3x3 = convert_2Dtensor_from_9c_to_3x3(T2)
    T2_rot = rotate_2D_tensor(T2_3x3, Qtrans,Q)
    T2_rot = convert_2DTensor_from_3x3_to_9c(T2_rot)
    return T2_rot, T2_3x3
##################################################
# Functions for extracting Reference data of various papers
##################################################

## NOTE:  extract data of paper https://doi.org/10.1017/jfm.2022.1013
def extract_reference_data_180ReT(path = './dataverse_files/'):
    file = open(f'{path}PIPE_Re180_MEAN.dat')
    file = file.readlines()
    data_u = file[8:]
    file = open(f'{path}PIPE_Re180_RSTE_uzuz.dat')
    file =file.readlines()
    data_budgets_uu= file[8:]
    file = open(f'{path}PIPE_Re180_RSTE_uruz.dat')
    file =file.readlines()
    data_budgets_uv= file[8:]
    file = open(f'{path}PIPE_Re180_RSTE_utut.dat')
    file =file.readlines()
    data_budgets_ww= file[8:]
    file = open(f'{path}PIPE_Re180_RSTE_urur.dat')
    file =file.readlines()
    data_budgets_vv= file[8:]
    file = open(f'{path}PIPE_Re180_RMS.dat')
    file =file.readlines()
    data_rms= file[8:]
    reference_data_u = np.zeros((len(data_u),8))
    reference_data_rms = np.zeros((len(data_rms), 9))
    reference_data_budgets = np.zeros((4,len(data_u),9))
    for i in range(len(data_u)):
        reference_data_u[i] = data_u[i].split()
    for i in range(len(data_rms)):
        reference_data_rms[i] = data_rms[i].split()
    for i in range(len(data_budgets_uu)):
        reference_data_budgets[0,i] = data_budgets_uu[i].split()
        reference_data_budgets[1,i] = data_budgets_vv[i].split()
        reference_data_budgets[2,i] = data_budgets_ww[i].split()
        reference_data_budgets[3,i] = data_budgets_uv[i].split()
    reference_data_budgets[3,:,3:9] = -reference_data_budgets[3,:,3:9]

    return reference_data_u, reference_data_rms, reference_data_budgets

## NOTE:  extract data of paper https://doi.org/10.1017/jfm.2022.1013
def extract_reference_data_550ReT(path = './dataverse_files/'):
    file = open(f'{path}PIPE_Re550_MEAN.dat')
    file = file.readlines()
    data_u = file[8:] 
    file = open(f'{path}PIPE_Re550_RSTE_uzuz.dat')
    file =file.readlines()
    data_budgets_uu= file[8:]
    file = open(f'{path}PIPE_Re550_RSTE_uruz.dat')
    file =file.readlines()
    data_budgets_uv= file[8:]
    file = open(f'{path}PIPE_Re550_RSTE_utut.dat')
    file =file.readlines()
    data_budgets_ww= file[8:]
    file = open(f'{path}PIPE_Re550_RSTE_urur.dat')
    file =file.readlines()
    data_budgets_vv= file[8:]
    file = open(f'{path}PIPE_Re550_RMS.dat')
    file =file.readlines()
    data_rms= file[8:]
    reference_data_u = np.zeros((len(data_u),8))
    reference_data_rms = np.zeros((len(data_rms), 9))
    reference_data_budgets = np.zeros((4,len(data_u),9))
    for i in range(len(data_u)):
        reference_data_u[i] = data_u[i].split()
    for i in range(len(data_rms)):
        reference_data_rms[i] = data_rms[i].split()
    for i in range(len(data_budgets_uu)):
        reference_data_budgets[0,i] = data_budgets_uu[i].split()
        reference_data_budgets[1,i] = data_budgets_vv[i].split()
        reference_data_budgets[2,i] = data_budgets_ww[i].split()
        reference_data_budgets[3,i] = data_budgets_uv[i].split()
    reference_data_budgets[3,:,3:9] = -reference_data_budgets[3,:,3:9]

    return reference_data_u, reference_data_rms, reference_data_budgets

## NOTE:  extract data of paper https://doi.org/10.1017/jfm.2022.1013
def extract_reference_data_1000ReT(path = './dataverse_files/'):
    file = open(f'{path}PIPE_Re1K_MEAN.dat')
    file = file.readlines()
    data_u = file[8:]
    file = open(f'{path}PIPE_Re1K_RSTE_uzuz.dat')
    file =file.readlines()
    data_budgets_uu= file[8:]
    file = open(f'{path}PIPE_Re1K_RSTE_uruz.dat')
    file =file.readlines()
    data_budgets_uv= file[8:]
    file = open(f'{path}PIPE_Re1K_RSTE_utut.dat')
    file =file.readlines()
    data_budgets_ww= file[8:]
    file = open(f'{path}PIPE_Re1K_RSTE_urur.dat')
    file =file.readlines()
    data_budgets_vv= file[8:]
    file = open(f'{path}PIPE_Re1K_RMS.dat')
    file =file.readlines()
    data_rms= file[8:]
    reference_data_u = np.zeros((len(data_u),8))
    reference_data_rms = np.zeros((len(data_rms), 9))
    reference_data_budgets = np.zeros((4,len(data_u),9))
    for i in range(len(data_u)):
        reference_data_u[i] = data_u[i].split()
    for i in range(len(data_rms)):
        reference_data_rms[i] = data_rms[i].split()
    for i in range(len(data_budgets_uu)):
        reference_data_budgets[0,i] = data_budgets_uu[i].split()
        reference_data_budgets[1,i] = data_budgets_vv[i].split()
        reference_data_budgets[2,i] = data_budgets_ww[i].split()
        reference_data_budgets[3,i] = data_budgets_uv[i].split()
    reference_data_budgets[3,:,3:9] = -reference_data_budgets[3,:,3:9]

    return reference_data_u, reference_data_rms, reference_data_budgets

## ## NOTE:  extract data of paper https://doi.org/10.1007/s10494-013-9482-8
def extract_reference_data_360ReT(path = './dataverse_files/'):
    file = open(f'{path}360_Re_1.dat')
    file = file.readlines()
    data_1 = file[24:]
    file = open(f'{path}360_Re_2.dat')
    file = file.readlines()
    data_2 = file[20:]
    file = open(f'{path}360_Re_3.dat')
    file = file.readlines()
    data_3 = file[20:]
    file = open(f'{path}360_RR_Budget.dat')
    file = file.readlines()
    data_rr_budget = file[21:]
    file = open(f'{path}360_RZ_Budget.dat')
    file = file.readlines()
    data_rz_budget = file[21:]
    file = open(f'{path}360_TT_Budget.dat')
    file = file.readlines()
    data_tt_budget = file[21:]
    file = open(f'{path}360_ZZ_Budget.dat')
    file = file.readlines()
    data_zz_budget = file[21:]
    reference_data_1 = np.zeros((len(data_1), 9))
    reference_data_2 = np.zeros((len(data_2), 7))
    reference_data_3 = np.zeros((len(data_3), 10))
    reference_data_rr_budget = np.zeros((len(data_rr_budget), 8))
    reference_data_rz_budget = np.zeros((len(data_rz_budget), 8))
    reference_data_tt_budget = np.zeros((len(data_tt_budget), 8))
    reference_data_zz_budget = np.zeros((len(data_zz_budget), 8))
    for i in range(len(data_1)):
        reference_data_1[i] = data_1[i].split()
    for i in range(len(data_2)):
        reference_data_2[i] = data_2[i].split()
    for i in range(len(data_3)):
        reference_data_3[i] = data_3[i].split()
    for i in range(len(data_rr_budget)):
        reference_data_rr_budget[i] = data_rr_budget[i].split()
    for i in range(len(data_rz_budget)):
        reference_data_rz_budget[i] = data_rz_budget[i].split()
    reference_data_rz_budget[:,2:] = -reference_data_rz_budget[:,2:]
    for i in range(len(data_tt_budget)):
        reference_data_tt_budget[i] = data_tt_budget[i].split()
    for i in range(len(data_zz_budget)):
        reference_data_zz_budget[i] = data_zz_budget[i].split()
    return reference_data_1, reference_data_2, reference_data_3, reference_data_rr_budget, reference_data_rz_budget, reference_data_tt_budget, reference_data_zz_budget

def calculate_turb_law(karman_const, B, yplus):
    uplus_log_law = ((1/karman_const) * np.log(yplus)) + B
    sublayer = np.linspace(0,12,100)
    return uplus_log_law, sublayer
##################################################
# functions to calculate surface data
##################################################
def calculate_mean_radius(XYZ):
    X, Y, Z = extract_X_Y_Z(XYZ)
    R, T = cartesian_to_polar(X, Y)
    R = R[0,...]

    return np.mean(R)

def calculate_Effective_streamwise_slope(XYZ,Lz, Lth):
    X, Y, Z = extract_X_Y_Z(XYZ)
    R, T = cartesian_to_polar(X, Y)
    R = R[0,...]
    dz = Lz / (R.shape[1]-1)
    dth = Lth / (R.shape[0]-1)
    dr_dth, dr_dz = np.gradient(R, dth, dz)
    es = (1/(R.shape[0]*R.shape[1])) * (np.sum(np.abs(dr_dz)))
    return es

def calculate_Effective_azimuthal_slope(XYZ,Lz, Lth):
    X, Y, Z = extract_X_Y_Z(XYZ)
    R, T = cartesian_to_polar(X, Y)
    R = R[0,...]
    dz = Lz / (R.shape[1]-1)
    dth = Lth / (R.shape[0]-1)
    dr_dth, dr_dz = np.gradient(R, dth, dz)
    es = (1/(R.shape[0]*R.shape[1])) * (np.sum(np.abs(dr_dth)))
    return es

def calculate_Effective_full_slope(XYZ,Lz, Lth):
    X, Y, Z = extract_X_Y_Z(XYZ)
    R, T = cartesian_to_polar(X, Y)
    R = R[0,...]
    dz = Lz / (R.shape[1]-1)
    dth = Lth / (R.shape[0]-1)
    # R is (ntheta, nz): axis 0 -> theta (spacing dth), axis 1 -> z (spacing dz).
    dr_dth, dr_dz = np.gradient(R, dth, dz)
    full_grad = np.sqrt(dr_dz**2+dr_dth**2)
    es = (1/(R.shape[0]*R.shape[1])) * (np.sum(full_grad))
    return es

def calculate_Effective_streamwise_slope_flat(Y, M, N, Lz, Lth):

    print('Y.shaape: ',Y.shape)
    Y = np.squeeze(Y)
    print('Y.shaape: ',Y.shape)
    #R, T = cartesian_to_polar(X, Y)
    #R = R[0,...]
    dz = Lz / M
    dth = Lth / N
    dr_dth, dr_dz = np.gradient(Y, dz, dth) 
    es = (1/(M*N)) * (np.sum(np.abs(dr_dz)))
    print('dr_dz: ', dr_dz)
    print('Y: ', Y)
    
    return es

def calculate_rms_height(XYZ):
    X, Y, Z = extract_X_Y_Z(XYZ)
    R, T = cartesian_to_polar(X, Y)
    R = R[0,...]
    deviations = R - np.mean(R) #calculate height deviations, --> calculate the variation around the "middle" of the surface
    Sq = np.sqrt(np.sum(deviations**2)/R.size)#(M*polyOrder*N*polyOrder))
    
    return Sq

def calculate_skewness(XYZ, Sq):
    X, Y, Z = extract_X_Y_Z(XYZ)
    R, T = cartesian_to_polar(X, Y)
    R = R[0,...]
    deviations = R - np.mean(R)
    skew = np.sum(pow(deviations, 3))/(R.shape[0]*R.shape[1]*pow(Sq,3))
    return skew

def calculate_mean_height(XYZ):
    X, Y, Z = extract_X_Y_Z(XYZ)
    R, T = cartesian_to_polar(X, Y)
    R = R[0,...]
    mean_height = np.mean(R)
    return mean_height

def calculate_max_peakToValley_height(XYZ):
    X,Y,Z = extract_X_Y_Z(XYZ)
    R, T = cartesian_to_polar(X, Y)
    R = R[0,...]
    Sz = np.max(R) - np.min(R)
    return Sz

def calculate_avg_roughness_height(XYZ):
    X,Y,Z = extract_X_Y_Z(XYZ)
    R, T = cartesian_to_polar(X, Y)
    R = R[0,...]
    R = R - np.mean(R)
    Sa = np.mean(np.abs(R))
    return np.mean(Sa)

def calculate_meltdown_radius(XYZ, pipe_R=1.0):
    """Equivalent smooth-pipe radius after 'melting' the roughness so the
    hills fill the valleys (fluid/solid volume conserved).

    The rough wall sits at R(theta, z). Volume conservation in cylindrical
    coordinates gives pi*R_melt^2 = (1/2pi)*integral R^2 dtheta, i.e.
    R_melt = sqrt(<R^2>), the area/volume-equivalent smooth radius.

    Returns
    -------
    R_melt : equivalent smooth radius
    k_melt : meltdown height = pipe_R - R_melt (effective radius reduction)
    """
    X, Y, Z = extract_X_Y_Z(XYZ)
    R, T    = cartesian_to_polar(X, Y)
    R       = R[0, ...]                  # rough wall surface (n_theta, n_z)
    R_melt  = np.sqrt(np.mean(R**2))     # volume/area-conserving radius
    k_melt  = pipe_R - R_melt
    return R_melt, k_melt

def calculate_mean_peak_to_valley_height(XYZ, nTiles):
    """Mean peak-to-valley height (Smz / Sz_mean).

    The surface is divided into nTiles x nTiles equal regions and the
    peak-to-valley range (max - min of R) is averaged over all tiles.
    nTiles=5 → 25 tiles, consistent with ISO 25178 / Pirozzoli (2023).
    """
    X,Y,Z = extract_X_Y_Z(XYZ)
    R,T = cartesian_to_polar(X,Y)
    R = R[0,...]
    pts0 = (R.shape[0] // nTiles) * nTiles   # trim to exact multiple
    pts1 = (R.shape[1] // nTiles) * nTiles
    R_cropped = R[:pts0, :pts1]
    tile0 = pts0 // nTiles
    tile1 = pts1 // nTiles
    R_tiled = R_cropped.reshape(nTiles, tile0, nTiles, tile1)
    peakToValleyHeight = R_tiled.max(axis=(1, 3)) - R_tiled.min(axis=(1, 3))
    return np.mean(peakToValleyHeight)
    
##################################################
# random stuff
##################################################
# Extract simulation time at which the calculation of statistics started
def find_stats_start_time():
    fluid_stats_file_head = sp.run("head -c100 ../fluid_stats-10.f00000", shell = True, capture_output=True, text = True)
    fluid_stats_file_head = fluid_stats_file_head.stdout.split()
    stats_start_time = float(fluid_stats_file_head[7])

    return stats_start_time

def extract_X_Y_Z(XYZ):
    X = XYZ[...,0]
    Y = XYZ[...,1]
    Z = XYZ[...,2]
    return X, Y, Z

def extract_surface_gen_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    #-----------------------------------
    # Extract scalar quantities
    # -----------------------------------

    def _to_float(s):
        # Values may be written as plain numbers or, under NumPy 2.x repr(),
        # wrapped as e.g. "np.float64(9.39)".
        s = s.strip()
        if s.startswith("np.float") and s.endswith(")"):
            s = s[s.index("(") + 1:-1]
        return float(s)

    k = None
    lam = None
    beta = None

    for i, line in enumerate(lines):
        if "M:" in line:
            M = _to_float(lines[i+1])
        if "N:" in line:
            N = _to_float(lines[i+1])
        if "shape parameter k:" in line:
            k = _to_float(lines[i+1])
        if 'scaling parameter lambda:' in line:
            lam = _to_float(lines[i+1])
        if 'beta:' in line or 'lam0:' in line:   # spectrum parameter (old: beta, new: lam0)
            beta = _to_float(lines[i+1])

    # -----------------------------------
    # Extract matrix A
    # -----------------------------------

    idx_A = lines.index("A:\n") + 1
    idx_sep = lines.index("###########################\n", idx_A)

    A_lines = lines[idx_A:idx_sep]

    A = np.loadtxt(A_lines)

    # -----------------------------------
    # Extract matrix B
    # -----------------------------------

    idx_B = lines.index("B:\n") + 1

    B_lines = lines[idx_B:]

    B = np.loadtxt(B_lines)

    return A, B, k,lam,beta,  M, N
##################################################
# Functions for writing and reading plot data
##################################################

# wrote the data calculated in postprocessing to txt file, to make plotting the data easy
def write_plot_data_to_txt(filepath, scalars, yplus_table, r_table, sublayer):
    """Write post-processing plot data to a human-readable column-oriented text file.

    scalars     : dict[str, float]       – metadata (Re_tau, u_tau, …)
    yplus_table : dict[str, 1-D array]   – profiles indexed by y+ (all same length)
    r_table     : dict[str, 1-D array]   – profiles indexed by r  (all same length)
    sublayer    : 1-D array              – viscous-sublayer reference (y+ = U+)
    """
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)

    sep = '# ' + '=' * 62 + '\n'

    with open(filepath, 'w') as f:
        # ── metadata ───────────────────────────────────────────────────────
        f.write(sep)
        f.write('# METADATA\n')
        f.write(sep)
        for key, val in scalars.items():
            f.write(f'{key:<20} {float(val):.6e}\n')
        f.write('\n')

        # ── yplus-indexed profiles ─────────────────────────────────────────
        keys_yp = list(yplus_table.keys())
        n_yp = len(next(iter(yplus_table.values())))
        f.write(sep)
        f.write(f'# YPLUS_PROFILES  N={n_yp}\n')
        f.write(sep)
        _col_hdr = lambda keys: '#' + ' '.join(
            [f'{keys[0]:>21}'] + [f'{k:>22}' for k in keys[1:]]
        ) + '\n'
        f.write(_col_hdr(keys_yp))
        data_yp = np.column_stack([np.asarray(yplus_table[k], dtype=float) for k in keys_yp])
        np.savetxt(f, data_yp, fmt='%22.6e')
        f.write('\n')

        # ── r-indexed profiles ─────────────────────────────────────────────
        keys_r = list(r_table.keys())
        n_r = len(next(iter(r_table.values())))
        f.write(sep)
        f.write(f'# R_PROFILES  N={n_r}\n')
        f.write(sep)
        f.write(_col_hdr(keys_r))
        data_r = np.column_stack([np.asarray(r_table[k], dtype=float) for k in keys_r])
        np.savetxt(f, data_r, fmt='%22.6e')
        f.write('\n')

        # ── sublayer reference ─────────────────────────────────────────────
        sublayer = np.asarray(sublayer, dtype=float)
        f.write(sep)
        f.write(f'# SUBLAYER  N={len(sublayer)}\n')
        f.write(sep)
        f.write(f'#{"sublayer":>21}\n')
        np.savetxt(f, sublayer, fmt='%22.6e')

# read the generated txt file with the postprocessed data to enable plotting
def read_plot_data_from_txt(filepath):
    """Read a file written by write_plot_data_to_txt.

    Returns
    -------
    scalars  : dict[str, float]
    sections : dict[str, dict[str, np.ndarray]]
               Keys: 'YPLUS_PROFILES', 'R_PROFILES', 'SUBLAYER'
               Each value maps column name → 1-D array.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    scalars = {}
    sections = {}
    SECTION_TAGS = ('YPLUS_PROFILES', 'R_PROFILES', 'SUBLAYER')
    i = 0

    while i < len(lines):
        stripped = lines[i].strip()

        if not stripped or '===' in stripped:
            i += 1
            continue

        if stripped == '# METADATA':
            i += 1
            while i < len(lines):
                l = lines[i].strip()
                if '===' in l:
                    i += 1
                    continue
                if not l:
                    break
                if not l.startswith('#'):
                    parts = l.split()
                    if len(parts) == 2:
                        scalars[parts[0]] = float(parts[1])
                i += 1

        elif any(stripped.startswith(f'# {tag}') for tag in SECTION_TAGS):
            tag = next(t for t in SECTION_TAGS if stripped.startswith(f'# {t}'))
            i += 1
            while i < len(lines) and '===' in lines[i]:   # skip closing separator
                i += 1
            if i >= len(lines) or not lines[i].strip().startswith('#'):
                continue
            col_names = lines[i].strip()[1:].strip().split()
            i += 1
            data_lines = []
            while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith('#'):
                data_lines.append(lines[i])
                i += 1
            if not data_lines:
                continue
            raw = np.loadtxt(data_lines)
            if raw.ndim == 1:
                raw = raw.reshape(-1, 1) if len(col_names) == 1 else raw.reshape(1, -1)
            sections[tag] = {col: raw[:, j] for j, col in enumerate(col_names)}

        else:
            i += 1

    return scalars, sections


def read_new_plot_data(filepath):
    """Read plot_data.txt written in the # SCALARS / # ARRAY format.

    Returns
    -------
    scalars : dict[str, float]
    arrays  : dict[str, np.ndarray]   – keyed by array name (e.g. 'Rij', 'yplus')
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    scalars = {}
    arrays  = {}
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        if stripped == '# SCALARS':
            i += 1
            while i < len(lines):
                l = lines[i].strip()
                if l.startswith('#'):
                    break
                parts = l.split()
                if len(parts) == 2:
                    scalars[parts[0]] = float(parts[1])
                i += 1

        elif stripped.startswith('# ARRAY '):
            tokens    = stripped.split()          # ['#', 'ARRAY', name, 'shape=(...)']
            name      = tokens[2]
            shape_str = tokens[3].replace('shape=', '').strip('()')
            shape     = tuple(int(x) for x in shape_str.split(',') if x.strip())
            i += 1
            data_lines = []
            while i < len(lines) and not lines[i].strip().startswith('#'):
                data_lines.append(lines[i])
                i += 1
            if data_lines:
                raw = np.fromstring(' '.join(data_lines), sep=' ')
                arrays[name] = raw.reshape(shape) if len(shape) > 1 else raw
            continue

        i += 1

    return scalars, arrays


def read_piro(filepath):
    """Read a Pirozzoli inner-scaled statistics file (e.g. stats_inn_4400_grit.dat).

    The file is a whitespace-delimited table with a single header line and the
    columns (all in wall units):
        y+   Ux+   Ur+   Utheta+   <ux^2>   <ur^2>   <utheta^2>   <uxur>

    Returns
    -------
    data : dict[str, np.ndarray]
        Mapping column name -> 1-D array.  Convenience keys 'yplus' and 'Uplus'
        are aliases for 'y+' and 'Ux+'.  Rows are sorted by increasing y+.
    """
    col_names = ['y+', 'Ux+', 'Ur+', 'Utheta+',
                 '<ux^2>', '<ur^2>', '<utheta^2>', '<uxur>']
    raw = np.loadtxt(filepath, skiprows=1)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    order = np.argsort(raw[:, 0])          # sort by increasing y+
    raw = raw[order]
    data = {col: raw[:, j] for j, col in enumerate(col_names)}
    data['yplus'] = data['y+']
    data['Uplus'] = data['Ux+']
    return data


##################################################
# Functions for computing additional data
##################################################
def compute_skinFrictionCoefficient(tauWall,UVW, rho = 1.0):
    Cf = tauWall/(0.5*rho*(UVW[:,0]**2))

    return Cf

def calculate_bulk_velocity_U(UVW, r, r_max):
    """Area-weighted bulk (mean) velocity in the streamwise (U) direction.

    For a circular pipe of radius r_max the bulk velocity is the cross-section
    average of the streamwise velocity,
        U_b = (1/A) * integral U dA = (2/r_max**2) * integral_0^r_max U(R) R dR,
    with R the distance from the pipe axis.  The radial weighting R accounts for
    the larger area of the outer annuli, so a plain mean of U would overweight
    the (slow) near-wall region.

    Parameters
    ----------
    UVW   : (nr, 3) or (nr,) array – radially-resolved mean velocity; the first
            component (or the array itself) is the streamwise velocity U.
    r     : (nr,) array – wall distance of each radial station (0 at the wall),
            as returned by calculate_yplus.
    r_max : pipe radius.

    Returns
    -------
    U_b : bulk velocity in the streamwise direction.
    """
    U = UVW[..., 0] if UVW.ndim > 1 else UVW
    R = r_max - r                 # distance from the pipe axis (0 at centre)
    order = np.argsort(R)         # integrate from the axis (R=0) outward
    R = R[order]
    U = U[order]
    # np.trapz was renamed to np.trapezoid in NumPy 2.0; fall back for old versions
    trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
    U_b = 2.0 / r_max**2 * trapezoid(U * R, R)
    return U_b


##################################################
# Interpolate multiple instantaneous fields
##################################################

def interpolate_multiple_fields(full_fname_mesh , these_field_file_names , xyz , 
                       interpolation_output_fname="interpolated_fields.hdf5" , 
                       if_pass_points_to_rank0_only=True):
    from mpi4py import MPI  # equivalent to the use of MPI_init() in C
    import numpy as np
    from pysemtools.datatypes.msh import Mesh
    from pysemtools.datatypes.field import FieldRegistry
    from pysemtools.datatypes.coef import Coef
    from pysemtools.io.ppymech.neksuite import pynekread
    from pysemtools.interpolation.probes import Probes

    comm = MPI.COMM_WORLD

    # intialize the mesh and some fields
    msh = Mesh(comm, create_connectivity=False)
    fields = FieldRegistry(comm)

    # read in the mesh
    if comm.Get_rank() == 0:
        print('reading mesh file')
    pynekread(full_fname_mesh, comm, msh=msh, data_dtype=np.single)
    if comm.Get_rank() == 0:
        print('mesh file read')
    # compute coef, for interpolation
    coef = Coef(msh, comm, get_area=False)

    ###########################################################################################
    # initiate probes
    probes = Probes(
        comm,
        probes=xyz,
        msh=msh,
        point_interpolator_type="multiple_point_legendre_numpy",
        max_pts=256,
        find_points_iterative=[True, 1],
        global_tree_type="rank_bbox",
        output_fname = interpolation_output_fname
    )
    if comm.Get_rank() == 0:
        print('initiated probes')
    ###########################################################################################
    for fname in these_field_file_names:
        
        if comm.Get_rank() == 0:
            print("----------- working on file: ", fname)

        for this_file_key in ['vel_0','vel_1','vel_2','pres']:
            if comm.Get_rank() == 0:
                print('---on key: ', this_file_key )

            # load the field
            fields.add_field(
                comm,
                field_name=this_file_key,
                file_type="fld",
                file_name=fname,
                file_key=this_file_key,
                dtype=np.single,
            )

        if comm.Get_rank() == 0:
            print('presumably done reading the field')

##        for this_file_key in ['u','v','w','p']:
        # interpolate the fields
        probes.interpolate_from_field_list(
            fields.t, [ fields.registry["vel_0"] , fields.registry["vel_1"] , fields.registry["vel_2"] , fields.registry["pres"] ] , 
            comm, write_data=True
        )


def build_consolidated_stats_index(comm, data_path, stats_file_types, stat_start_time,
                                   output_folder="./", out_name="fluid_stats_index.json"):
    """Index every restart's stats series, then merge into ONE time-sorted index
    whose intervals are the true incremental windows (t - previous t)."""
    from pysemtools.postprocessing.file_indexing import index_files_from_folder
    index_files_from_folder(comm, folder_path=data_path, run_start_time=0,
                            stat_start_time=stat_start_time, output_folder=output_folder,
                            file_type=list(stats_file_types), include_file_contents=True,
                            include_time_interval=True, overwrite=True)
    out_path = os.path.join(output_folder, out_name)
    if comm.Get_rank() == 0:
        entries = []
        for ftype in stats_file_types:
            idx_name = os.path.join(output_folder,
                                    os.path.basename(ftype).split(".")[0] + "_index.json")
            if not os.path.exists(idx_name):   # skip series not present on disk
                continue
            with open(idx_name) as f:
                idx = json.load(f)
            entries += [v for v in idx.values() if isinstance(v, dict) and "time" in v]

        entries.sort(key=lambda e: e["time"])          # global time order
        consolidated = {"statistics_start_time": stat_start_time}
        prev = stat_start_time
        for i, e in enumerate(entries):                 # re-chain the windows
            e["time_previous_output"] = prev
            e["time_interval"] = e["time"] - prev
            prev = e["time"]
            consolidated[i] = e
        with open(out_path, "w") as f:
            json.dump(consolidated, f, indent=4)
    comm.Barrier()
    return out_path