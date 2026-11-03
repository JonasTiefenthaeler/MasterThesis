import numpy as np
from mpi4py import MPI
from functions import *
import glob
import os
import pysemtools.interpolation.utils as interp_utils


nelemZ = 246
nelemTh = 123
polyOrder = 7
targetReTau = 763
r_max = 1.
roughness_scaling = 1.3349
r_crit = 0.5
stretching = 1.05
Reynolds_number = 9500

lengthZ = 8*np.pi
lengthTh = 2*np.pi

surface_file_name = 'SurfaceGenOutput.txt'
mesh_field = 'field0.fXXXXX' # usually the first field file, which contains the mesh
interpolation_output_name = 'interpolated_intantaneous_fields.hdf5'
rough_surface = True
build_inverse = True

data_dir = '/dir/to/fields'
output_dir = '/dir/for/output'
comm = MPI.COMM_WORLD

os.chdir(data_dir)

XYZ_inverse = None
# Extract the shape of the rough surface
if rough_surface:
    if comm.Get_rank() == 0:
        print('Extracting surface gen file')
    A, B, k, lam, beta, M, N = extract_surface_gen_file( surface_file_name)
    if comm.Get_rank() == 0:
        print('Extracted surface gen file')

# Map rough surface to smooth
if rough_surface and build_inverse:
    maxR = map_rough_to_smooth_neko(dir = data_dir, comm=comm, fname_mesh_rough=mesh_field, fname_mesh_smooth='smooth_'+mesh_field, Lz=lengthZ, Lth=lengthTh, A=A, B=B, amplitude=roughness_scaling, nModM=int(M), nModN=int(N), pipe_R=r_max, r_crit=r_crit)
    if comm.Get_rank() == 0:
        print('Mapped rough mesh to smooth - starting inverse mapping interpolation points')

xyz = None
# build intrerpolaiton points and map them into the roughness domain, if necessary
if comm.Get_rank() == 0:
    xyz, Nstruct_from_function, X_smooth, Y_smooth, Z_smooth = user_defined_interpolation_points(nelemZ=nelemZ,nelemTh= nelemTh,polyOrder= polyOrder,targetRetau= targetReTau,rmax= r_max,stretching= stretching, reynoldsnumber=Reynolds_number, roughSurface=rough_surface, Lz=lengthZ, Lth=lengthTh)
    XYZ_smooth = np.stack([X_smooth,Y_smooth,Z_smooth], axis =-1)

    if rough_surface and build_inverse:
        xyz_main, Nstruct_main, X_inverse, Y_inverse, Z_inverse = inverseSurfaceMapping_concentricPoints(XYZ_smooth, nModM=int(M),nModN=int(N), A=A, B=B, pipe_R=r_max,r_crit= r_crit, Lz=lengthZ, Lth=lengthTh, amplitude=roughness_scaling)
        print('Inverted Interpolation points')
        XYZ_inverse = np.stack([X_inverse, Y_inverse, Z_inverse], axis=-1)  # (nr, ntheta, nz, 3)
        Nstruct = [XYZ_inverse.shape[0], XYZ_inverse.shape[1], XYZ_inverse.shape[2]]
        xyz = interp_utils.transform_from_array_to_list(Nstruct[0], Nstruct[1], Nstruct[2],
            [np.transpose(X_inverse,(2,0,1)), np.transpose(Y_inverse,(2,0,1)), np.transpose(Z_inverse,(2,0,1))])
        xyz = np.ascontiguousarray(xyz)

# collect fields from the run directory and its restartX subdirectories (paths
# relative to data_dir, which is the cwd). Each restart resets its output
# numbering to the index it resumed from, so the file index is not unique across
# directories - the simulation time from the fld header is, and it also drops
# any directory that is a plain copy of another. Ordered by time.
def field_time(fname):
    return float(open(fname, 'rb').read(132).split()[7])

restart_dirs = [d for d in glob.glob('restart*') if os.path.isdir(d) and d[7:].isdigit()]
search_dirs = ['.'] + sorted(restart_dirs, key=lambda d: int(d[7:]))
seen = {}
for d in search_dirs:
    for f in sorted(glob.glob(os.path.join(d, 'field0.f*'))):
        seen.setdefault(field_time(f), os.path.normpath(f))
field_names = [seen[t] for t in sorted(seen)]
fname_mesh = 'smooth_'+mesh_field

if comm.Get_rank() == 0:
    print('field names:', field_names)
# interpolate multiple fileds by building probes at the mesh containing filed and applying them at all other fields
interpolate_multiple_fields(full_fname_mesh=fname_mesh, these_field_file_names=field_names, xyz=xyz, interpolation_output_fname=os.path.join(output_dir, interpolation_output_name), if_pass_points_to_rank0_only=True)