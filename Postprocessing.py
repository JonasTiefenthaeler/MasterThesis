import time
script_start_time = time.time()
from mpi4py import MPI
import numpy as np
import os
import h5py


from functions import *
from plotting import *

from pysemtools.postprocessing.file_indexing import index_files_from_folder
from pysemtools.postprocessing.statistics.time_averaging import average_field_files
from pysemtools.postprocessing.statistics.RS_budgets import compute_and_write_additional_pstat_fields
import pysemtools.interpolation.utils as interp_utils
import json



naming_scheme = 'naming_scheme'

case_name = 'case_name'
dir = f"/path/for{case_name}/data_output" # path for data output
data_path = f"/path/for/data/input/{case_name}" #define path for input data
surface_file_name = 'SurfaceGenOutput.txt' # filename of file, which defines surface

targetReTau = 230   # Alter, such that the interpolation point generatoion reproduces the radial grid stored in pstat3d_format.hdf5 (nr_bulk=98) when do_interpolation=False
Reynolds_number = 2650
## define the mesh parameters for interpolation points. Can but dosent need to match simulation mesh. If possible with reasonable compute effort, a match is recommended.
nelemZ = 80
nelemTh = 40
polyOrder = 7

# define sin shape of pipe surface, in case a sinusoidal pipe is simulated (However, not tested, if it works properly!!) 
nmodZ = 16
nmodTh = 8 #8
kplus = 10
r_crit = 0.5
roughness_scaling = 1.0

# define size of pipe domain
r_max = 1.0
lengthZ = 8*np.pi
lengthTh = 2*np.pi
stretching = 1.05

sin_surface = False

rough_surface = True
do_interpolation = False    # Needs to set to true, if first postproessesing of that dataset. To save on postprocessing time, this can be set to false, if the interpolation has already been done. If so, the necessary data is read from the pstat_3d.hdf5 file produced in the interpolation process

# NOTE: take a look at the masters thesis for an explanation of the idea behind the two switches
do_transform_into_smooth = True
build_inverse = True

fname_budget = 'pstat3d_format.hdf5'                                # name of the hdf5 file containing calculated budgets
fname_averaged = 'averaged_interpolation_fields.hdf5'

stats_files = ["fluid_stats-0.fld", "fluid_stats-1.fld", "fluid_stats-2.fld", "fluid_stats-3.fld", "fluid_stats-4.fld", "fluid_stats-5.fld", "fluid_stats-6.fld", "fluid_stats-7.fld", "fluid_stats-8.fld"] # Enables the postprocessing of multiple stats files, which occurs if a simulation consists of multiple restarts
stats_start_time = 300.0   # = "start_time" of the fluid_stats sampler in the .case
os.chdir(dir) #change path such that the written files are written to dir path
comm = MPI.COMM_WORLD
if comm.Get_rank() == 0:
    xyz, Nstruct_from_function, X_smooth, Y_smooth, Z_smooth = user_defined_interpolation_points(nelemZ=nelemZ,nelemTh= nelemTh,polyOrder= polyOrder,targetRetau= targetReTau,rmax= r_max,stretching= stretching, reynoldsnumber=Reynolds_number, roughSurface=rough_surface, Lz=lengthZ, Lth=lengthTh)
else:
    xyz = Nstruct_from_function = X_smooth = Y_smooth = Z_smooth = None

if comm.Get_rank() == 0:

    grid = pv.StructuredGrid(X_smooth, Y_smooth, Z_smooth)
    grid.save('InterpolationPoints_smooth.vts')
    XYZ_smooth = np.stack([X_smooth[2:],Y_smooth[2:],Z_smooth[2:]], axis =-1)
    XYZ_wall = np.stack([X_smooth[:2], Y_smooth[:2], Z_smooth[:2]], axis = -1) # Creates extra set of interpolation points, that follows the wall, such that the wall friction can be computed. The wall points are getting added to the interpolation points later
if rough_surface:
    A, B, k, lam, beta, M, N = extract_surface_gen_file('../'+ surface_file_name)

# Interpolates data from DNS mesh onto generated pointcloud
if do_interpolation:
    from pysemtools.postprocessing.statistics.RS_budgets import interpolate_all_stat_and_pstat_fields_onto_points
    
    consolidated_index = build_consolidated_stats_index(comm, data_path, stats_files, stats_start_time, output_folder="./")   # one time-sorted index over all restarts
    average_field_files(comm, field_index_name=consolidated_index, output_folder="./", output_name="fluid_stats_averaged", output_batch_t_len=500000, mesh_index="0", dtype=np.single, rel_tol=0.05, output_word_size=4, write_mesh=True)    # time-weighted average over all restarts into one file

    averaged_field_name = "fluid_stats_averaged0.f00000"
    fname_mesh = averaged_field_name
    fname_mean = averaged_field_name
    fname_stat = fname_mean
    which_code = "neko"
    nek5000_stat_type = ""
    if_do_dssum_on_derivatives = False

    ## compute field files cotaining dervatives
    compute_and_write_additional_pstat_fields(dir, fname_mesh, fname_mean, fname_stat, if_write_mesh=True, which_code=which_code, nek5000_stat_type=nek5000_stat_type, if_do_dssum_on_derivatives=if_do_dssum_on_derivatives)   # Compute additional statistics that are needed to obtain budgets

    ## transform sin surface into smooth
    if do_transform_into_smooth:
        averaged_field_name_smooth = 'smooth_'+averaged_field_name
        fname_mesh = averaged_field_name_smooth
        fname_mean = averaged_field_name_smooth
        fname_stat = fname_mean
        if sin_surface:
            map_sin_to_smooth(dir=dir, fname_mesh_orig=averaged_field_name,fname_mesh_deform= averaged_field_name_smooth,pipe_R= r_max,r_crit=r_crit,kplus= kplus,Re_tau= targetReTau,lengthZ= lengthZ,nmodZ= nmodZ)
        if rough_surface:
            max_R = map_rough_to_smooth_neko(dir = dir, comm=comm, fname_mesh_rough=averaged_field_name, fname_mesh_smooth= averaged_field_name_smooth, Lz=lengthZ, Lth=lengthTh, A=A, B=B, amplitude=roughness_scaling, nModM=int(M), nModN=int(N), pipe_R=r_max, r_crit=r_crit)
            if comm.Get_rank() == 0:
                print('max:R: ', max_R)

    ### Build interpolation points
    if comm.Get_rank() == 0:    # call function to build interpolation points
        
        if build_inverse:
            xyz_main, Nstruct_main, X_inverse, Y_inverse, Z_inverse = inverseSurfaceMapping_concentricPoints(XYZ_smooth, nModM=int(M),nModN=int(N), A=A, B=B, pipe_R=r_max,r_crit= r_crit, Lz=lengthZ, Lth=lengthTh, amplitude=roughness_scaling)

            XYZ_inverse = np.stack([X_inverse, Y_inverse, Z_inverse], axis=-1)  # (nr, ntheta, nz, 3)
            XYZ_full = np.concatenate([XYZ_wall, XYZ_inverse], axis=0)          # (nr+2, ntheta, nz, 3)
            X_full, Y_full, Z_full = extract_X_Y_Z(XYZ_full)                    # each (nr+2, ntheta, nz)
            Nstruct = [XYZ_full.shape[0], XYZ_full.shape[1], XYZ_full.shape[2]]
            xyz = interp_utils.transform_from_array_to_list(Nstruct[0], Nstruct[1], Nstruct[2],
                [np.transpose(X_full,(2,0,1)), np.transpose(Y_full,(2,0,1)), np.transpose(Z_full,(2,0,1))])
        xyz = np.ascontiguousarray(xyz)
        pv.PolyData(xyz).save('InterpolationPoints_all.vtp')
    else:
        xyz = None

    ## Do the interpolation.  Some options and paths are set for the interpolation function
    if_do_dssum_before_interp = False           # whether to do dssum before interpolation
    if_create_boundingBox_for_interp = False    #wheter to create a bounding box around the point cloud (useful for cases where there is a cluster of points somewhere)
    if_pass_points_to_rank0_only = True         # pass all popints to rank0 only. This is to avoind passing duplicate points
    interpolate_all_stat_and_pstat_fields_onto_points(dir, fname_mesh, fname_mean, fname_stat, xyz, which_code=which_code, nek5000_stat_type=nek5000_stat_type, if_do_dssum_before_interp=if_do_dssum_before_interp, if_create_boundingBox_for_interp=if_create_boundingBox_for_interp, if_pass_points_to_rank0_only=if_pass_points_to_rank0_only)  # Perform the interpolation with the generated points
    If_convert_to_single = True       # converts the statistics/budgets data into single precision
    If_average = True
    Nstruct = Nstruct_from_function  # [nr+2, ntheta, nz]
    interpolation_averaging_func = av_func_interp
    from pysemtools.postprocessing.statistics.RS_budgets_interpolatedPoints_notMPI import read_interpolated_stat_hdf5_fields
    from pysemtools.postprocessing.statistics.RS_budgets_interpolatedPoints_notMPI import calculate_budgets_in_Cartesian
    if comm.Get_rank() == 0:
        read_interpolated_stat_hdf5_fields(dir, Reynolds_number, If_average, If_convert_to_single,  Nstruct, interpolation_averaging_func,  output_fname = fname_averaged)
        calculate_budgets_in_Cartesian(path_to_files=dir, input_filename=fname_averaged, output_filename=fname_budget)    # Calculate all Budgets


## use the interpolated data and calculate budgets from it. first define some input data for the functions

if comm.Get_rank() == 0:

    ## extract data from the pstat_file that contains all data
    with h5py.File(fname_budget, "r") as input_file:
            XYZ             = np.array(input_file['XYZ'][:])
            UVW             = np.array(input_file['UVW'][:])
            P               = np.array(input_file['P'][:])
            Rij             = np.array(input_file['Rij'][:])  
            dUidXj          = np.array(input_file['dUidXj'][:])
            Prod_ij         = np.array(input_file['Prod_ij'][:]) 
            Diss_ij         = np.array(input_file['Diss_ij'][:])
            PRS_ij          = np.array(input_file['PRS_ij'][:]) 
            Conv_ij         = np.array(input_file['Conv_ij'][:]) 
            PressTrans_ij   = np.array(input_file['PressTrans_ij'][:]) 
            TurbTrans_ij    = np.array(input_file['TurbTrans_ij'][:]) 
            VelPressGrad_ij = np.array(input_file['VelPressGrad_ij'][:]) 
            ViscDiff_ij     = np.array(input_file['ViscDiff_ij'][:])
            Rer             = np.float64(input_file['Rer_here'])
      

    ## create functions to do some necessary linear algebra and use them to rotate the tensors
    from pysemtools.postprocessing.statistics.tensor_manipulations import rotate_6c2D_tensor
    from pysemtools.postprocessing.statistics.tensor_manipulations import rotate_vector
    from pysemtools.postprocessing.statistics.tensor_manipulations import define_rotation_tensor_from_vectors
    print('NAN in XYZ: ', np.isnan(XYZ).any()) # A dubug print

    N_wall_r =  2 # The datapoints have data, that match the rough pipe. A set of two data points is appended to the array, that cointain wall-fitted data, that is used to calculate the wall friction.
    # Wall data: smooth coords → map to rough for friction calculation
    XYZ_wall = XYZ[:N_wall_r, ...].copy()
    XYZ_wall_rough = XYZ_wall.copy()
    if do_transform_into_smooth:
        if sin_surface:
            XYZ_wall_rough = map_smooth_to_sin(XYZ_wall_rough, targetReTau, lengthZ, nmodZ, kplus, pipe_R=r_max, r_crit=r_crit)
        if rough_surface:
            start_time = time.time()
            XYZ_wall_rough = map_smooth_to_rough(XYZ=XYZ_wall_rough, A=A, B=B, nModM=int(M), nModN=int(N), Lz=lengthZ, Lth=lengthTh, pipe_R=r_max, r_crit=r_crit, amplitude=roughness_scaling)
            R_melt, k_melt = calculate_meltdown_radius(XYZ_wall_rough, pipe_R=r_max)

    # Main data: already in rough coords from inverseSurfaceMapping
    XYZ_main = XYZ[N_wall_r:, ...]
    XYZ_plot = XYZ_smooth  # smooth coords, correctly ordered (nr, ntheta, nz, 3)

    X, Y, Z = extract_X_Y_Z(XYZ_main)
    grid = pv.StructuredGrid(X, Y, Z)
    grid.save("XYZ.vts")

    # Split field variables. Removes wall points for further computations
    Rij_wall    = Rij[:N_wall_r, ...]
    dUidXj_wall = dUidXj[:N_wall_r, ...]
    P_wall      = P[:N_wall_r, ...]

    UVW             = UVW[N_wall_r:, ...]
    P               = P[N_wall_r:, ...]
    Rij             = Rij[N_wall_r:, ...]
    dUidXj          = dUidXj[N_wall_r:, ...]
    Prod_ij         = Prod_ij[N_wall_r:, ...]
    Diss_ij         = Diss_ij[N_wall_r:, ...]
    PRS_ij          = PRS_ij[N_wall_r:, ...]
    Conv_ij         = Conv_ij[N_wall_r:, ...]
    PressTrans_ij   = PressTrans_ij[N_wall_r:, ...]
    TurbTrans_ij    = TurbTrans_ij[N_wall_r:, ...]
    VelPressGrad_ij = VelPressGrad_ij[N_wall_r:, ...]
    ViscDiff_ij     = ViscDiff_ij[N_wall_r:, ...]

    # Coordinate system and tensor rotation based on main (rough) grid
    print('Computing transformation matrices')
    v1,v2,v3        = define_cylindrical_coord_system_vectors(XYZ_main)
    Q               = define_rotation_tensor_from_vectors(v1,v2,v3)
    Qtrans          = np.transpose(Q,axes=(0,1,2,4,3))

    print('Rotating tensors')
    UVW                             = rotate_vector(UVW, Qtrans, Q)
    Rij                             = rotate_6c2D_tensor(Rij,Qtrans,Q)
    dUidXj_test, dUidXj_3x3_test    = rotate_9c2D_tensor(dUidXj, Qtrans, Q)
    Prod_ij                         = rotate_6c2D_tensor(Prod_ij,Qtrans,Q)
    Diss_ij                         = rotate_6c2D_tensor(Diss_ij,Qtrans,Q)
    PRS_ij                          = rotate_6c2D_tensor(PRS_ij,Qtrans,Q)
    Conv_ij                         = rotate_6c2D_tensor(Conv_ij,Qtrans,Q)
    PressTrans_ij                   = rotate_6c2D_tensor(PressTrans_ij,Qtrans,Q)
    TurbTrans_ij                    = rotate_6c2D_tensor(TurbTrans_ij,Qtrans,Q)
    VelPressGrad_ij                 = rotate_6c2D_tensor(VelPressGrad_ij,Qtrans,Q)
    ViscDiff_ij                     = rotate_6c2D_tensor(ViscDiff_ij,Qtrans,Q)

    UVW_local = np.zeros_like(Rij)

    UVW_local[...,0] = (UVW[...,0] * UVW[...,0])
    UVW_local[...,1] = (UVW[...,1] * UVW[...,1])
    UVW_local[...,2] = (UVW[...,2] * UVW[...,2])
    UVW_local[...,3] = (UVW[...,0] * UVW[...,1])
    UVW_local[...,4] = (UVW[...,0] * UVW[...,2])
    UVW_local[...,5] = (UVW[...,1] * UVW[...,2])

    # Wall friction from rough-mapped wall data
    print('Compute wall friction')
    mean_force_wall, fluid_nu, mean_tauW, mean_presDrag, tauWall_field, presDrag_field = calculate_friction(XYZ=XYZ_wall_rough, Rij=Rij_wall, dUidXj=dUidXj_wall, P=P_wall, Rer=Rer)
    print('mean force wall: ', mean_force_wall)
    u_tau           = np.sqrt(mean_force_wall) 
    delta_nu        = fluid_nu / u_tau
    Re_tau          = 1./delta_nu

    # calculate y_plus values
    yplus, r = calculate_yplus(XYZ = XYZ_plot, delta_nu=delta_nu, r_max=r_max)
   
    ## calculate the average of the fileds, to be able to plot 1d data
    print('Averaging fileds')
    UVW                 = avg_from_3D_to_1D(UVW, return_counts=False)
    P                   = avg_from_3D_to_1D(P, return_counts=False)
    Rij                 = avg_from_3D_to_1D(Rij, return_counts=False)
    dUidXj_test         = avg_from_3D_to_1D(dUidXj_test, return_counts=False)
    Prod_ij             = avg_from_3D_to_1D(Prod_ij, return_counts=False)
    Diss_ij             = avg_from_3D_to_1D(Diss_ij, return_counts=False)
    PRS_ij              = avg_from_3D_to_1D(PRS_ij, return_counts=False)
    Conv_ij             = avg_from_3D_to_1D(Conv_ij, return_counts=False)
    PressTrans_ij       = avg_from_3D_to_1D(PressTrans_ij, return_counts=False)
    TurbTrans_ij        = avg_from_3D_to_1D(TurbTrans_ij, return_counts=False)
    VelPressGrad_ij     = avg_from_3D_to_1D(VelPressGrad_ij, return_counts=False)
    ViscDiff_ij         = avg_from_3D_to_1D(ViscDiff_ij, return_counts=False)
    Res_ij              = Prod_ij + Diss_ij + PRS_ij + Conv_ij + PressTrans_ij + TurbTrans_ij + ViscDiff_ij

    UVW_DA              = avg_from_3D_to_1D(UVW_local, return_counts=False)   # intrinsic average of the local products

    # compute dispersive stress. Look at thesis for explanation
    Dij = np.empty_like(UVW_DA)
    Dij[...,0] = UVW_DA[...,0] - UVW[...,0]*UVW[...,0]   # uu
    Dij[...,1] = UVW_DA[...,1] - UVW[...,1]*UVW[...,1]   # vv
    Dij[...,2] = UVW_DA[...,2] - UVW[...,2]*UVW[...,2]   # ww
    Dij[...,3] = UVW_DA[...,3] - UVW[...,0]*UVW[...,1]   # uv
    Dij[...,4] = UVW_DA[...,4] - UVW[...,0]*UVW[...,2]   # uw
    Dij[...,5] = UVW_DA[...,5] - UVW[...,1]*UVW[...,2]   # vw

    U_bulk = calculate_bulk_velocity_U(UVW, r, r_max)
        
    ## calculate the log_law of turbulent U+
    karman_const = 0.41
    B_loglaw = 5.2
    yplus = yplus[1:]
    uplus_log_law, sublayer = calculate_turb_law(karman_const, B_loglaw, yplus)
    reference_data_1, reference_data_2, reference_data_3, reference_data_rr_budget, reference_data_rz_budget, reference_data_tt_budget, reference_data_zz_budget = extract_reference_data_360ReT()

    # write results for the whole domain to txt files, for easier plotting later
    write_tauWall_field_to_txt(os.path.join(dir, f'tau_wallField_{naming_scheme}.txt'), tauWall_field)
    write_tauWall_field_to_txt(os.path.join(dir, f'pres_drag_{naming_scheme}.txt'), presDrag_field)
    write_plot_data_to_txt(
        os.path.join(dir, f'{naming_scheme}.txt'),
        scalars={
            'Re_tau':   float(Re_tau),
            'u_tau':    float(u_tau),
            'delta_nu': float(delta_nu),
            'r_max':    float(r_max),
            'fluid_nu': float(fluid_nu),
            'mean_tauW': float(mean_tauW),
            'mean_presDrag': float(mean_presDrag)
        },
        yplus_table={
            'yplus':          yplus,
            'U':              UVW[1:, 0],
            'uplus_log_law':  uplus_log_law,
            'Rij_uu':         Rij[1:, 0],
            'Rij_vv':         Rij[1:, 1],
            'Rij_ww':         Rij[1:, 2],
            'Rij_uv':         Rij[1:, 3],
            'Dij_uu':         Dij[1:, 0],
            'Dij_vv':         Dij[1:, 1],
            'Dij_ww':         Dij[1:, 2],
            'Dij_uv':         Dij[1:, 3],
            'Prod_uu':        Prod_ij[1:, 0],
            'Prod_vv':        Prod_ij[1:, 1],
            'Prod_ww':        Prod_ij[1:, 2],
            'Prod_uv':        Prod_ij[1:, 3],
            'Diss_uu':        Diss_ij[1:, 0],
            'Diss_vv':        Diss_ij[1:, 1],
            'Diss_ww':        Diss_ij[1:, 2],
            'Diss_uv':        Diss_ij[1:, 3],
            'PRS_uu':         PRS_ij[1:, 0],
            'PRS_vv':         PRS_ij[1:, 1],
            'PRS_ww':         PRS_ij[1:, 2],
            'PRS_uv':         PRS_ij[1:, 3],
            'PressTrans_uu':  PressTrans_ij[1:, 0],
            'PressTrans_vv':  PressTrans_ij[1:, 1],
            'PressTrans_ww':  PressTrans_ij[1:, 2],
            'PressTrans_uv':  PressTrans_ij[1:, 3],
            'TurbTrans_uu':   TurbTrans_ij[1:, 0],
            'TurbTrans_vv':   TurbTrans_ij[1:, 1],
            'TurbTrans_ww':   TurbTrans_ij[1:, 2],
            'TurbTrans_uv':   TurbTrans_ij[1:, 3],
            'ViscDiff_uu':    ViscDiff_ij[1:, 0],
            'ViscDiff_vv':    ViscDiff_ij[1:, 1],
            'ViscDiff_ww':    ViscDiff_ij[1:, 2],
            'ViscDiff_uv':    ViscDiff_ij[1:, 3],
            'Res_uu':         Res_ij[1:, 0],
            'Res_vv':         Res_ij[1:, 1],
            'Res_ww':         Res_ij[1:, 2],
            'Res_uv':         Res_ij[1:, 3],
        },
        r_table={
            'r':      r,
            'Rij_uv': Rij[:, 3],
            'Dij_uv': Dij[:, 3],
            'dUdr':   dUidXj_test[:, 1],
        },
        sublayer=sublayer,
    )
    ## plot the data
    os.chdir(dir + '/plots')
    print('Creating plots')
    plot_uplus_yplus(yplus, UVW, u_tau, uplus_log_law, sublayer, reference_data_1, Re_tau)
    plot_rs_plus(yplus, Rij, u_tau, reference_data_1, Re_tau)
    plot_all_budgets(yplus, u_tau, delta_nu, Re_tau, Prod_ij, Diss_ij, PRS_ij, PressTrans_ij, TurbTrans_ij, ViscDiff_ij, Res_ij, reference_data_rr=reference_data_rr_budget, reference_data_rz=reference_data_rz_budget, reference_data_tt=reference_data_tt_budget, reference_data_zz=reference_data_zz_budget)
    plot_Rij(r, r_max, Rij, u_tau)
    plot_total_stress(r = r, rmax=r_max, Rij = Rij, Dij = Dij, dUidXj=dUidXj_test, u_tau=u_tau, fluid_nu=fluid_nu)
    plot_total_stress_andDij(r = r, rmax=r_max, Rij=Rij, dUidXj=dUidXj_test, Dij=Dij, u_tau=u_tau,fluid_nu=fluid_nu )
    avg_retau = plot_retau(dir = dir, Re_b =Reynolds_number)

    if comm.Get_rank() == 0:
        print("delta_nu: ", delta_nu)
        print("u_tau: ", u_tau)
        print("fluid nu: ", fluid_nu)
        print("Re_tau: ", Re_tau)
        print("yplus: ", yplus)
        print("Prod_ij.shape: ", Prod_ij.shape)
        print('avg retau: ', avg_retau)
        print('difference between calculated and extracted retau in percent: ',(Re_tau/avg_retau-1)*100 )
    
    with open('PP_data.txt','w') as dat:
        dat.write(f'Calculated Retau from wall friction: {Re_tau}\n')
        dat.write(f'Extracted Retau from forcing: {avg_retau}\n')
        dat.write(f'Difference between calculated and extracted Retau in percent: {(Re_tau/avg_retau-1)*100}\n')
        dat.write(f'delta_nu: {delta_nu}\n')
        dat.write(f'u_tau: {u_tau}\n')
        dat.write(f'fluid_nu: {fluid_nu}\n')
        dat.write(f'yplus: {yplus}\n')
        if do_transform_into_smooth and rough_surface:
            dat.write(f'meltdown radius R_melt: {R_melt}\n')
            dat.write(f'meltdown height k_melt: {k_melt}\n')
        dat.write(f'Time taken to do Postprocessing in seconds: {time.time() - script_start_time}\n')
    print('time taken for complete PP script: ', time.time() - script_start_time)

