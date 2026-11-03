## Various plotting functions

import matplotlib.pyplot as plt
import numpy as np
import subprocess as sp
import plot_scirpts.plot_style as plot_style  # applies the shared plot styling



index = lambda k: np.s_[1:,k] # Define index, such that it can be inserted into the plot functions

def plot_uplus_yplus(yplus, UVW, u_tau, uplus_log_law, sublayer, reference_data_u, Re_tau):
    k = 0
    plt.figure(999, figsize = (10,10))
    plt.semilogx(yplus, UVW[index(k)]/u_tau, 'k', label = 'simulation')
    plt.semilogx(yplus, uplus_log_law, ':b', label = 'log law')
    plt.semilogx(sublayer, sublayer, ':r', label = 'sublayer')
    if reference_data_u is not None:
        plt.semilogx(reference_data_u[:,1], reference_data_u[:,2], ':k', label = 'reference data smooth pipe')
    plt.xlim((.5,Re_tau))
    plt.ylim((-0.5,(uplus_log_law[-1] +2)))
    plt.legend()
    plt.xlabel('y+')
    plt.ylabel('U+')
    plt.savefig('U+ over y+', dpi = 200)

def plot_rs_plus(yplus, Rij, u_tau, reference_data_rms, Re_tau):
    k = np.s_[0:3]
    plt.figure(2, figsize = (10,10))
    plt.semilogx(yplus, np.sqrt(np.abs(Rij[index(k)] /u_tau**2)) ,'k')
    plt.semilogx(yplus, np.abs(Rij[1:,3] /u_tau**2) ,'k')
    if reference_data_rms is not None:
        plt.semilogx(reference_data_rms[:,1],  reference_data_rms[:,4], ':k', label='reference')
        plt.semilogx(reference_data_rms[:,1],  reference_data_rms[:,5], ':k')
        plt.semilogx(reference_data_rms[:,1],  reference_data_rms[:,6], ':k')
        plt.semilogx(reference_data_rms[:,1],  np.abs(reference_data_rms[:,7]), ':k')
    plt.xlim((.5,Re_tau))
    plt.title('RS plus')
    plt.xlabel('y+')
    plt.savefig('RSplus', dpi = 200)

def plot_all_budgets(yplus, u_tau, delta_nu, Re_tau, Prod_ij, Diss_ij, PRS_ij, PressTrans_ij, TurbTrans_ij, ViscDiff_ij, Res_ij, reference_data_rr, reference_data_rz, reference_data_tt, reference_data_zz):
    titles = ['uu', 'vv', 'ww', 'uv']
    reference_files = [reference_data_zz, reference_data_rr, reference_data_tt, reference_data_rz]
    for i in range(4):
        k = np.s_[i]
        plt.figure(11+i, figsize = (10,10))
        plt.semilogx(yplus,Prod_ij[index(k)] /(u_tau**3/delta_nu)       ,'k', label = 'Prdocuction')
        plt.semilogx(yplus,Diss_ij[index(k)] /(u_tau**3/delta_nu)       ,'b', label = 'Dissipation')
        plt.semilogx(yplus,PRS_ij[index(k)] /(u_tau**3/delta_nu)        ,'r', label = 'PRS')
        plt.semilogx(yplus,PressTrans_ij[index(k)] /(u_tau**3/delta_nu) ,'g', label = 'Pres. Trans.')
        plt.semilogx(yplus,TurbTrans_ij[index(k)] /(u_tau**3/delta_nu)  ,'c', label = 'Turb. Trans.')
        plt.semilogx(yplus,ViscDiff_ij[index(k)] /(u_tau**3/delta_nu)   ,'grey', label = 'Vis. Diff.')
        plt.semilogx(yplus,Res_ij[index(k)] /(u_tau**3/delta_nu)        ,'-.k', label = 'Residual')

        if reference_files[i] is not None:
            plt.semilogx(reference_files[i][:,1], reference_files[i][:,2], ':k', label = 'reference')
            plt.semilogx(reference_files[i][:,1], reference_files[i][:,7], ':b')
            plt.semilogx(reference_files[i][:,1], reference_files[i][:,5], ':r')
            plt.semilogx(reference_files[i][:,1], reference_files[i][:,4], ':g')
            plt.semilogx(reference_files[i][:,1], reference_files[i][:,3], ':c')
            plt.semilogx(reference_files[i][:,1], reference_files[i][:,6], 'grey', ls = ':')

        plt.xlim((.5,Re_tau))
        plt.legend()
        plt.title('Budgets of ' + titles[i] + ' over y+')
        plt.savefig(f"budget_{i}")    


def plot_Rij(r, rmax, Rij, u_tau):
    plt.figure(20, figsize=(10,10))
    line_x = np.linspace(1,0,100)
    line_y = np.linspace(0,1,100)
    plt.plot(r/rmax, -Rij[:,3]/u_tau**2)
    plt.plot(line_x,line_y, ':k')
    plt.xlim(0,1)
    plt.title('Turbulent shear stress over r/R')
    plt.savefig("u'v' over r_R")

def plot_total_stress(r,rmax,Rij, Dij, dUidXj, u_tau, fluid_nu):
    plt.figure(21, figsize=(10,10))
    line_x = np.linspace(1,0,100)
    line_y = np.linspace(0,1,100)
    total_stress = fluid_nu*dUidXj[:,1]/u_tau**2 - Rij[:,3]/u_tau**2 -Dij[:,3]/u_tau**2
    plt.plot(r/rmax, total_stress, 'g',label = 'total stress')
    plt.plot(r/rmax, -Rij[:,3]/u_tau**2, 'b', label='Reynolds shear stress')
    plt.plot(r/rmax, fluid_nu*dUidXj[:,1]/u_tau**2, 'r', label='viscous shear stress')
    plt.plot(line_x,line_y, ':k', label = 'reference')
    mask = (r/rmax >= 0.9) & (r/rmax <= 1.0)
    coeffs = np.polyfit((r/rmax)[mask], total_stress[mask], 1)
    x_fit = np.linspace(0, 1, 100)
    plt.plot(x_fit, np.polyval(coeffs, x_fit), '-.', color='gray', label=f'linear fit (y-intercept = {coeffs[1]:.3f})')
    plt.ylim(-0.1,1.1)
    plt.yticks(np.arange(-0.05,1.05,0.05))
    plt.xlim(0,1)
    plt.legend()
    plt.title('Total shear stress over r/R')
    plt.savefig("Total shear stress over r_R")

def plot_total_stress_andDij(r,rmax,Rij, dUidXj, Dij, u_tau, fluid_nu):
    plt.figure(23, figsize=(10,10))
    line_x = np.linspace(1,0,100)
    line_y = np.linspace(0,1,100)
    total_stress = fluid_nu*dUidXj[:,1]/u_tau**2 - Rij[:,3]/u_tau**2 - Dij[:,3]/u_tau**2
    plt.plot(r/rmax, total_stress, 'g',label = 'total stress')
    plt.plot(r/rmax, -Rij[:,3]/u_tau**2, 'b', label='Reynolds shear stress')
    plt.plot(r/rmax, -Dij[:,3]/u_tau**2, 'm', label='dispersive shear stress')
    plt.plot(r/rmax, fluid_nu*dUidXj[:,1]/u_tau**2, 'r', label='viscous shear stress')
    plt.plot(line_x,line_y, ':k', label = 'reference')
    mask = (r/rmax >= 0.9) & (r/rmax <= 1.0)
    coeffs = np.polyfit((r/rmax)[mask], total_stress[mask], 1)
    x_fit = np.linspace(0, 1, 100)
    plt.plot(x_fit, np.polyval(coeffs, x_fit), '-.', color='gray', label=f'linear fit (y-intercept = {coeffs[1]:.3f})')
    plt.ylim(-0.1,1.1)
    plt.yticks(np.arange(-0.05,1.05,0.05))
    plt.xlim(0,1)
    plt.legend()
    plt.title('Total shear stress over r/R')
    plt.savefig("Rij Dij visc over r_R")
    
def plot_retau(Re_b, dir):
    import subprocess as sp
    sp.run('''awk '$3=="z" {print $4, $5}' ../../output_file_*.out > ../volflow.dat''', shell = True)
    vol = np.genfromtxt('../volflow.dat')
    avg_retau = np.mean(np.sqrt(vol[len(vol)//2:,1]/2)*Re_b)
    plt.figure(22, figsize=(10,10))
    plt.plot(vol[:,0],np.sqrt(vol[:,1]/2)*Re_b,'r-',label='pressure')
    plt.ylim(0,avg_retau+0.2*avg_retau)
    plt.xlim(0,vol[-1,0])
    plt.ylabel(r'$Re_\tau$')
    plt.xlabel(r'Time, $\delta/U_b$')
    plt.title(r'Calculated $Re_\tau$ from forcing over Time')
    plt.legend()
    plt.savefig('retau',dpi = 200)
    return avg_retau


def add_uplus_comparison(fig_num, yplus, U, u_tau, color, label):
    """Add one dataset's U+ profile to a comparison figure."""
    plt.figure(fig_num)
    plt.semilogx(yplus, U / u_tau, color=color, label=label)


def finalize_uplus_comparison(fig_num, yplus, uplus_log_law, sublayer, Re_tau, savename='compare_uplus'):
    """Add reference lines, axis labels, and save the U+ comparison figure."""
    plt.figure(fig_num)
    plt.semilogx(yplus, uplus_log_law, ':b', label='log law')
    plt.semilogx(sublayer, sublayer,   ':r', label='sublayer')
    plt.xlim(0.5, Re_tau)
    plt.ylim(-0.5, uplus_log_law[-1] + 2)
    plt.xlabel('y+')
    plt.ylabel('U+')
    plt.legend()
    plt.tight_layout()
    plt.savefig(savename, dpi=200)


def add_rs_comparison(fig_num, yplus, rij, u_tau, color, label):
    """Add one dataset's Rij profiles to a comparison figure.

    rij : dict with keys 'uu', 'vv', 'ww', 'uv' (1-D arrays).
    """
    rs_components = [('uu', 1), ('vv', 1), ('ww', 1), ('uv', -1)]
    rs_styles     = ['-', '--', ':', '-.']
    rs_tex        = [r'$R_{uu}^+$', r'$R_{vv}^+$', r'$R_{ww}^+$', r'$-R_{uv}^+$']
    plt.figure(fig_num)
    for (key, sign), ls, tex in zip(rs_components, rs_styles, rs_tex):
        plt.semilogx(yplus, sign * rij[key] / u_tau**2,
                     color=color, linestyle=ls, label=f'{label} {tex}')


def finalize_rs_comparison(fig_num, Re_tau, savename='compare_Rij'):
    """Add axis labels and save the Rij comparison figure."""
    plt.figure(fig_num)
    plt.xlabel('y+')
    plt.ylabel(r'$R_{ij}^+$')
    plt.xlim(0.5, Re_tau)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(savename, dpi=200)
