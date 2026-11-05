# MasterThesis
Relevant code of my Master Thesis "Direct numerical simulation of turbulent flow in rough pipes".

The goal is to proved a code framework to rebuild the simulation cases using some infromation of the thesis.
Within the python scripts, some paths and some other values need to be adjusted, according to the individual case.
The necessary steps are:

    1.  Using "parameterOptimzation.py" uses the user defined surface parameters to finde suitable values for  the random distribution and power spectrum, such that the surface generatrion algortihm produces the desired surface. Final output is a .txt file, that contains the fourier coefficient matrices A and B, with which the surface is recreated in Neko
    2. Run the Neko simulation with your rough pipe. Use the user file "rough_pipe.f90" and a mesh of a smooth pipe for that. The "SurfaceGenOutput.txt" file has to be in the same directory as the neko user file, such that the A and B matices can be extracted. You can change the initial condition in the neko user file as you like.
    3. Postprocess the data produced by the DNS. Use "Postprocessing.py". A few data paths and values have to be altered according to your case. Take a good look at the postprocessing script! It is mostly based on the "budgets" example of the PySEMTools examples. It was however necessary to do several changes. The script automatically produces some of the standard plots and writes them to the defiend outputDirectory/plots. Additionally, a txt file is produced, which stores the relevant arrays, which may be used later for additional plots, without the necessity to rerun the postprocessing
    4. (optional) Use the functions in "plotting.py" to produce plots as you like. Use the generated .txt file produced by the postprocessing.