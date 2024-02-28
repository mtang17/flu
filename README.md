# Sequential immunization with influenza vaccines broadens antibody responses - computational simulations

This repository contains original code created for the study titled “Repeated vaccination with homologous influenza hemagglutinin broadens human antibody responses to unmatched flu viruses” by Deng, Tang, Chakraborty, Lingwood (2024). 

The main code “simulation_code.py” is written so that multiple simulations in parallel can be submitted to a cluster using Slurm job arrays. The input for the simulations are read from each line of a text file. The text files provided in the “inputs” folder contain a header line followed by simulation inputs with one line per simulation. The output is saved as a pickle data file (.pkl) in the location specified in “helperfunctions.py”.

To reproduce the results from the study:
1. Modify getFileLocation in “helperfunctions.py” to set the location where data files will be stored and retrieved form.
2. Run simulations using input files provided in the “inputs” folder. The files are named such that all files with ***_VaxX (where *** is a descriptor and X is a number 1-4) correspond to the same parameters. Allow the previous vaccine simulation to finish before running the next vaccine simulation (about 5 hours per simulation). A sample script file “flu.sh” is provided. Modify the partition, mail-user, array, and input text file as appropriate. 
3. The output pickle data files can be analyzed using the JupyterNotebook “simulation_results.ipynb”.
