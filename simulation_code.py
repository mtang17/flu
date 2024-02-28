#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import time
import random
import numpy as np
import pickle
import gc
import ast
from numpy.random import multivariate_normal
from scipy.optimize import fsolve
from scipy.sparse import csr_matrix
from copy import deepcopy
from helperfunctions import *

# FOR MORE DETAILS ON PARAMETERS, SEE initializeParameters IN helperfunctions.py
    
def getnumTcells(param):
    """
    Summary:
     Returns the non-dimensionalized availability of helper T cells 
     over time in a GC

    Outputs:
     numTcells: array containing number of T cells over time

    Inputs:
     param: parameters struct
    """
    tmax = param['tmax']
    d_Tfh = 0.01
    dt = param['dt']
    tspan = np.arange(0, tmax + dt, dt)
    numTcells = np.zeros(shape = tspan.shape)
    numTmax = param['numTmax']
    if tmax <= 14:
        numTcells = numTmax * tspan / 14
    else:
        d14idx = round(14 / dt + 1)
        numTcells[:d14idx] = numTmax * tspan[:d14idx] / 14
        for i in range(d14idx, len(tspan)):
            numTcells[i] = numTcells[i-1] * np.exp(-d_Tfh * dt)
    return numTcells
        
    
def epitopeMasking(agconc, abconc, Ka, param):
    """
    Calculates free antigen concentrations considering epitope masking (or lack
    thereof) based on the current Ag and Ab concentration, Ab affinity

    Output: 
      agconc_Epmask: 2x3 array; Antigen concentration after epitope masking 
                     Dim1: soluble, IC-FDC
                     Dim2: Dominant (ep 1), Subdominant1 (ep 2), Subdominant2 (ep 3) epitope

    Inputs: 
      agconc: 1x4 vector; Concentrations of Antigens; 
                          soluble, IC1, IC2, IC3 (ICs are separted based on epitope)
      abconc: 3x3 vector; Concentration of antibodies
              Dim1 - IgM(natural),IgM(immune),IgG; Dim2-Epitopes
      Ka: 3x3 vector; Antibody binding affinities (Ka)
      param: parameter struct
    """
    masking = param['masking']
    q12 = param['q12']
    q13 = param['q13']
    q23 = param['q23']
    agconc_Epmask = np.zeros(shape=(2, param['n_ep']))
    if sum(agconc) == 0:
        return agconc_Epmask

    sumconc = np.sum(abconc, axis=0)
    sumKa = np.sum(abconc * Ka, axis=0)

    # If no masking, overlap will be zero.
    overlap = [[1, q12, q13], [q12, 1, q23], [q13, q23, 1]]
    abconc = np.dot(sumconc, overlap)
    Ka_avg = np.dot(sumKa, overlap) / abconc

    # Equilibrium Receptor-Ligand Binding L+R->IC
    L = abconc / 5  # Ab amount for each epitope, say there are 5 epitopes within each dominance class. Unimportant for qualitative results
    R = sum(agconc) # Total Ag concentration
    IC = (R + L + 1/Ka_avg - np.sqrt(np.square(R + L + 1/Ka_avg) - 4 * R * L)) / 2 #bound antigen
    IC = np.nan_to_num(IC, 0)
    
    agconc_Epmask[0,:] = (sum(agconc) - masking*IC) * (agconc[0] / sum(agconc))       # soluble
    agconc_Epmask[1,:] = (sum(agconc) - masking*IC) * (sum(agconc[1:])) / sum(agconc) # FDC
    
    return agconc_Epmask


def getDosingParameters(param):
    """
    Summary:
    Initialize the antigen and antibody concentrations, 
    antibody affinity, and the parameters that define the dosing profile

    Outputs:
      agconc: 1x4 vector; Concentrations of Antigens; 
                          soluble, IC1, IC2, IC3
      abconc: 3x4 vector; Concentration of antibodies
              Dim1 - IgM(natural),IgM(immune),IgG; Dim2-Epitopes
      Ka: 3x4 vector; Antibody binding affinities (Ka)
      param: parameter struct

    Input:
      param: parameter struct
    """
    
    T, k, numshot = param['T'], param['k'], param['numshot']
    agconc = np.zeros(param['n_ep'] + 1)
    abconc = np.zeros(shape = (3, param['n_ep']))
    abconc[0,:] = param['IgM0'] / param['n_ep']
    Ka = 1e-3 * np.ones(shape = (3, param['n_ep'])) # nM-1
    
    if T == 0: # bolus
        param['F0'] = 0
        param['dose_t'] = 0
        param['dose'] = param['Ag0']
    elif T > 0:
        raise ValueError('T > 0: slow delivery not used in this study')
        
    return agconc, abconc, Ka, param


def getNaiveBcells(param):
    """
    Summary
     Initialize the naive B cells with their lineages, targets, affinities,
     and mutational fitness landscapes

    Outputs
     naiveBcells: 200 x 2210 x 8 array
      Dim 1 - GCs, Dim 2 - B cells, Dim 3 - Properties
      Properties: linegae, target, vax strain affinity, variant1 affinity, variant2 affinity, activated time
     mutations - 1x3 cell array of 200 x 2210 x 80 array 
                 Dim1 - GCs; Dim2 - B cells; Dim3 - Residues
                 Effects of mutations against vax strain (strain 1), variant1 (strain 2), variant2 (strain 3)


    Inputs
     param: parameter struct 
    """
    M_GC = param['M_GC']
    E1h = param['E1h']
    dE12 = param['dE12']
    dE13 = param['dE13']
    p2 = param['p2']
    p3 = param['p3']
    f0 = param['f0']
    rho = param['rho']
    n_var = param['n_var']
    n_ep = param['n_ep']
    NaiveMax = param['NaiveMax']
    MemoryMax = param['MemoryReentryMax']
    n_res = param['n_res'] # number of residues
    dE = 0.2               # class size
    classnum = 11          # number of class bins
    N = 2000               # number of naive precursors
    
    # Find out number of B cells in each class
    maxclasses = np.around(np.array([E1h-f0, E1h-f0-dE12, E1h-f0-dE13]) / dE) + 1
    # Bin at which the value of frequency is 1 for dominant/subdominant cells
    fitnessarray = np.linspace(f0, f0 + 2, classnum) # 6 to 8 with interval of 0.2
    r = np.zeros(n_ep) # slopes of the geometric distribution
    for i in range(0, n_ep):
        if maxclasses[i] > 1:
            func = lambda x: N - (x ** maxclasses[i] - 1) / (x - 1)
            r[i] = fsolve_mult(func, guess = 1.1)
        else:
            r[i] = N

    # 3 x 11 array, number of naive B cells in each fitness class
    naivebcellsarr = np.zeros(shape = (n_ep, classnum))
    p = [(1-p2-p3), p2, p3]
    for i in range(0, n_ep):
        if maxclasses[i] > 1:
            naivebcellsarr[i, :classnum] = p[i] * r[i] ** (maxclasses[i] - (np.arange(classnum) + 1))
        elif maxclasses[i] == 1:
            naivebcellsarr[i, 0] = p[i] * N
            
    # Array of naive B cells
    # Dim 1 - GC number; Dim 2 - Lineage number
    # Dim 3 - properties (linegae, target, WT affinity, variant1 affinity, variant2 affinity, activated time)
    naivebcells = np.zeros(shape = (M_GC, NaiveMax + MemoryMax, param['naivefieldnum']))

    #Calculate change in affinity for affinity-changing mutations using correlated log-normal distribution
    MutationPDF = param['MutationPDF'] #param for distributions
    Sigma = [[]] * n_ep
    for ep in range(n_ep):
        # Create correlation matrix between vax strain and variants for each epitope
        if ep == 0: #dom
            Sigma[ep] = np.array([[1, 0.4, 0.4], [0.4, 1, 0], [0.4, 0, 1]])
        else:
            Sigma[ep] = np.diag(np.ones(n_var))
            for var in range(n_var-1): #non-WT var
                rho_val = rho[ep-1][var] #each row of matrix corresponds to a different epitope
                Sigma[ep][0][var+1] = rho_val
                Sigma[ep][var+1][0] = rho_val
    mutations = [[]] * 3
    mutations[0] = np.zeros(shape = (M_GC, NaiveMax + MemoryMax, n_res)) # Affinity change to Vax Strain
    mutations[1] = np.zeros(shape = (M_GC, NaiveMax + MemoryMax, n_res)) # Affinity change to Variant1
    mutations[2] = np.zeros(shape = (M_GC, NaiveMax + MemoryMax, n_res)) # Affinity change to Variant2
    
    # For each GC, initialize B cells
    for i in range(M_GC):
        randu = np.random.uniform(size = naivebcellsarr.shape)
        naivebcellsint = np.floor(naivebcellsarr + randu).astype(int)
        
        # Stochastically round up or down the frequency to get integer numbers
        # Indexing takes care of the epitope type
        idx = 0
        for ep_type in range(n_ep):
            for j, fitness in enumerate(fitnessarray):
                idx_new = idx + naivebcellsint[ep_type, j]
                naivebcells[i, idx: idx_new, 0] = np.arange(idx, idx_new) + 1 # Lineage
                naivebcells[i, idx: idx_new, 1] = ep_type + 1                 # Target
                naivebcells[i, idx: idx_new, 2] = fitness                     # Vax Strain aff
                naivebcells[i, idx: idx_new, 3] = f0                          # Variant1 aff
                naivebcells[i, idx: idx_new, 4] = f0                          # Variant2 aff

                mu = np.array([0]*n_var)
                sigma = MutationPDF[1] ** 2 * Sigma[ep_type]
                num = (idx_new - idx) * n_res
                X = MutationPDF[0] + multivariate_normal(mu, sigma, num)
                dE = np.log10(np.exp(1)) * (np.exp(X) - MutationPDF[2])
                for var in range(n_var):
                    reshaped = np.reshape(dE[:, var], (idx_new - idx, n_res), order = 'F')
                    mutations[var][i, idx:idx_new,:] = reshaped
                idx = idx_new
                
    return naivebcells, mutations




def naiveFlux(naiveBcells, conc, param):
    """
    Summary:
     Finds the logical indices of naive B cells that will enter GC

    Outputs:
     incomingnaive: M_GC x length of naive B cells array of logical indices

    Inputs:
     Array of naive B cells, concentration, parameter struct
    """
    M_GC = param['M_GC']
    concarray = np.zeros(shape = naiveBcells[M_GC : 2 * M_GC].shape)
    for ep in range(param['n_ep']):
        term1 = conc[ep] * (naiveBcells[M_GC : 2 * M_GC] == ep + 1)
        term2 = (naiveBcells[5 * M_GC : 6 * M_GC] == 0) #incoming time
        concarray = concarray + term1 * term2
        
    term1 = (concarray / param['C0'])
    term2 = (10 ** (naiveBcells[M_GC * 2 : M_GC * 3, :] - param['f0']))
    activation_signal = (term1 * term2) ** param['w2']
    if param['w1'] > 0: # alternative Ag capture model
        term1 = (param['w1'] + 1) * activation_signal
        term2 = param['w1'] + activation_signal
        activation_signal = term1 / term2
    
    minarr = np.minimum(activation_signal, 1)
    activation = minarr > np.random.uniform(size = activation_signal.shape)
    n_activated = activation.sum(axis = 1)   
    
    if any_(n_activated): # at least one B cell is intrinsically activated
        activated_fitness = activation * activation_signal
        avgfitness = np.zeros(M_GC)
        for k in range(M_GC):
            avgfitness[k] = np.mean(activated_fitness[k, activated_fitness[k, :] > 0])
        term1 = param['Nmax'] / n_activated / avgfitness
        term2 = activated_fitness
        helpamount = term1[:, np.newaxis] * term2
        lambda_ = param['lambdamax'] * (helpamount / (1 + helpamount))
        
        boolarr2 = np.random.uniform(size=activation.shape) < lambda_ * param['dt']
        incomingnaive = activation & boolarr2
        
    else:
        incomingnaive = np.array([])
        
    return incomingnaive




def birthDeath(gcBcells, conc, currentTcell, currentGCnum, param):
    """
    Summary:
     Finds the indices of GC B cells that are positively selected and that
     will undergo apoptosis

    Outputs:
     gcBirthIdx, gcDeathIdx: M_GC x length of GC arrays. Logical indices

    Inputs:
     GC B cells, concentration, T cell number, current number of B cells in
     GC, parameter struct
    """
    M_GC = param['M_GC']
    concarray = np.zeros(shape = gcBcells[M_GC + 0: 2 * M_GC].shape)
    for ep in range(param['n_ep']):
        concarray = concarray + conc[ep] * (gcBcells[M_GC + 0: 2 * M_GC] == ep + 1)
        
    term1 = concarray / param['C0']
    term2 = 10 ** (np.minimum(gcBcells[M_GC * 2 + 0: M_GC * 3], 10) - param['f0'])
    activation_signal = (term1 * term2) ** param['w2']
    if param['w1'] > 0:
        term1 = (param['w1'] + 1) * activation_signal
        term2 = param['w1'] + activation_signal
        activation_signal = term1 / term2
    
    minarr = np.minimum(activation_signal, 1)
    activation = minarr > np.random.uniform(size = activation_signal.shape)
    n_activated = activation.sum(axis = 1)
    
    if any_(n_activated): # at least one B cell is intrinsically activated
        activated_fitness = activation * activation_signal
        avgfitness = np.zeros(M_GC)
        for k in range(M_GC):
            avgfitness[k] = np.mean(activated_fitness[k, activated_fitness[k, :] > 0])
        term1 = currentTcell / n_activated / avgfitness
        term2 = activated_fitness
        helpamount = term1[:, np.newaxis] * term2
        beta = param['betamax'] * (helpamount / (1 + helpamount))
        beta[np.isnan(beta)] = 0
        temp = reshape_(find(beta))
        randu = np.random.uniform(size = temp.shape)
        temp_idx = randu < beta[invind(temp, beta.shape[0])] * param['dt']
        temp = temp[temp_idx]
        gcBirthIdx = np.zeros(shape = beta.shape).astype(bool)
        gcBirthIdx[invind(temp, beta.shape[0])] = True
    else:
        gcBirthIdx = np.array([])
        
    gcDeathIdx = np.zeros(shape = activation.shape).astype(bool)
    for k in range(M_GC):
        cGCnum = int(currentGCnum[k])
        live = gcBcells[k, :cGCnum] != 0
        mu = param['mu']
        liveprob = np.random.uniform(size = cGCnum) < mu * param['dt']
        gcDeathIdx[k, :cGCnum] = liveprob & live
        
    return gcBirthIdx, gcDeathIdx



def getMemoryMutations(memory, mutations, param):
    """
    Summary:
     Returns the mutation sizes of the memory cells. This enables only the
     mutation sizes of relevant memory cells to be saved, for storage space.
     Then, this information can be accessed upon next vaccination.

    Outputs:
     memory: Array of GC-derived memory cells. 
     mutationsCompact:3 x (max number of memory cells in GC) x 80 array

    Inputs:
     memory: Array of GC-derived memory cells. 
     mutations: 1x3 cell array of mutations
     param: parameter struct
    """
    m1, m2, m3 = memory.shape
    uniqueCloneNum = 0
    warnFlag = False
    mutationsCompact = np.zeros(shape = (param['n_ep'], m2, param['n_res']))
    for k in range(m1):
        setdiff = set(np.unique(np.squeeze(memory[k, :, 0]))).difference(set([0]))
        uniqueClones = np.sort(list(setdiff)).astype(int)
        for clone in uniqueClones:
            uniqueCloneNum += 1
            memory[k, np.squeeze(memory[k, :, 0] == clone), 9] = uniqueCloneNum
            for i in range(param['n_ep']):
                if uniqueCloneNum >= mutationsCompact.shape[1]: # expand mutationsCompact
                    warnFlag = True
                    zeros = np.zeros(shape = mutationsCompact.shape)
                    mutationsCompact = np.concatenate([mutationsCompact, zeros], axis=1)
                mutationsCompact[i, uniqueCloneNum - 1, :] = mutations[i][k, clone - 1, :]

    if warnFlag:
        print('Warning: uniqueCloneNum larger than max num PC')

    mutationsCompact = mutationsCompact[:, :uniqueCloneNum, :]
    return memory, mutationsCompact


def memPCDeath(plasmaCells, plasmaCellsEGC, param):
    """
    Summary: Find indices of PCs that will undergo apoptosis
    
    Inputs: GC and EGC-derived PC arrays, parameter struct
    
    Outputs: 
     pcDeathIdx: M_GC x (GC-derived PC max capacity) logical indices
     egcPCDeathIdx: 1 x (EGC-derived PC max capacity) logical indices
    """
    M_GC = param['M_GC']
    rand1 = np.random.uniform(size = plasmaCells[:M_GC, :].shape)
    rand2 = np.random.uniform(size = plasmaCellsEGC[0].shape)
    threshold = param['d_pc'] * param['dt']
    pcDeathIdx = np.logical_and(plasmaCells[:M_GC, :] > 0, rand1 < threshold)
    egcPCDeathIdx = np.logical_and(plasmaCellsEGC[0] > 0,  rand2 < threshold)
    return pcDeathIdx, egcPCDeathIdx




def updateConcentrations(agconc,
                         abconc,
                         Ka,
                         Ka_var1,
                         Ka_var2,
                         plasmaBlasts,
                         plasmaCells,
                         plasmaCellsEGC,
                         t,
                         param):
    """
    Summary:
    Updates the antigen and antibody concentrations, antibody affinities

    Outputs: 
      agconc: 1x4 vector; Concentrations of Antigens; 
                          soluble, IC1, IC2, IC3
      abconc: 3x3 vector; Concentration of antibodies
              Dim1 - IgM(natural),IgM(immune),IgG; Dim2-Epitopes
      Ka: 3x3 vector; Antibody binding affinities (Ka) to vax strain; Dim 1 - IgM(natural), IgM(immune), IgG, Dim 2 - Epitopes
      Ka_var1: 3x3 vector; Antibody binding affinities (Ka) to Variant 1 strain i.e. strain 2; Dim 1 - IgM(natural), IgM(immune), IgG, Dim 2 - Epitopes
      Ka_var2: 3x3 vector; Antibody binding affinities (Ka) to Variant 2 strain i.e. strain 3; Dim 1 - IgM(natural), IgM(immune), IgG, Dim 2 - Epitopes
      param: parameter struct

    Inputs: 
      agconc, abconc, Ka, Ka_var1, Ka_var2 are same as the outputs
      plasmaBlasts, plasmaCells, plasmaCellsEGC are arrays of plasma cells
      t: current time
      param: parameter struct
    """
    # Check if antigen should be given at current time
    if any_(t == param['dose_t']):
        agconc[0] = agconc[0] + param['dose']
    
    # Get average affinity of the serum and soluble Ag and Ab concentrations
    R, L = agconc[0], np.sum(abconc[:])
    Ka_avg = ((Ka * abconc)).sum() / L
    
    #Get rates for deposition of IC onto FDC
    if (L > 0) and (R > 0):
        term1 = R + L + 1 / Ka_avg
        IC_soluble = (term1 - np.emath.sqrt(term1 ** 2 - 4 * R * L)) / 2 # Equil
        # Same dimensions as abconc / Ka, rates of IC-FDC depoition by the Ab species
        r = param['k_deposit'] * IC_soluble * Ka * abconc / (Ka * abconc).sum()
        if (~np.isreal(IC_soluble)) or (np.isnan(IC_soluble)): # check if im or nan
            raise ValueError('Imaginary or nan for soluble IC concentration')
    else: # If no antigen or antibody
        r = np.zeros(Ka.shape)
        
    abconc_old = deepcopy(abconc) # temporarily save current Ab concentrations
    
    # Check if any of the Ab species concentration will go to 0
    # If this is going to happen, then rescale the reaction rates
    decay_Ab = -r - np.array([0, param['d_IgM'], param['d_IgG']])[:, np.newaxis] * abconc
    rescale_idx = abconc < -decay_Ab * param['dt']
    rescale_factor = abconc / (-decay_Ab * param['dt'])
    if any_(rescale_idx.flatten()):
        print(f'Reaction rates rescaled. Time is {t:.2f}')
        r[rescale_idx] = r[rescale_idx] * rescale_factor[rescale_idx]
        decay_Ab[rescale_idx] = decay_Ab[rescale_idx] * rescale_factor[rescale_idx]
        if any_(np.isnan(decay_Ab.flatten())):
            raise ValueError('Rescaled Ab decay rates contain NaN')
            
    # Check if the soluble Ag concentration will go to 0. Rescale if needed.
    decay_Ag = -param['d_Ag'] * agconc[0] - np.sum(r) # Net consumption of soluble Ag
    if agconc[0] < -decay_Ag * param['dt']:
        rescale_factor = agconc[0] / (-decay_Ag * param['dt'])
        decay_Ag = decay_Ag * rescale_factor
        rnew = r * rescale_factor
        decay_Ab = decay_Ab + r - rnew
        r = rnew
        
    # Update the antigen and antibody concentrations for injection/consumption
    term1 = decay_Ag * param['dt']
    term2 = param['F0'] * np.exp(param['k'] * t) * (t < param['T']) * param['dt']
    agconc[0] = agconc[0] + term1 + term2
    term1 = np.sum(r, axis=0) * param['dt']
    term2 = agconc[1:] * param['d_IC'] * param['dt']
    agconc[1:] = agconc[1:] + term1 - term2
    abconc = abconc + decay_Ab * param['dt']
    abconc[np.abs(abconc) < 1e-10] = 0
    if any_(agconc.flatten() < 0) or any_(abconc.flatten() < 0):
        raise ValueError('Negative concentration')
        
    # Update the antibody concentration and affinity for production
    M_GC = param['M_GC']
    Ig_new = np.zeros(shape = (3, param['n_ep']))
    Ka_new = np.array([Ig_new, Ig_new, Ig_new]) # shape depends on num of variants
    Affinity = np.empty(shape = (param['n_var'], 3), dtype=object)
    Target = np.empty(shape = 3, dtype=object)
    
    for i in range(param['n_var']): # variants
        Affinity[i, 0] = plasmaBlasts[(i + 2) * M_GC: (i + 3) * M_GC]
        Affinity[i, 1] = plasmaCells[(i + 2) * M_GC: (i + 3) * M_GC]
        Affinity[i, 2] = plasmaCellsEGC[i + 2, :]
        threshold = t - param['delay']
        term2 = plasmaBlasts[5 * M_GC: 6 * M_GC, :] < threshold # time cell was produced
        Target[0] = plasmaBlasts[M_GC: 2 * M_GC, :] * term2
        term2 = plasmaCells[6 * M_GC: 7 * M_GC, :] < threshold # time cell was produced
        Target[1] = plasmaCells[M_GC: 2 * M_GC, :] * term2
        Target[2] = plasmaCellsEGC[1, :]
        
    for Ig_type in range(3): # IgM, IgG-GCPC, IgG-EGCPC
        for target in range(param['n_ep']):
            if any_(Target[Ig_type].flatten() - 1 == target):
                # Ig production
                term2 = param['r_IgM'] * (Ig_type == 0) + param['r_IgG'] * (Ig_type > 0)
                Ig_new[Ig_type, target] = (Target[Ig_type] - 1 == target).sum() * term2
                # Ka of new Ig
                for variant in range(param['n_var']):
                    targ_idx = Target[Ig_type] - 1 == target 
                    term2 = np.mean(10 ** (Affinity[variant, Ig_type][targ_idx] - 9)) #convert from nM to M
                    Ka_new[variant, Ig_type, target] = term2
    
    # Update amounts
    term2 = np.vstack([Ig_new[0, :], Ig_new[1:3, :].sum(axis=0)]) * param['dt']
    abconc[1: 3, :] = abconc[1: 3, :] + term2
    
    # Update Ka
    all_Ka = [Ka, Ka_var1, Ka_var2]
    for variant in range(param['n_var']):
        current_sum = (abconc_old + param['dt'] * decay_Ab) * all_Ka[variant]
        arr = np.array([[0, 0 ,0], [1, 0 ,0], [0, 1, 1]])
        new_sum = current_sum + arr @ (Ig_new * Ka_new[variant] * param['dt'])
        new_Ka = new_sum / abconc
        new_Ka[abconc == 0] = 0
        if any_(new_Ka.flatten() < 0) or any_(np.abs(new_Ka).flatten() > 1e11):
            print("Warning: Error Ka value, negative or too large")
        new_Ka[np.isnan(new_Ka)] = 0
        all_Ka[variant] = new_Ka
    Ka = all_Ka[0]
    Ka_var1 = all_Ka[1]
    Ka_var2 = all_Ka[2]
    epsilon = 1e-10
    abconc[abconc < epsilon] = 0
    agconc[agconc < epsilon] = 0
    
    return agconc, abconc, Ka, Ka_var1, Ka_var2




def updateHistory(result,
                  gcBcells,
                  plasmaCells,
                  memoryCells,
                  plasmaCellsEGC,
                  memoryCellsEGC,
                  agconc,
                  abconc,
                  Ka,
                  Ka_var1,
                  Ka_var2,
                  agconc_Epmask,
                  tspan_summary,
                  storeidx,
                  param):
    """
    Summary:
      Store the summary statistics of the current status of the simulation.
      This function is called repeatedly at a given time interval
    Output:
      result: Struct array that summarizes the outcome of the simulation. 
      ***** See the documentation for runGCs for details. *****

    Inputs:
      result, and various arrays and values from the simulation
    """
    n = len(tspan_summary)
    M_GC = param['M_GC']
    if len(result) == 0:
        result['param'] = param
        result['gc'] = {}
        result['gc']['numbytime'] = np.zeros(shape = (M_GC, param['n_ep'] * n, 4))
        result['gc']['affbytime'] = np.zeros(shape = (M_GC, param['n_ep'] * n, 4))
        result['gc']['numbylineage'] = np.zeros(shape = (M_GC, n, 2010))
        result['conc'] = {}
        result['conc']['concarray'] = np.zeros(shape = (4, param['n_ep'] + 1, n))
        result['conc']['concarray_Epmask'] = np.zeros(shape = (2, param['n_ep'], n))
        result['conc']['Kaarray'] = np.zeros(shape = (3, param['n_ep'], n))
        result['conc']['Kaarray_var1'] = np.zeros(shape = (3, param['n_ep'], n))
        result['conc']['Kaarray_var2'] = np.zeros(shape = (3, param['n_ep'], n))
        # dim1: GCtarget1, GCtarget2, EGCtarget1, EGCtarget2
        # dim3: affinities
        result['output'] = {}
        result['output']['pcnumbytime'] = np.zeros(shape = (2, param['n_ep'] * n, 4))
        result['output']['memnumbytime'] = np.zeros(shape = (2, param['n_ep'] * n, 4))
        result['output']['pcaffbytime'] = np.zeros(shape = (2, param['n_ep'] * n, 4));
        result['output']['memaffbytime'] = np.zeros(shape = (2, param['n_ep'] * n, 4))
        
    # Recording results
    # concentration
    conc1 = reshape_(agconc, row = True)
    conc2 = np.hstack([np.zeros(shape = (3, 1)), abconc]) # IgM(natural), IgM(immune), IgG
    result['conc']['concarray'][:, :, storeidx] = np.concatenate([conc1, conc2], axis=0)
    result['conc']['Kaarray'][:, :, storeidx] = Ka
    result['conc']['Kaarray_var1'][:, :, storeidx] = Ka_var1
    result['conc']['Kaarray_var2'][:, :, storeidx] = Ka_var2
    result['conc']['concarray_Epmask'][:, :, storeidx] = agconc_Epmask
    
    # Number and affinity of GC B cell
    if any_(any_(gcBcells)):
        numbyaff, affprct = cellsNumAff(gcBcells, M_GC, param)
        result['gc']['numbytime'][:, storeidx + n * np.arange(param['n_ep']), :] = numbyaff
        result['gc']['affbytime'][:, storeidx + n * np.arange(param['n_ep']), :] = affprct
        lineage = gcBcells[:M_GC, :]
        for k in range(M_GC):
            hist = np.histogram(lineage[k, :], bins = np.arange(2011) + 1)[0]
            result['gc']['numbylineage'][k, storeidx, :] = hist
    
    # Number and affinity of memory and plasma cells
    PCs = [flatten1D(buildup3D(plasmaCells, param['pcfieldnum'])), plasmaCellsEGC]
    MEMs = [flatten1D(buildup3D(memoryCells, param['memfieldnum'])), memoryCellsEGC]
    for i in range(2): #GC and EGC
        numbyaff, affprct = cellsNumAff(PCs[i], 1, param)
        result['output']['pcnumbytime'][i, storeidx + n * np.arange(param['n_ep'])] = numbyaff
        result['output']['pcaffbytime'][i, storeidx + n * np.arange(param['n_ep'])] = affprct
        
        numbyaff, affprct = cellsNumAff(MEMs[i], 1, param)
        result['output']['memnumbytime'][i, storeidx + n * np.arange(param['n_ep'])] = numbyaff
        result['output']['memaffbytime'][i, storeidx + n * np.arange(param['n_ep'])] = affprct
    
    return result

def cellsNumAff(cellsarr, M, param):
    """
    Obtain the summary of number and affinities of B cells
    Outpus:
      numbyaff: 1x3x4 array; Dim1,2,3 - GC, Epitope, 
                # of B cells with affinities greater than 6, 7, 8, 9
      affprct: 1x3x4 aray; Dim1,2,3 - GC, Epitope,
                100, 90, 75, 50 percentiles affinities
    Inputs:
      cellsarr: 2D array of B cells, each column representing a B cell.
                Can be GC, memory, or plasma cells
      M: Number of GC/EGC
      param: parameter struct
    """
    
    thresholds = np.array([6, 7, 8, 9])
    percentile = np.array([100, 90, 75, 50])
    aff = cellsarr[M * 2 + 0: M * 3]
    aff_reshape = reshape_(aff, row = True)
    target = cellsarr[M * 1 + 0: M * 2]
    target_reshape = reshape_(target, row = True)
    
    numbyaff = np.zeros(shape = (M, param['n_ep'], 4))
    affprct = np.zeros(shape = (M, param['n_ep'], 4))
    
    for i in range(4):
        for ep in range(param['n_ep']):
            temp = reshape_((aff * (target == ep + 1)) > thresholds[i], row = True)
            numbyaff[:, ep, i] = np.sum(temp, axis=1)
            for k in range(M):
                temp = prctile((aff_reshape[k, target_reshape[k, :] == ep + 1]).T,
                               percentile[i]).T
                affprct[k, ep, i] = temp
                
    return np.squeeze(numbyaff), np.squeeze(affprct)


def birthDeathEGC(memoryCellsEGC, conc, param):
    """
    Summary:
     Finds the indices of EGC memory B cells that will give birth to either
     new memory cells or to plasma cells

    Outputs:
     memBirthIdx, plasmaIdx: Indices (not logical). Length sums up to the
     number of EGC memory cells. 

    Inputs: EGC-derived memory cells, concentration, parameter struct
    """
    concarray = np.zeros(shape = memoryCellsEGC[1, :].shape)
    for ep in range(param['n_ep']):
        concarray = concarray + conc[ep] * (memoryCellsEGC[1, :] == ep + 1)
        
    term1 = concarray / param['C0']
    term2 = 10 ** (np.minimum(memoryCellsEGC[2, :], 10) - param['f0'])
    activation_signal = (term1 * term2) ** param['w2']
    
    if param['w1'] > 0:
        term1 = (param['w1'] + 1) * activation_signal
        term2 = param['w1'] + activation_signal
        activation_signal = term1 / term2
        
    minarr = np.minimum(activation_signal, 1)
    activation = minarr > np.random.uniform(size = activation_signal.shape)
    n_activated = activation.sum()
    
    if any_(n_activated): # at least one B cell is intrinsically activated
        activated_fitness = activation * activation_signal
        avgfitness = np.mean(activated_fitness[activated_fitness > 0])
        term1 = param['M_GC'] * param['numTmax'] / n_activated / avgfitness
        term2 = activated_fitness
        helpamount = term1 * term2
        beta = param['betamaxEGC'] * (helpamount / (1 + helpamount))
        beta[np.isnan(beta)] = 0
        
        temp = find(beta)
        randu = np.random.uniform(size = temp.shape[0])
        temp_idx = randu < beta[temp] * param['dt']
        temp = temp[temp_idx]
        
        r = np.random.uniform(size = temp.shape) < 0.6
        tempcpy1, tempcpy2 = deepcopy(temp), deepcopy(temp)
        plasmaIdx = temp[r]
        memBirthIdx = temp[~r]
        
    else:
        memBirthIdx = np.array([])
        plasmaIdx = np.array([])
    
    return memBirthIdx, plasmaIdx


def splitMemory(memoryCellsEGC, memoryMutations, mutations, param):
    """
    Summary:
      Takes in the pre-existing memory cells and their fitness landscapes,
      then selects a pre-define fraction of the memory B cells to be added to
      naive B cells for potential re-activation

    Outputs:
      memoryToGC: (M_GC*10) x MemoryReentryMax array.
                  Memory cells that will be added to naive B cells
      mutations: 1 x 3 cell array of mutations
      memoryToEGC: 10 x N, N is the number of memory cells that goes to EGC

    Inputs:
      memoryCellsEGC: EGC-derived memory cells 
      memoryMutations: 3 x N x 80 array. Mutation sizes of all unique memory
                       cells. Dim1 - target; Dim2 - Number of unique memory
                       Dim3 - Residues
      mutations: 1x3 cell of mutation sizes
      param: parameter struct
    """
    random.seed(1)
    np.random.seed(1)
    
    # Select the memory cells to be added to re-activation pool
    N_mem = findlast(memoryCellsEGC[0, :]) + 1
    toGC = np.random.binomial(1, param['memToGCFrac'], size = N_mem)
    memoryToGC = memoryCellsEGC[:, toGC == 1]
    memoryToGC = alignLeft(memoryToGC, 1)
    memoryCellsEGC[:, toGC == 1] = 0
    memoryToEGC = alignLeft(memoryCellsEGC, 0)
    
    # Distribute the memory cells over the secondary GCs
    N_memToGC = memoryToGC.shape[1]
    memPoolGC = np.zeros(shape = (param['M_GC'],
                                  param['MemoryReentryMax'],
                                  param['naivefieldnum']))
    memNumGC = np.zeros(param['M_GC']).astype(int)
    for i in range(N_memToGC):               # iterate over selected memory cells
        k = np.random.randint(param['M_GC']) # randomly select GC
        memNumGC[k] = memNumGC[k] + 1
        memPoolGC[k, memNumGC[k], 0] = memNumGC[k] + param['NaiveMax']        # lineage
        memPoolGC[k, memNumGC[k], 1:5] = memoryToGC[1:5, i]           # target, affinities
        memPoolGC[k, memNumGC[k], 6:8] = memoryToGC[7:9, i]   # mutation states
        for j in range(param['n_var']): # copy over mutation states
            val = memoryMutations[j, int(memoryToGC[9, i]) - 1, :]
            mutations[j][k, memNumGC[k] + param['NaiveMax'], :] = val

    memoryToGC = flatten2D(memPoolGC)
    
    return memoryToGC, mutations, memoryToEGC


# In[2]:


def runGCs(vaxnum,
           E1h,
           dE12,
           dE13,
           p2,
           p3,
           masking,
           C0,
           w1,
           w2,
           q12,
           q13,
           q23,
           memToGCFrac,
           outputprob,
           outputpcfrac,
           rho,
           earlybooster,
           tmax,
           first,
           last):
    """
    Summary:
      Store the summary statistics of the current status of the simulation.
      This function is called repeatedly at a given time interval

    Output:
      result: Struct array that have the following fields
        param: struct containing the parameters 
               **See documentation for initializeParameters.m**
        naive: 200 x 2210 x 8 array
            Array of naive B cells
            Dim1: GC; Dim2: Lineage
            Dim3: 1.Lineage 2.Target 3.Vax-aff, 4.Variant1-aff, 5. Variant2-aff
                  6. Time of activation, 7-8: First and last 40 residues in
                  decimal number (convert to binary for mutation state)
        gc: struct with info about GCs
          gc.numbytime: 200 x N x 4 array; N=(tmax*4+1)*(3). 
            # of GC B cells with affinities greater than 6, 7, 8, 9
            Dim1: GC
            Dim2: First N/3 columns are epitope 1, next N/3 columns are epitope 2, last N/3 columns are epitope 3
            Dim3: Time from 0 to tmax every 0.25 day
          gc.affbytime: similar to gc.numbytime, but 100, 90, 75, 50 
            percentiles affinities of GC B cells
          gc.numbylineage: 200 x N x 2010 array; N=(tmax/7+1). 
            # of GC B cells in each lineage
            Dim1: GC; Dim2: Time from 0 to tmax every 7 days 
            Dim3: Lineage number
          gc.finalgc: 200 x 3000 x 6 array;
            Array of Gc B cells at the last time point
            Dim1: GC; Dim2: B cells
            Dim3: 1.Lineage 2.Target 3.Vax-aff, 4.Var1-aff, 5. Var2-aff, 6. Num of mutations
        conc: struct with info about concentrations
          conc.concarray: 4 x 4 x (tmax*4+1) array
            Concentrations of antigen and antibodies 
            Dim1: 1.Ag 2.IgM(natural) 3.IgM(immune) 4.IgG
            Dim2: For Ag(row1): Soluble, IC1, IC2, IC3
                  For Abs(rows2-4): Empty, Target1, Target2, Target3
            **See documentation for updateConcentrations for more details**
          conc.concarray_Epmask: 2 x 3 x (tmax*4+1) array
            Concentrations of antigen considering epitope masking (or lack
            thereof)
            Dim1: Soluble, IC-FDC
            Dim2: Epitopes
          conc.Kaarray: 3 x 3 x (tmax*4+1) array
            Affinities (Ka; nM^-1) of antibodies for vax strain (strain 1)
            Dim1: 1.IgM(natural), 2.IgM(immune), 3.IgG
            Dim2: Epitopes
          conc.Kaarray_var: Similar to conc.Kaarray but Variant1 (strain 2) affinities
        output: struct with info about GC output cells
          output.pcnumbytime: similar to gc.numbytime, but for PCs
          output.memnumbytime: similar to gc.numbytime, but for Mem B cells
          output.pcaffbytime: similar to gc.affbytime, but for PCs
          output.memaffbytime: similar to gc.affbytime, but for Mem B cells
        memoryCellsEGC: 10 x N array; N may vary; Array of EGC-derived Mems
                        Dim 1: Lineage, target,vax affinity, variant1 affinity, variant2 affinity, nummut, activatedtime, 
                        mutationstaet1, mutationstate2, uniqueCloneIndex
        plasmaCellsEGC: 7 x N array; N may vary; Array of EGC-derived PCs
                        Dim 1: Lineage, target, vax affinity, variant1 affinity, variant2 affinity, nummut, activatedtime
        dead: struct with info about dead PCs
          dead.plasmaCells: 7x(2*10^6) array; Array of dead GC-derived PCs
          dead.PCnum: Scalar; number of dead GC-derived PCs
          dead.plasmaCellsEGC: 7x(2*10^6) array;Array of dead EGC-derived PCs
          dead.numPC: Scalar; number of dead EGC-derived PCs

    Inputs:
     varargin: Various parameters that define the simulation condition. 
          For details, see documentation for function "initializeParameters"
    """
    saveresult = 1 # Change to 0 if don't want to save result
    param = initializeParameters(vaxnum,
                                   E1h,
                                   dE12,
                                   dE13,
                                   p2,
                                   p3,
                                   masking,
                                   C0,
                                   w1,
                                   w2,
                                   q12,
                                   q13,
                                   q23,
                                   memToGCFrac,
                                   outputprob,
                                   outputpcfrac,
                                   rho,
                                   earlybooster,
                                   tmax,
                                   first,
                                   last)
    random.seed(param['first'])
    np.random.seed(param['first'])
    fnm = getFileLocation(param)
    tstart = time.perf_counter()
    
    ########################################################
    # Initialization
    ########################################################
    M_GC = param['M_GC']
    N_GC_MAX = param['N_GC_MAX']
    N_PC_MAX = param['N_PC_MAX']
    N_PB_MAX = param['N_PB_MAX']
    N_MEM_MAX = param['N_PC_MAX']
    n_res = param['n_res']
    
    ########################################################
    #Initialize naive B cells
    ########################################################
    naiveBcells, mutations = getNaiveBcells(param)
    # Lineage, target, vax affinity, variant1 affinity, variant2 affinity, nummut
    gcBcells = np.zeros(shape = (M_GC, N_GC_MAX, param['gcfieldnum']))
    # Each GC B cell mutation state is a string of 0s and 1s of length n_res
    gcMutations = np.zeros(shape = (M_GC, N_GC_MAX * n_res))
    # Lineage, target, vax affinity, variant1 affinity, variant2 affinity, nummut, activatedtime
    plasmaCells = np.zeros(shape = (M_GC, N_PC_MAX, param['pcfieldnum']))
    # Lineage, target, vax affinity, variant1 affinity, variant2 affinity, activatedtime 
    plasmaBlasts = np.zeros(shape = (M_GC, N_PB_MAX, param['pcfieldnum'] - 1))
    # Lineage, target,vax affinity, variant1 affinity, variant2 affinity, nummut, activatedtime, 
    # mutationstaet1, mutationstate2, uniqueCloneIndex
    memoryCells = np.zeros(shape = (M_GC, N_MEM_MAX, param['memfieldnum']))
    dead = {'plasmaCells': np.zeros(shape = (param['pcfieldnum'], N_PC_MAX * M_GC)),
            'PCnum': 0}
    
    # This will track the last non-empty entries of each row in gcBcells array
    # Not equal to the number of GC B cells
    numGC = np.zeros(M_GC)
    numPB = np.zeros(M_GC)
    numPC = np.zeros(M_GC)
    numMC = np.zeros(M_GC)
    
    result = {}
    
    # Flatten the array to 2D for the convenience of operations
    naiveBcells = flatten2D(naiveBcells)
    gcBcells = flatten2D(gcBcells)
    plasmaCells = flatten2D(plasmaCells)
    plasmaBlasts = flatten2D(plasmaBlasts)
    memoryCells = flatten2D(memoryCells)
    
    agconc, abconc, Ka, param = getDosingParameters(param)
    Ka_var1 = Ka #Just the same shape as Ka, doesn't actually have significant values yet
    Ka_var2 = Ka
    
    ########################################################
    # Load pre-existing memory cells and concentrations
    ########################################################
    if param['vaxnum'] > 1:
        previous = expand(pickle.load(open(fnm[param['vaxnum'] - 2], 'rb')))
        memoryCellsEGC = np.zeros(shape = (param['memfieldnum'], N_PC_MAX * M_GC))
        numMCEGC = 0
        plasmaCellsEGC = np.zeros(shape = (param['pcfieldnum'], N_PC_MAX * M_GC))
        numPCEGC = 0
        
        # Memory cells
        GC_derived_mem = alignLeft(flatten1D(previous['output']['finalmem']),  1)
        EGC_derived_mem = alignLeft(reshape_(previous['memoryCellsEGC']), 1)
        existingMemory = np.concatenate([GC_derived_mem, EGC_derived_mem], axis=1)
        # Get memory B cells that will go into re-entry pool
        memoryToGC, mutations, memoryToEGC = splitMemory(existingMemory,
                                                         previous['output']['memMutations'],
                                                         mutations, param)
        naiveBcells = np.concatenate([naiveBcells, memoryToGC], axis=1)
        memoryCellsEGC[:, : memoryToEGC.shape[1]] = memoryToEGC
        numMCEGC = numMCEGC + memoryToEGC.shape[1]
        
        # Plasma cells
        existingPC = alignLeft(flatten1D(previous['output']['finalpc']), 1)
        if len(existingPC) != 0:
            numPCEGC = existingPC.shape[1]
            plasmaCellsEGC[:, :numPCEGC] = existingPC
            
        dead['plasmaCellsEGC'] = np.zeros(shape = (param['pcfieldnum'], N_PC_MAX * M_GC))
        dead['numPC'] = 0
        
        # Load antigen concentration, antibody concentration and affinity 
        agconc = np.squeeze(previous['conc']['concarray'][0, :param['n_ep'] + 1, -1]) #XXX
        abconc = np.squeeze(previous['conc']['concarray'][1:4, 1:param['n_ep'] + 1, -1])
        Ka = np.squeeze(previous['conc']['Kaarray'][:, :, -1])
        Ka_var1 = np.squeeze(previous['conc']['Kaarray_var1'][:, :, -1])
        Ka_var2 = np.squeeze(previous['conc']['Kaarray_var2'][:, :, -1])
        
    else:
        # Memory cells
        memoryCellsEGC = np.zeros(shape = (param['memfieldnum'], 1))
        numMCEGC = 0
        
        # Plasma cells
        plasmaCellsEGC = np.zeros(shape = (param['pcfieldnum'], 1))
        numPCEGC = 0
        dead['plasmaCellsEGC'] = np.zeros(shape = (param['pcfieldnum'], 1))
        dead['numPC'] = 0
    
    ########################################################
    # Initialize for competitive phase
    ########################################################
    agconc_Epmask = epitopeMasking(agconc, abconc, Ka, param) # get effective conc
    numTcell = getnumTcells(param)                            # get number of T cells
    tspan = np.arange(0, param['tmax'] + param['dt'], param['dt'])
    tspan_summary = param["tspan_summary"]
    T = 0                                                     # Total simulation time
    
    ########################################################
    # Competitive phase begins
    ########################################################

    for idx, t in enumerate(tspan):
        
        if idx % 10 == 0:
            gc.collect()
        
        currentTcell = numTcell[idx]
        conc = np.array([param['Ageff'], 1]) @ agconc_Epmask
        
        # Flux of naive B cells
        ###############################
        # get incoming naive B cells
        incomingLogicIdx = naiveFlux(naiveBcells, conc, param)
        
        if any_(any_(incomingLogicIdx)): # if any naive B cell incoming
            n_naive = naiveBcells.shape[1]
            entry1 = np.zeros(shape = (5 * M_GC, n_naive)).astype(bool) #properties in array before incoming time entry
            entry3 = np.zeros(shape = (2 * M_GC, n_naive)).astype(bool) #properties in array after incoming time entry
            entry_time_idx = np.concatenate([entry1, incomingLogicIdx, entry3], axis = 0)
            naiveBcells[entry_time_idx] = (t + 0.5 * param['dt']) # time of entry           
            
            # Add naive B cells to GCs
            numIncoming = np.sum(incomingLogicIdx, axis=1)
            for k in np.where(numIncoming > 0)[0]:
                ind1 = k + M_GC * np.array([0, 1, 2, 3, 4, 6, 7]) #Everything except time of entry
                ind2 = np.nonzero(incomingLogicIdx[k, :])[0]
                incoming = reshape_(matindget(naiveBcells, ind1, ind2))

                for j in range(numIncoming[k]):
                    copies = param['naiveprolnum'] # makes 4 copies
                    # new GC B cells
                    newIdx = np.arange(numGC[k] + 0, numGC[k] + copies).astype(int)
                    val = np.tile(reshape_(incoming[0: 5, j]), (1, copies)).flatten()
                    arange1 = np.arange(k, 4 * M_GC+ k + 1, M_GC).astype(int)
                    ind1, ind2 = matind(arange1, newIdx)
                    gcBcells[ind1, ind2] = val
                    numGC[k] = numGC[k] + copies
                    # if the new GC B cells are from pre-existing memory cells
                    if (incoming[0, j] > 2010) and any_(incoming[5: 7, j] > 0): # second part of and statement refers to mutations
                        reshaped = np.reshape(int2bit(incoming[5: 7, j], n_res / 2),
                                              (n_res, 1),
                                              order = 'F').T
                        val = np.tile(reshaped, (1, copies))
                        gcMutations[k, res2ind(newIdx, np.arange(n_res), param)] = val
                                
        # GC B cells birth and death
        ###############################
        if any_(any_(numGC)): # if any GC B cell
            # get index of birth and death
            gcBirthIdx, gcDeathIdx = birthDeath(gcBcells, conc, currentTcell, numGC, param)
            
            # Birth
            ###################
            if any_(any_(gcBirthIdx)):
                numBirths = np.sum(gcBirthIdx, axis=1)
                # Lineage, Target, Affinity Vax, Affinity Var1, Affinity Var2, Mutnum
                dtCells = np.zeros(shape = (M_GC * param['gcfieldnum'], max(numBirths)))
                # Keep track of mutated residue, if a mutation occurs
                dtMuts = np.zeros(shape = (M_GC, max(numBirths)))
                
                for k in range(M_GC):
                    birthidx = np.nonzero(gcBirthIdx[k, :])[0]
                    # keeps track of the actual daughter cell indices in order
                    # to copy over mutation state later
                    mutidx = np.zeros(birthidx.size).astype(int)
                    jj = 0
                    if len(birthidx):
                        for j in range(len(birthidx)):
                            # mutation can only happen to the daughter cell
                            r1, r2 = np.random.uniform(size = 2)
                            # positively selected cell differentiates into Mem or PC
                            if r1 < param['outputprob']:
                                arange1 = np.arange(k, 5 * M_GC + k + 1, M_GC).astype(int)
                                term1 = gcBcells[arange1, birthidx[j]]
                                term2 = [t + 0.5 * param['dt']]
                                val = np.concatenate([term1, term2])
                                arange1 = np.arange(k, k + 6 * M_GC + 1, M_GC).astype(int)
                                # becomes a plasma cell
                                if np.random.uniform() < param['outputpcfrac']:
                                    plasmaCells[arange1, int(numPC[k]) + 0] = val
                                    numPC[k] = numPC[k] + 1
                                # becomes a memory cell
                                else:
                                    memoryCells[arange1, int(numMC[k]) + 0] = val
                                    
                                    rind = res2ind(birthidx[j], np.arange(n_res / 2), param)
                                    gcMi = gcMutations[k, rind]
                                    reshaped = np.reshape(gcMi, (int(n_res / 2), -1))
                                    val = bit2int(reshaped, int(n_res / 2))
                                    memoryCells[k + 7 * M_GC, int(numMC[k]) + 0] = val
                                    
                                    rind = res2ind(birthidx[j], np.arange(n_res / 2 + 0, n_res), param)
                                    gcMi = gcMutations[k, rind]
                                    reshaped = np.reshape(gcMi, (int(n_res / 2), -1))
                                    val = bit2int(reshaped, int(n_res / 2))
                                    memoryCells[k + 8 * M_GC, int(numMC[k]) + 0] = val 
                                    
                                    numMC[k] = numMC[k] + 1
                                    
                                numBirths[k] = numBirths[k] - 1 # no daughter cell
                                gcDeathIdx[k, birthidx[j]] = 1  # parent cell removed
                                
                            # Mutation leads to death    
                            elif r2 < 0.3:
                                numBirths[k] = numBirths[k] - 1 # no daughter cell
                                
                            # No mutation until t = 6, Jacob, Miller, Kelsoe 1992 Immunol. Cell. Bio
                            elif (r2 < 0.8) or (t < 6):
                                arange1 = np.arange(k, 5 * M_GC + k + 1, M_GC)
                                arange2 = np.arange(k, 5 * M_GC + k + 1, M_GC)
                                dtCells[arange1, jj] = gcBcells[arange2, birthidx[j]]
                                
                                if t > 6:
                                    # silent mutation
                                    dtCells[5 * M_GC + k, jj] = dtCells[5 * M_GC + k, jj] + 1 
                                mutidx[jj] = birthidx[j]
                                jj += 1
                            
                            # Mutation changes affinity
                            else:
                                arange1 = np.arange(k, 5 * M_GC + k + 1, M_GC) 
                                dtCells[arange1, jj] = gcBcells[arange1, birthidx[j]]
                                lineage = int(gcBcells[k, birthidx[j]]) # get lineage
                                lin_ind = lineage - 1                   # index is lineage - 1
                                mutres = np.random.randint(n_res) + 1   # needs to be greater than 0
                                mutind = mutres - 1                     # but index is subtracted by 1
                                dE = mutations[0][k, lin_ind, mutind]    # Vax affinity
                                dEvar1 = mutations[1][k, lin_ind, mutind] # Variant1 affinity
                                dEvar2 = mutations[2][k, lin_ind, mutind] # Variant2 affinity
                                already_mutated = gcMutations[k, res2ind(birthidx[j], mutind, param)]
                                if already_mutated: # Mutate from 1 to 0
                                    dtMuts[k, jj] = -mutres
                                    dE = -dE
                                    dEvar1 = -dEvar1
                                    dEvar2 = -dEvar2
                                    # Affinity changing
                                    dtCells[5 * M_GC + k, jj] = dtCells[5 * M_GC + k, jj] - 100 
                                else:
                                    dtMuts[k, jj] = mutres
                                    dtCells[5 * M_GC + k, jj] = dtCells[5 * M_GC + k, jj] + 100 
                                    
                                # Update the affinity and sanity check
                                dtCells[k + M_GC * 2, jj] = dtCells[k + M_GC * 2, jj] - dE
                                dtCells[k + M_GC * 3, jj] = dtCells[k + M_GC * 3, jj] - dEvar1
                                dtCells[k + M_GC * 4, jj] = dtCells[k + M_GC * 4, jj] - dEvar2
                                # no mutation can be super beneficial
                                if dE < -8:
                                    raise ValueError('Beneficial mutation too large')
                                if any_(dtCells[k + M_GC * np.array([2, 3, 4]), jj] > 16):
                                    raise ValueError('Affinity impossibly high')
                                mutidx[jj] = birthidx[j]
                                jj += 1
                                
                        
                        # Keep track of the mutational state of each GC B cell
                        if numBirths[k] > 0:
                            newIdx = np.arange(numGC[k] + 0, numGC[k] + numBirths[k])
                            arange1 = np.arange(k, 5 * M_GC + k + 1, M_GC).astype(int)  
                            ind1, ind2 = matind(arange1, newIdx)
                            ind3, ind4 = matind(arange1, np.arange(numBirths[k]).astype(int))
                            gcBcells[ind1, ind2] = dtCells[ind3, ind4]
                            
                            # 1 if 0 -> 1, -1 if 1 -> 0, 0 otherwise
                            dirpos = (dtMuts[k, np.arange(numBirths[k])] > 0).astype(int)
                            dirneg = (dtMuts[k, np.arange(numBirths[k])] < 0).astype(int)
                            mutDirection = dirpos - dirneg
                            dtMuts[k, :] = np.abs(dtMuts[k, :]) # index of mutated residue
                            
                            
                            # Copy over the mutations state, then update it
                            for j in range(numBirths[k]):
                                val = gcMutations[k, res2ind(mutidx[j], np.arange(n_res), param)]
                                gcMutations[k, res2ind(numGC[k] + j, np.arange(n_res), param)] = val
                                
                            mutated = np.nonzero(mutDirection)[0]
                            resIdx = res2ind(newIdx[mutated], dtMuts[k, mutated], param) - 1
                            gcMutations[k, resIdx] = gcMutations[k, resIdx] + mutDirection[mutated]                                 
                            numGC[k] = numGC[k] + numBirths[k]
            
                            
                            
            # Deaths
            ###################
            for k in range(M_GC): # Death or exit as memory
                deathidx = np.nonzero(gcDeathIdx[k, :])[0]
                arange1 = np.arange(k, k + 5 * M_GC + 1, M_GC).astype(int) 
                ind1, ind2 = matind(arange1, deathidx)
                gcBcells[ind1, ind2] = 0
                for idx in deathidx:
                    gcMutations[k, res2ind(idx, np.arange(n_res).astype(int), param)] = 0
                
                # Cleanup of the arrays to remove the 0s in between and align to left
                if numGC[k] > N_GC_MAX * 0.95:
                    liveBcells = np.nonzero(gcBcells[k])[0]
                    arange1 = np.arange(k, k + 5 * M_GC + 1, M_GC) 
                    ind1, ind2 = matind(arange1, np.arange(len(liveBcells)).astype(int))
                    ind3, ind4 = matind(arange1, liveBcells)
                    arange2 = np.arange(len(liveBcells), gcBcells.shape[1]).astype(int)
                    ind5, ind6 = matind(arange1, arange2)
                    gcBcells[ind1, ind2] = gcBcells[ind3, ind4]
                    gcBcells[ind5, ind6] = 0
                    numGC[k] = len(liveBcells)
                    
                    val = gcMutations[k, res2ind(liveBcells, np.arange(n_res), param)]
                    gcMutations[k, res2ind(np.arange(numGC[k]), np.arange(n_res), param)] = val
                    gcMutations[k, res2ind(numGC[k] + 0, 0, param):] = 0 # 1 to 0
                    
                    if numGC[k] > N_GC_MAX * 0.95: # if array is almost filled, then expand
                        shape = (M_GC, int(param['N_GC_MAX'] / 2), param['gcfieldnum'])
                        tmplst = [gcBcells, flatten2D(np.zeros(shape = shape))]
                        gcBcells = np.concatenate(tmplst, axis=1)
                        zeros = np.zeros(shape = (M_GC, int(param['N_GC_MAX'] / 2 * n_res)))
                        gcMutations = np.concatenate([gcMutations, zeros], axis=1)
                        N_GC_MAX += param['N_GC_MAX'] / 2
                        
                        
        # Output cells birth and death
        ###############################
        # EGC birth
        if (param['vaxnum'] > 1) and (t < 6):
            memBirthIdx, plasmaIdx = birthDeathEGC(memoryCellsEGC[:, np.arange(numMCEGC)], conc, param)
            if any_(plasmaIdx):
                newIdx = np.arange(numPCEGC + 0, numPCEGC + len(plasmaIdx))
                ind1, ind2 = matind(np.arange(plasmaCellsEGC.shape[0]), newIdx)
                ind3, ind4 = matind(np.arange(7), plasmaIdx) #num of properties for PC
                plasmaCellsEGC[ind1, ind2] = memoryCellsEGC[ind3, ind4]
                numPCEGC = numPCEGC + len(plasmaIdx)
            if any_(memBirthIdx):
                newIdx = np.arange(numMCEGC + 0, numMCEGC + len(memBirthIdx))
                ind1, ind2 = matind(np.arange(memoryCellsEGC.shape[0]), newIdx)
                ind3, ind4 = matind(np.arange(memoryCellsEGC.shape[0]), memBirthIdx)
                memoryCellsEGC[ind1, ind2] = memoryCellsEGC[ind3, ind4]
                numMCEGC = numMCEGC + len(memBirthIdx)
                
        # EGC death
        pcDeathIdx, egcPCDeathIdx = memPCDeath(plasmaCells, plasmaCellsEGC, param)
        arange1 = np.arange(dead['numPC'], dead['numPC'] + np.sum(egcPCDeathIdx))
        dead['plasmaCellsEGC'][:, arange1] = plasmaCellsEGC[:, egcPCDeathIdx]
        dead['numPC'] = dead['numPC'] + np.sum(egcPCDeathIdx)
        temp = plasmaCells.T
        repmat = np.tile(pcDeathIdx, (param['pcfieldnum'], 1)).T
        deadPCs = np.reshape(temp[repmat], (-1, param['pcfieldnum']), order = 'F').T
        dead['plasmaCells'][:, np.arange(dead['PCnum'], dead['PCnum'] + deadPCs.shape[1])] = deadPCs
        dead['PCnum'] = dead['PCnum'] + deadPCs.shape[1]
        #used to be length in matlab, but length uses last dimension while len uses first
        if dead['numPC'] > (0.95 * dead['plasmaCellsEGC'].shape[-1]):
            zeros = np.zeros(shape = dead['plasmaCellsEGC'].shape)
            dead['plasmaCellsEGC'] = np.concatenate([dead['plasmaCellsEGC'], zeros], axis = 1)
        if dead['PCnum'] > (0.95 * dead['plasmaCells'].shape[-1]):
            zeros = np.zeros(shape = dead['plasmaCells'].shape)
            dead['plasmaCells'] = np.concatenate([dead['plasmaCells'], zeros], axis = 1)
        plasmaCellsEGC[:, egcPCDeathIdx] = 0
        plasmaCells[np.tile(pcDeathIdx, (param['pcfieldnum'], 1))] = 0
        
        # Array scaling
        if any_(numMC > (0.95 * memoryCells.shape[1])):
            tmplst = [memoryCells, np.zeros(shape = memoryCells.shape)]
            memoryCells = np.concatenate(tmplst, axis = 1)
        if numMCEGC > (0.95 * memoryCellsEGC.shape[1]):
            tmplst = [memoryCellsEGC, np.zeros(shape = memoryCellsEGC.shape)]
            memoryCellsEGC = np.concatenate(tmplst, axis=1)
        if numPCEGC > (0.95 * plasmaCellsEGC.shape[1]):
            plasmaCellsEGC = alignLeft(plasmaCellsEGC, 0)
            numPCEGC = findlast(plasmaCellsEGC[0, :])
            if numPCEGC > (0.95 * plasmaCellsEGC.shape[1]):
                zeros = np.zeros(shape = plasmaCellsEGC.shape)
                plasmaCellsEGC = np.concatenate([plasmaCellsEGC, zeros], axis = 1)
        
        T += time.perf_counter() - tstart
        
        # Update the concentrations
        ###############################
        arange1 = np.arange(numPB.max()).astype(int)
        arange2 = np.arange(numPC.max()).astype(int)
        agconc, abconc, Ka, Ka_var1, Ka_var2 = updateConcentrations(agconc,
                                                          abconc,
                                                          Ka,
                                                          Ka_var1,
                                                          Ka_var2,
                                                          plasmaBlasts[:, arange1],
                                                          plasmaCells[:, arange2],
                                                          plasmaCellsEGC,
                                                          t,
                                                          param)
        agconc_Epmask = epitopeMasking(agconc, abconc, Ka, param)
        
        # Store results intermittently
        ###############################
        if t in tspan_summary:
            storeidx = np.where(tspan_summary == t)[0][0]
            result = updateHistory(result,
                                   gcBcells,
                                   plasmaCells,
                                   memoryCells,
                                   plasmaCellsEGC,
                                   memoryCellsEGC,
                                   agconc,
                                   abconc,
                                   Ka,
                                   Ka_var1,
                                   Ka_var2,
                                   agconc_Epmask,
                                   tspan_summary,
                                   storeidx, param)
        if t in np.arange(1, 180 + 1):
            print(f't = {t}')
            print(f'elapsed time: {(time.perf_counter() - tstart):.1f} s')
            
    
    # End competitive phase
    
    ########################################################
    # Cleanup and store result
    ########################################################
    for k in range(M_GC):
        liveBcells = np.nonzero(gcBcells[k, :])[0]
        if liveBcells.size:
            arange1 = np.arange(k, k + 5 * M_GC + 1, M_GC) 
            ind1, ind2 = matind(arange1, np.arange(len(liveBcells)))
            ind3, ind4 = matind(arange1, liveBcells)
            ind5, ind6 = matind(arange1, np.arange(len(liveBcells), gcBcells.shape[1]))
            gcBcells[ind1, ind2] = gcBcells[ind3, ind4]
            gcBcells[ind5, ind6] = 0
            numGC[k] = len(liveBcells)
            ind1 = res2ind(np.arange(numGC[k]), np.arange(n_res), param)
            ind2 = res2ind(liveBcells, np.arange(n_res), param)
            ind3 = res2ind(numGC[k] + 1, 0, param)
            gcMutations[k, ind1] = gcMutations[k, ind2]
            gcMutations[k, ind3:] = 0
        arange1 = np.arange(k, k + (param['pcfieldnum'] - 2) * M_GC + 1, M_GC)
        arange2 = np.arange(k, k + (param['pcfieldnum'] - 1) * M_GC + 1, M_GC)
        arange3 = np.arange(k, k + (param['memfieldnum'] - 2) * M_GC + 1, M_GC)
        plasmaBlasts[arange1] = alignLeft(plasmaBlasts[arange1], 0)
        plasmaCells[arange2] = alignLeft(plasmaCells[arange2], 0)
        memoryCells[arange3] = alignLeft(memoryCells[arange3], 0)
        
    plasmaCellsEGC = np.squeeze(alignLeft(plasmaCellsEGC, 0))
    memoryCellsEGC = np.squeeze(alignLeft(memoryCellsEGC, 0))
    
    # Reshape back to 3D arrays
    result['output']['finalpb'] = buildup3D(plasmaBlasts, param['pcfieldnum'] - 1)
    result['output']['finalpc'] = buildup3D(plasmaCells, param['pcfieldnum'])
    result['output']['finalmem'] = buildup3D(memoryCells, param['memfieldnum'])
    result['naive'] = buildup3D(naiveBcells, param['naivefieldnum'])
    result['gc']['finalgc'] = buildup3D(gcBcells, param['gcfieldnum'])
    result['plasmaCellsEGC'] = plasmaCellsEGC
    result['memoryCellsEGC'] = memoryCellsEGC
    result['dead'] = dead
    
    idx = 4 * np.arange(0, param['tmax'] + 1, 7)
    result['gc']['numbylineage'] = result['gc']['numbylineage'][:, idx, :]
    val = getMemoryMutations(result['output']['finalmem'], mutations, param)
    result['output']['finalmem'], result['output']['memMutations'] = val

    # Compress and save to file
    if saveresult:
        print(fnm[param['vaxnum'] - 1])
        pickle.dump(compress(result), open(fnm[param['vaxnum'] - 1], 'wb'))

    return result


# Run code using script
with open(sys.argv[1]) as f: # read text file
    lines = f.readlines()

ix = int(sys.argv[2]) # which line of text file (each line corresponds to one simulation, run in parallel), start at 1 so skip header
line = lines[ix]
L = line.split('\t') # each parameter in a line is separated by tab ie \t
args=[]
for l in L:
    if '[[' in l: # to handle rho matrix
        matrix = ast.literal_eval(l)
        m=[]
        for row in matrix:
            r = [float(el) for el in row]
            m.append(r)
        args.append(m)
    elif '.' in l:
        args.append(float(l))
    else:
        args.append(int(l))
result = runGCs(*args)