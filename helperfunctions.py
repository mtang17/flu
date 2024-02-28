import os
import time
import random
import numpy as np
import pickle
import gc
from numpy.random import multivariate_normal
from scipy.optimize import fsolve
from scipy.sparse import csr_matrix
from copy import deepcopy

def fsolve_mult(f, guess = 1.1):
    """
    Scipy fsolve doesn't always work. Try fsolve with many
    different initial guesses.
    """
    r = fsolve(f, guess)
    num_tries = 0
    while f(r) > 0.05:
        guess += 0.2
        r = fsolve(f, guess)
        num_tries += 1
        if guess > 10:
            guess = -10
        if num_tries > 2000:
            raise ValueError('fsolve tried too many times')
    return r

def flatten1D(x):
    """
    Flatten to 1D matrix.
    """
    transposed = np.transpose(x, (2, 1 ,0))
    return np.reshape(transposed, (x.shape[2], -1), order = 'F')

def flatten2D(x):
    """
    Flatten to 2D matrix.
    """
    transposed = np.transpose(x, (1, 0, 2))
    return np.reshape(transposed, (x.shape[1], -1), order = 'F').T

def buildup3D(A, d3):
    """
    Return to 3D matrix.
    """
    reshape = np.reshape(A.T, (A.T.shape[0], -1, d3), order = 'F')
    return np.transpose(reshape, (1, 0, 2))

def findlast(x):
    """
    Find last nonzero element of matrix.
    """
    reshaped = np.reshape(x, -1, order = 'F')
    return np.nonzero(reshaped)[0][-1]

def compress(dictionary):
    """
    Turn arrays in a dictionary into sparse matricies if fraction of
    nonzero elements is less than 0.5. Scipy cannot turn 3D matrices 
    into sparse matrices, so they are flattened and the shape is 
    stored in the dictionary.
    """
    newdict = deepcopy(dictionary)
    for key, value in dictionary.items():
        if isinstance(value, np.ndarray):
            if np.nonzero(value)[0].size / value.size < 0.5:
                newdict[key + 'shape'] = value.shape
                newdict[key] = csr_matrix(value.flatten())      
        elif isinstance(value, dict):
            newdict[key] = compress(value)
    return newdict

def expand(dictionary):
    """
    Expand a compressed dictionary. Flattened arrays are reshaped using the stored shape.
    """
    newdict = deepcopy(dictionary)
    for key, value in dictionary.items():
        if isinstance(value, csr_matrix):
            newdict[key] = np.reshape(value.toarray(), dictionary[key + 'shape'])
        elif isinstance(value, dict):
            newdict[key] = expand(value)
    return newdict

def find(array, mode = 'zero'):
    """
    Print indices with a single index, used for debugging against matlab.
    Only for 1d/2d/3d matrices
    """
    assert len(array.shape) in [0, 1, 2, 3]
    assert mode in ['zero', 'nan', 'bool']
    if len(array.shape) == 0:
        return np.empty(0)
    functions = {'zero': np.nonzero,
                 'nan': lambda x: np.where(~np.isnan(x)),
                 'bool': lambda x: np.where(x == True)}
    l3 = lambda x, y: np.sort(x[1] * y.shape[0] + x[0] + y.shape[0] * y.shape[1] * x[2])
    linearinds = {1: lambda x, y: x[0],
                  2: lambda x, y: np.sort(x[1] * y.shape[0] + x[0]),
                  3: l3}
    inds = functions[mode](array)
    return linearinds[len(array.shape)](inds, array)


def nrand(size):
    """
    Generate normal random numbers from uniform random numbers. Used for debugging
    """
    assert isinstance(size, int) is True
    newsize = size if size % 2 == 0 else size + 1
    a = np.random.uniform(size = newsize)
    u1 = a[: int(len(a)/2)]
    u2 = a[int(len(a)/2):]
    z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
    return np.concatenate([z1, z2])[: size]


def mvnrand(seed, mu, sigma, size = 1):
    """
    Generate multivariate random numbers. Used for debugging
    """
    assert seed > 0
    np.random.seed(seed)
    L = np.linalg.cholesky(sigma)

    if size == 0:
        return np.empty(shape = (0, len(mu)))
    else:
        rands = np.reshape(nrand(size = int(len(mu) * size)), (len(mu), size), order = 'F')
        return mu + (L @ rands).T


def randuniform(seed, size = 1):
    """
    Sample uniform random numbers in same way that matlab does with
    a certain random seed. Used for debugging
    """
    assert seed > 0
    assert isinstance(size, int) or isinstance(size, tuple)
    
    np.random.seed(seed)
    if isinstance(size, tuple):
        assert len(size) in [0, 1, 2, 3]
        if len(size) == 0:
            return np.empty(shape = 0)
    if isinstance(size, int) or (isinstance(size, tuple) and len(size) == 1):
        out = np.random.uniform(size = size)
        if len(out) == 1:
            return out[0]
        else:
            return out
    if len(size) == 2:
        x, y = size
        return np.reshape(np.random.uniform(size = size), (y, x)).T
    elif len(size) == 3:
        x, y, z = size
        return np.reshape(np.random.uniform(size = size), (z, y, x)).T
    
def matlab_percentile(in_data, percentiles):
    """
    Calculate percentiles in the way IDL and Matlab do it.

    By using interpolation between the lowest an highest rank and the
    minimum and maximum outside.

    Parameters
    ----------
    in_data: numpy.ndarray
        input data
    percentiles: numpy.ndarray
        percentiles at which to calculate the values

    Returns
    -------
    perc: numpy.ndarray
        values of the percentiles
    """

    data = np.sort(in_data)
    p_rank = 100.0 * (np.arange(data.size) + 0.5) / data.size
    perc = np.interp(percentiles, p_rank, data, left=data[0], right=data[-1])
    return perc

def prctile(arr, percentile, axis = 1):
    """
    Do matlab_percentile for a matrix.
    """
    if len(arr.shape) == 2:
        if axis == 0:
            return np.array([matlab_percentile(row, percentile) for row in arr])
        elif axis == 1:
            return np.array([matlab_percentile(row, percentile) for row in arr.T])
        else:
            raise ValueError('prctile does not do matrices more than 2D')
    elif len(arr.shape) == 1 and arr.shape[0] != 0:
        return matlab_percentile(arr, percentile)
    elif len(arr.shape) == 1 and arr.shape[0] == 0:
        return np.array([np.nan])
    
def invind(inds, nrows):
    """
    Get 2D indices from linear indices and number of rows of matrix
    """
    j = np.floor(inds / nrows).astype(int)
    i = inds - j * nrows
    return (i, j)
    
    
def reshape_(x, row = False):
    """
    If vector, then reshape to a 1D column matrix.
    """
    lshape = len(x.shape)
    if lshape == 0:
        raise ValueError('Empty vector')
    elif lshape == 1 and row is False:
        return np.reshape(x, (-1, 1))
    elif lshape == 1 and row is True:
        return np.reshape(x, (1, -1))
    return x


def matind(x, y):
    """
    Get linear indices for 2D indexing the way matlab does
    """
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    xp = np.repeat(x, y.size).astype(int)
    yp = np.tile(y, x.size).astype(int)
    return xp, yp

def matindget(A, x, y):
    """
    Apply linear indices from matind to a matrix.
    """
    xp, yp = matind(x, y)
    return np.reshape(A[xp, yp], (x.size, y.size))

def any_(x):
    """
    Equivalent of any function in Matlab.
    """
    if (isinstance(x, int) or isinstance(x, float)) and (x != 0):
        return 1
    elif (isinstance(x, int) or isinstance(x, float)) and (x == 0):
        return 0
    if len(x.shape) > 1:
        first_ind = np.where(np.array(x.shape) > 1)[0][0]
        y = np.zeros(shape = x.shape[first_ind + 1:])
        inds = np.nonzero(x)[first_ind + 1:]
        if len(inds) > 0:
            y[np.nonzero(x)[first_ind + 1:]] = 1
        return y
    else:
        if len(np.nonzero(x)[0]) > 0:
            return 1
        else:
            return 0
    return y


def alignLeft(arr, trim):
    """
    Summary
     Helper function that takes in an array as input and then aligns non-zero 
     columns to the left.  

    Inputs
     arr: 2-D array. Each column must be either all-zeros or non-zeros. 
     trim: 0 or 1. If 0, the array is trimed at the last non-zero column

    Outputs
     arr: Left-aligned and/or trimmed array
    """
    
    temp = np.nonzero(arr[0,:])[0]
    if trim:
        arr = arr[:, temp]
    else:
        arr[:, :len(temp)] = arr[:, temp]
        arr[:, len(temp):] = 0
        
    return arr


def bit2int(X, n):
    """
    Summary:
     Converts n-digit binaries into integers. See also int2bit

    Inputs:
     X: Matrix whose columns represent n-digit binary numbers
     n: positive integer

    Output:
     Y: Row vector equal to the number of columns of X, whose elements are
     integers corresponding to the columns of X.
    """
    s1, s2 = X.shape
    if n > s1:
        raise ValueError('n is greater than vector length')
    Y = np.zeros(s2)
    for i in range(s2):
        Y[i] = np.sum(np.exp2(np.arange(n-1, -1, -1)) * X[:n, i].T)
    return Y


def int2bit(X, n):
    """
    Summary:
      Converts integers into n-digit binary. See also bit2int
    Inputs:
      X: vector of non-negative integers, either a row or a column
      n: positive integer
    Output:
      Y: Matrix of size n-by-length of X. Each column corresponds to a
      n-digit binary number that corresponds to each element of X.
    """
    X = np.reshape(X, (-1, 1))
    Y = np.remainder(np.floor(X / ( np.exp2(np.arange(n-1, -1, -1)))), 2).T
    return Y

def res2ind(gcBcellNum, residues, param):
    """
    Summary:
      Helper function that allows easy indexing of gcBcellMutations array.
      It has two uses: 
       (1) If either all, first half, or second half of the residue
           indices for a group of B cell is wanted
       (2) If the index of specific residues of a B cell is wanted

    Inputs:
     For use (1): 
      gcBcellNum: row of indices of one or more B cells from same GC
      residues: only 1:80, 1:40, 41:80 are allowed
     For use (2):
      gcBcellNum: single index of a B cell
      residues: array of any numbers between 1 and 80
     Common:
      param: parameter struct

    Outputs: 
      k: a row array containing desired indices of the residues
    """
    cond1 = np.array_equal(residues, np.arange(param['n_res']))
    cond2 = np.array_equal(residues, np.arange(param['n_res']) / 2)
    cond3 = np.array_equal(residues, np.arange(param['n_res']  / 2, param['n_res']))
    
    if isinstance(gcBcellNum, int):
        gcBcellNum = np.array([gcBcellNum])
        
    # First use
    if cond1 or cond2 or cond3:
        # 0 used to be a 1 XXX
        k = np.tile((gcBcellNum.T - 0) * param['n_res'], (len(residues), 1)).T + residues
        k = np.reshape(k.T, (1, -1), order = 'F')
    # Second use
    else:
        k = (gcBcellNum - 0) * param['n_res'] + residues
    return k.astype(int)

def checkMutationScheme(gcBells, k, i, naiveBcells, mutations, gcBcellsMutation, M_GC, param):
    """
    Summary:
     For a specific GC B cell, check if the affinities are correct, by
     comparing it against affinities calculated from the mutation state and
     corresponding fitness landscape.

    Outputs:
     flag: 1 if the affinities are not calculated correctly.

    Inputs:
     gcBcells, naiveBcells: Arrays of GC and naive B cells
     k, i: Index of GC, Index of B cell
     mutations: 1x2 cell array containing mutation sizes
     gcBcellsMutation: 2D array containing the mutation states of GC B cells
     M_GC: Number of GCs 
     param: parameter struct
    """
    flag = 0
    BcellMutations = gcBcellsMutation[k, convertMutNum(i, np.arange(param['n_res']), param)]
    lineage = gcBcells[k, i]
    E_from_mut = np.zeros(2)
    term1 = naiveBcells[k + M_GC * 2, lineage]
    term2 = sum(mutations[0][k, lineage, BcellMutations == 1])
    E_from_mut[0] = term1 - term2
    E_from_mut[1] = param['f0'] - sum(mutations[1][k, lineage, BcellMutations == 1])
    
    if round(E_from_mut[0] * 1000) != round(gcBcells(k + 2 * M_GC, i) * 1000):
        flag = 1
        print(np.nonzero(BCellMutations)[0])
        raise ValueError(f'WT affinity not correct; k={k} i={i}')
    if round(E_from_mut[1] * 1000) != round(gcBCells[k + 3 * M_GC, i] * 1000):
        flag = 1
        print(np.nonzero(BCellMutations)[0])
        raise ValueError(f'Variant affinity not correct; k={k} i={i}')
    
def initializeParameters(vaxnum,
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
      Initializes the variables based on inputs.
      Also initialize parameters whose values are fixed. 

    Inputs:
      vaxnum: Integer between 1 and 4; indicates dose number of current simulation
      E1h, dE12, dE13: Defines naive B cell germline affinities
      p2, p3: Between 0 and 1. Fraction of naive B cells that target epitope 2 and epitope 3 respectively
      masking: 1 if epitope masking, 0 if not
      C0: Reference antigen concentration
      w1: 0 if this study. If not, then alternative Ag capture model is used. 
          Defines saturation of antigen capture. See Yang, et al. (2023) for alternate model.
      w2: Selection stringency. 0.5 in this paper.
      q12, q13, q23: 0 if no masking. Varied between 0 and 1 to define epitope overlap when masking is 1
      memToGCFrac: Fraction of pre-existing memory cells that can enter GC. 0 in this study.
      outputprob: Fraction of selected GC B cells that exit. 0.05 in this study.
      outputpcfrac: Fraction of exiting GC B cells that become plasma cell. 0.1 in this study.
      rho: 2x2 matrix to define epitope conservation of subdominant epitopes between vaccine and other strain.
           Rows correspond to epitope (epitope 2 and 3) and column corresponds to strain (strain 2 and 3).
           e.g. row 0, column 1 corresponds to how conserved epitope 2 is between the vaccine strain and strain 3.
      earlybooster: 0 in this study.
      tmax: Time duration of simulation. 360 in this study.
      first: First index of GCs (eg. 201)
      last: Last index of GCs (eg. 400) 

    Output:
      param: Struct having parameters and constants as fields
             See the body of the code for the fields and explanations
    """
    param = {"first": first,
             "last": last,
             "M_GC": last-first+1, #number of GCs
             "N_GC_MAX": 3000, #defines the maximal size of array of all GC B cells
             "N_PC_MAX": 20000, #defines the maximal size of array of all plasma cells
             "N_PB_MAX": 1000, #defines the maximal size of array of all plasmablasts
             "naivefieldnum": 8, #number of characteristics tracked for every naive B cell
             "gcfieldnum": 6, #number of characteristics tracked for every GC B cell
             "pcfieldnum": 7, #number of characteristics tracked for every plasma cell
             "memfieldnum": 10, #number of characteristics tracked for every memory B cell
             "vaxnum": vaxnum,
             "T": 0, #bolus shot
             "k": 0, #bolus shot
             "numshot": 1, #bolus shot
             "earlybooster": earlybooster,
             "E1h": E1h,
             "dE12": dE12,
             "dE13": dE13,
             "p2": p2,
             "p3": p3,
             "n_ep": 3, #number of epitopes
             "n_var": 3, #number of variants
             "NaiveMax": 2010, #defines maximal size of array of all naive B cells in one GC
             "memToGCFrac": memToGCFrac,
             "MemoryReentryMax": 200,
             "production": 1,
             "delay": 2,
             "masking": masking,
             "q12": q12,
             "q13": q13,
             "q23": q23,
             "outputprob": outputprob,
             "outputpcfrac": outputpcfrac,
             "mutprob": 1,
             "IgM0": 0.01,
             "Ag0": 10,
             "Ageff": 0.01,
             "C0": C0,
             "F0": np.nan,
             "dose": np.array([]),
             "dose_t": np.array([]),
             "tmax": tmax,
             "dt": 0.01,
             "f0": 6,
             "activation_threshold": 1,
             "w1": w1,
             "w2": w2,
             "MutationPDF": np.array([3.1, 1.2, 3.08]),
             "rho": rho,
             "n_res": 80,
             "k_deposit": 24,
             "d_Ag": 3,
             "d_IC": 0.15,
             "d_IgM": np.log(2)/28,
             "d_IgG": np.log(2)/28,
             "d_pc": np.log(2)/4,
             "lambdamax": 1,
             "betamax": 2.5,
             "betamaxEGC": 2.5,
             "mu": 0.5,
             "Nmax": 10,
             "numTmax": 1200,
             "naiveprolnum": 4}
    
    param['r_IgG'] = param['IgM0'] * param['production'] / param['M_GC']
    param['r_IgM'] = param['IgM0'] * param['production'] / param['M_GC']
    tspan_dt = 25 * param['dt']
    param["tspan_summary"] = np.arange(0,  tmax + tspan_dt, tspan_dt)
        
    return param

def getFileLocation(param):
    """
    Summary:
     Get the location/name of data files based on the parameters
    Outputs:
     fnm: 1 x vaxnum cell array. Contains the file location/name of data
          files for all previous vaccinations and current vaccination.
    Inputs: Parameter struct
    """
    data_locs = ['Data_Prime', 'Data_Secondary', 'Data_Tertiary', 'Data_Vax4']
    for i in range(param['vaxnum']):
        if param['memToGCFrac'] > 0 and i > 1: # if reentry of memory cells is allowed
            val = param['memToGCFrac']
            data_locs[i] = f'memToGC/memToGCFrac_{val:.3f}/{data_locs[i]}'
        if param['earlybooster'] and i == 3: # If Vax3 is given early
            data_locs[i] = f'earlyBooster/{data_locs[i]}'
        if param['masking']: # If epitope overlap exists
            val1 = param['q12']
            val2 = param['q13']
            val3 = param['q23']
            data_locs[i] = f'steric/steric_{val1:.2f}_{val2:.2f}_{val3:.2f}/{data_locs[i]}'
        if param['w1'] > 0: # If alternative model of Ag capture is used
            data_locs[i] = f'agCaptureSaturation/{data_locs[i]}'
        data_locs[i] = f'Flu_Data_Github/{data_locs[i]}' #DIRECTORY IN WHICH DATA PICKLE DATA FILES ARE SAVED
    
    inputs = [param['vaxnum'],
              param['E1h'],
              param['dE12'],
              param['dE13'],
              param['p2'],
              param['p3'],
              param['masking'],
              param['C0'],
              param['w1'],
              param['w2'],
              param['q12'],
              param['q13'],
              param['q23'],
              param['outputprob'],
              param['outputpcfrac'],
              param['rho'],
              param['tmax']]
    
    inputs = [str(item) for item in inputs]
    vaxtiming = np.array([0, 360, 360, 360]) # days, default vaccination schedule CHECK CHECK CHECK
    
    dirnm = [[]] * param['vaxnum']
    fnm = [[]] * param['vaxnum']
    idx_vaxnum = param['vaxnum'] - 1
    dirnm[idx_vaxnum] = f'./{data_locs[idx_vaxnum]}/{"_".join(inputs)}' #FILE PATH OF SAVED DATA
    timing = vaxtiming[idx_vaxnum] * (param['earlybooster'] == 0) + param['earlybooster']
    
    if param['vaxnum'] > 1:
        idx2_vaxnum = idx_vaxnum - 1
        strlst = [str(idx2_vaxnum + 1)] + inputs[1:-1] + [str(timing)]
        dirnm[idx2_vaxnum] = f'./{data_locs[idx2_vaxnum]}/{"_".join(strlst)}' #FILE PATH OF SAVED DATA
    
    if not os.path.exists(dirnm[idx_vaxnum]): #to handle possible errors when running the simulations in parallel on the cluster
        try:
            os.makedirs(dirnm[idx_vaxnum])
        except:
            print('error making directory')
        
    fstring = f'{param["first"]}_to_{param["last"]}.pkl'
    fnm = [f'{name}/{fstring}' for name in dirnm]
    
    if os.path.isfile(fnm[idx_vaxnum]):
        print('Warning: File already exists')
    
    return fnm
