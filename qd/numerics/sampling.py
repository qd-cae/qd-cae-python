
import numpy as np
from diversipy.hycusampling import improved_lhd_matrix


def uniform_lhs(nSamples, variables, **kwargs):
    '''Do a uniform Latin Hypercube Sampling

    Parameters
    ----------
    nSamples : int
        number of samples to draw
    variables : dict(str, tuple(mean,delta) )
        variable dictionary, the key must be the name whereas the value
        must be a tuple. The first value of the tuple is the lower bound
        for the variable and the second one is the upper bound
    **kwargs
        arguments passed on to diversipy.hycusampling.improved_lhd_matrix

    Returns
    -------
    column_names, samples : list(str), np.ndarray
        returns as first entry a list with the column names for the LHS
        and as second argument a numpy array with latin hypercube samples 
        ( shape is nSamples x len(variables) ) 
    '''

    assert isinstance(nSamples, int)
    assert isinstance(variables, dict)
    assert all( isinstance(var_name, str) for var_name in variables.keys() )
    assert all( isinstance(entry, (tuple,list,np.ndarray)) 
                                  for entry in variables.values() )

    variable_labels = sorted( variables.keys() )

    # extract variable limits
    vars_bounds = np.vstack( variables[label] for label in variable_labels )

    # lhs sampling in a unit square
    #data = maximin_reconstruction(nSamples, len(variable_labels), **kwargs)
    data = improved_lhd_matrix(nSamples, len(variable_labels), **kwargs)

    # adapt to variable limits: [0;1] -> [min, max]
    vars_min = vars_bounds[:,0]
    vars_max = vars_bounds[:,1]
    for iRow in range(data.shape[0]):
        data[iRow] = (vars_max-vars_min)*data[iRow]+vars_min

    return variable_labels, data