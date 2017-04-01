
import numpy as np
from diversipy.hycusampling import maximin_reconstruction


def uniform_lhs(nSamples, variables, **kwargs):
    '''Do a uniform Latin Hypercube Sampling

    Parameters
    ----------
    nSamples : int
        number of samples to draw
    variables : dict(str, tuple(lower, upper) )
        variable dictionary, the key must be the name whereas the value
        must be a tuple. The first value of the tuple is the lower bound
        for the variable and the second one is the upper bound
    **kwargs
        arguments passed on to diversipy.hycusampling.maximin_reconstruction

    Returns
    -------
    column_names : list(str)
        list with the column names for the LHS
    samples : np.ndarray
        numpy array with latin hypercube samples. Shape is nSamples x len(variables).

    Examples
    --------
        >>> from qd.numerics.sampling import uniform_lhs
        >>> 
        >>> variables = {'length':[0,10], 'angle':[-3,3]}
        >>> labels, data = uniform_lhs(nSamples=100, variables=variables)
        >>> labels
        ['angle', 'length']
        >>> data.shape
        (100, 2)
        >>> data.min(axis=0)
        array([-2.98394928,  0.00782609])
        >>> data.max(axis=0)
        array([ 2.8683843 ,  9.80865352])
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
    data = maximin_reconstruction(nSamples, len(variable_labels), **kwargs)

    # adapt to variable limits: [0;1] -> [min, max]
    vars_min = vars_bounds[:,0]
    vars_max = vars_bounds[:,1]
    for iRow in range(data.shape[0]):
        data[iRow] = (vars_max-vars_min)*data[iRow]+vars_min

    return variable_labels, data