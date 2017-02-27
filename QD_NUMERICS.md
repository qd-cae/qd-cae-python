
# Module qd.numerics

This module contains additional numerical functions.

functions: None

submoduls:
  - sampling

---------
# Sampling

This module contains functions related to sampling and DOEs.

## ```uniform_lhs(nSamples, variables, **kwargs)```

Do a uniform latin hypercube sampling.

Parameters:
  - int nSamples : number of samples to draw
  - dict(str, list(float,float)) variables : variables dictionary, the key must be the variable name and the value must be a tuple or list of the lower and upper bound for the variable
  - **kwargs : arguments passed on to from diversipy.hycusampling.improved_lhd_matrix

Returns:
  - list(str) column_names, np.ndarray samples : returns as first entry a list with the column names for the LHS and as second argument a numpy array with latin hypercube samples

Example:
```python
from qd.numerics.sampling import uniform_lhs
nSamples = 100
variables = {"var1":[0,1], "var2":[-4,4] }
column_names, samples = uniform_lhs(nSamples, variables)
```