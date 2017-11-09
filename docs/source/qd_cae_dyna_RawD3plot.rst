 
RawD3plot
------

A ``RawD3plot`` is reading all of the raw data within a d3plot file. In contrast to the other ``D3plot`` class does it give access to the unprocessed data within the file.

There are two purposes why this class was created. Firstly one can use and check the raw data oneself and do all kind of magic things with it. Secondly not using object orientation should speed up the reading process by avoiding many small allocations. 

The downside of the raw data access is the raw data itself. A d3plot has a very confusing structure. The file sometimes omits certain array elements and builds arrays in a very strange manner. Because the raw data arrays can be very confusing, one should not use this class without the `official LS-Dyna database guide`_.

.. _official LS-Dyna database guide: https://github.com/qd-cae/qd-cae-python/tree/master/docs

**This class is not finished and will change!** The raw data arrays are simply too confusing and need a further utility layer.

---------

These are all the data arrays available with their shape description. They are categorized according to the variable type.

**Float Data**:
 - timesteps *(nTimesteps)*
 - node_coordinates *(nTimesteps x nNodes x 3)*
 - node_displacement *(nTimesteps x nNodes x 3)*
 - node_velocity *(nTimesteps x nNodes x 3)*
 - node_acceleration *(nTimesteps x nNodes x 3)*
 - elem_solid_results *(nTimesteps x nSolids x nResults)*
 - elem_solid_deletion_info *(nTimesteps x nSolids)*
 - elem_shell_results_layers *(nTimesteps x nShells x nLayers x nResults)*
 - elem_shell_results *(nTimesteps x nShells x nResults)*
 - elem_shell_deletion_info *(nTimesteps x nShells)*
 - elem_tshell_results_layers *(nTimesteps x nTShells x nLayers x nResults)*
 - elem_tshell_results *(nTimesteps x nTShells x nResults)*
 - elem_tshell_deletion_info *(nTimesteps x nTShells)*
 - elem_beam_results *(nTimesteps x nBeams x nResults)*
 - elem_beam_deletion_info *(nTimesteps x nBeams)*
 - airbag_geom_state_float_results *(nTimesteps x nAirbags x nResults)*
 - airbag_particle_float_results *(nTimesteps x nParticles x nResults)*

**Integer Data**:
 - node_ids *(nNodes)*
 - elem_solid_ids *(nSolids)*
 - elem_solid_data *(nSolids x 9)* (contains nodes and material)
 - elem_shell_ids *(nShells)*
 - elem_shell_data *(nShells x 5)* (contains nodes and materials)
 - elem_tshell_ids *(nTShells)*
 - elem_tshell_data *(nTShells x 9)* (contains nodes and materials)
 - elem_beam_ids *(nBeams)*
 - elem_beam_data *(nBeams x 6)* (contains nodes and materials)
 - part_ids *(nParts)*
 - material_type_numbers *(nMaterials)*
 - airbag_geometry *(nAirbags x 4 or 5)*
 - airbag_variable_type_flag *(internally used only)*
 - airbag_geom_state_int_results *(nTimesteps x nAirbags x 1)* 
 - airbag_particle_int_results *(nTimesteps x nParticles x 3)*

**String Data**:
 - part_names *(nParts)*
 - airbag_all_variable_names 
 - airbag_geom_names (names for airbag_geometry)
 - airbag_geom_state_int_names (names for airbag_geom_state_int_results)
 - airbag_geom_state_float_names (names for airbag_geom_state_float_results)
 - airbag_particle_int_names (names for airbag_particle_int_results)
 - airbag_particle_float_names (names for airbag_particle_float_results)

---------

.. autoclass:: qd.cae.dyna.RawD3plot
    :members:
    :inherited-members:
    :private-members: 

    .. automethod:: __init__

    