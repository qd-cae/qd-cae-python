

/* --------------------- NODE --------------------- */
const char* qd_node_class_docs = R"qddoc(

    Container for handling nodal data.

    Examples
    --------
        Get nodes by

        >>> femfile = D3plot("path/to/d3plot")
        >>> node_list = femfile.get_nodes()
        >>> node = femfile.get_nodeByID(1)
        >>> node = femfile.get_nodeByIndex(1)
)qddoc";

const char* node_get_id_docs = R"qddoc(
    get_id()

    Get the id of the node.

    Returns
    -------
    id : int
        id of the node

    Examples
    --------
        >>> d3plot.get_nodeByID(1).get_id()
        1
)qddoc";

const char* node_str_docs = R"qddoc(
    __str__()

    String representation of a node.

    Returns
    -------
    ret : str
        node as string

    Examples
    --------
        >>> str(femfile.get_nodeByIndex(0))
        '<Node id:463>'
)qddoc";

const char* node_get_coords_docs = R"qddoc(
    get_coords()

    Get the geometric nodal coordinates. One has to load the
    displacements for getting the coordinates at all timesteps.

    Returns
    -------
    coords : np.ndarray
        coordinate vector (nTimesteps x 3)

    Examples
    --------
        >>> d3plot = D3plot("path/to/d3plot")
        >>> node.get_coords().shape
        (1L, 3L)
        >>> # load disp
        >>> d3plot.read_states("disp")
        >>> node.get_coords().shape
        (34L, 3L)
)qddoc";

const char* node_get_disp_docs = R"qddoc(
    get_disp()

    Get the time series of the displacement vector.

    Returns
    -------
    disp : np.ndarray
        time series of displacements

    Examples
    --------
        >>> node.get_disp().shape
        (34L, 3L)
)qddoc";

const char* node_get_vel_docs = R"qddoc(
    get_vel()

    Get the time series of the velocity vector.

    Returns
    -------
    disp : np.ndarray
        time series of displacements

    Examples
    --------
        >>> node.get_disp().shape
        (34L, 3L)
)qddoc";

const char* node_get_accel_docs = R"qddoc(
    get_accel()

    Get the time series of the acceleration vector.

    Returns
    -------
    disp : np.ndarray
        time series of acceleration

    Examples
    --------
        >>> node.get_accel().shape
        (34L, 3L)
)qddoc";

const char* node_get_elements_docs = R"qddoc(
    get_elements()

    Get the elements of the node.

    Returns
    -------
    elements : list(Element)
        elements of the node

    Examples
    --------
        >>> len( node.get_elements() )
        4
)qddoc";

/* --------------------- ELEMENT --------------------- */
const char* element_description = R"qddoc(

    Examples
    --------
        Get elements by

        >>> from qd.cae.dyna import *
        >>> femfile = D3plot("path/to/d3plot")
        >>> element_list = femfile.get_elements()
        >>> shells = femfile.get_elements(Element.shell)
        >>> id = 1
        >>> element = femfile.get_elementByID(Element.solid, id)
)qddoc";

const char* element_type_docs = R"qddoc(
    Element types:
     - Element.type.none (only used for filtering)
     - Element.type.beam
     - Element.type.shell
     - Element.type.solid
     - Element.type.tshell
)qddoc";

const char* element_get_id_docs = R"qddoc(
    get_id()

    Get the id of the element.

    Returns
    -------
    id : int
        id of the element

    Examples
    --------
        >>> d3plot.get_elementByID(Element.shell, 1).get_id()
        1
)qddoc";

const char* element_str_docs = R"qddoc(
    __str__()

    String representation of an element.

    Returns
    -------
    ret : str
        element as string

    Examples
    --------
        >>> str(femfile.get_elementByIndex(Element.shell,0))
        '<Element type:2 id:1>'
)qddoc";

const char* element_get_plastic_strain_docs = R"qddoc(
    get_plastic_strain()

    Get the plastic strain of the element, if it was read with ``d3plot.read_states``.

    Returns
    -------
    plastic_strain : np.ndarray
        time series of plastic strain

    Examples
    --------
        >>> element.get_plastic_strain().shape
        (34L,)
)qddoc";

const char* element_get_energy_docs = R"qddoc(
    get_energy()

    Get the energy of the element, if it was read with ``d3plot.read_states``.

    Returns
    -------
    energy : np.ndarray
        time series of element energy

    Examples
    --------
        >>> element.get_energy().shape
        (34L,)
)qddoc";

const char* element_get_strain_docs = R"qddoc(
    get_strain()

    Get the strain tensor of the element, if it was read with ``d3plot.read_states``.
    The strain vector contains the matrix components: [e_xx, e_yy, e_zz, e_xy, e_yz, e_xz]

    Returns
    -------
    strain : np.ndarray
        time series of the strain tensor data

    Examples
    --------
        >>> element.get_strain().shape
        (34L, 6L)
)qddoc";

const char* element_get_stress_docs = R"qddoc(
    get_strain()

    Get the stress tensor of the element, if it was read with ``d3plot.read_states``.
    The stress vector contains the matrix components: [s_xx, s_yy, s_zz, s_xy, s_yz, s_xz]

    Returns
    -------
    stress : np.ndarray
        time series of the stress tensor data

    Examples
    --------
        >>> element.get_stress().shape
        (34L, 6L)
)qddoc";

const char* element_get_stress_mises_docs = R"qddoc(
    get_stress_mises()

    Get the mises stress of the element, if it was read with ``d3plot.read_states``.

    Returns
    -------
    stress : np.ndarray
        time series of the mises stress

    Examples
    --------
        >>> element.get_stress_mises().shape
        (34L,)
)qddoc";

const char* element_get_nodes_docs = R"qddoc(
    get_nodes()

    Get the nodes of the element.

    Returns
    -------
    nodes : list(Node)
        nodes of the element

    Examples
    --------
        >>> elem_nodes = element.get_nodes()
)qddoc";

const char* element_get_coords_docs = R"qddoc(
    get_coords()

    Get the elements coordinates (mean of nodes).

    Returns
    -------
    coords : np.ndarray
        coordinate vector (nTimesteps x 3)

    Notes
    -----
        Returns an empty vector, if the element has no
        nodes.

    Examples
    --------
        >>> d3plot = D3plot("path/to/d3plot")
        >>> element = d3plot.get_elementByIndex(Element.shell, 1)
        >>> element.get_coords().shape
        (1L, 3L)
        >>> # load disp
        >>> d3plot.read_states("disp")
        >>> element.get_coords().shape
        (34L, 3L)
)qddoc";

const char* element_get_history_docs = R"qddoc(
    get_history_variables()

    Get the loaded history variables of the element.

    Returns
    -------
    history_vars : np.ndarray
        time series of the loaded history variables

    Examples
    --------
        >>> d3plot = D3plot("path/to/d3plot",read_states="history 1 shell max")
        >>> d3plot.get_elementByID(Element.shell, 1).get_history_variables().shape
        (34L, 1L)

    Notes
    -----
        The history variable column index corresponds to the order in which
        the variables were loaded
)qddoc";

const char* element_get_estimated_size_docs = R"qddoc(
    get_estimated_size()

    Get the average element edge size of the element.

    Returns
    -------
    size : float
        average element edge size

    Examples
    --------
        >>> element.get_estimated_size()
        2.542
)qddoc";

const char* element_get_type_docs = R"qddoc(
    get_type()

    Get the type of the element.

    Returns
    -------
    element_type : Element.type
        Element.beam, Element.shell or Element.solid

    Examples
    --------
        >>> d3plot.get_elementByID(Element.beam, 1).get_type()
        type.beam
        >>> d3plot.get_elementByID(Element.shell, 1).get_type()
        type.shell
        >>> d3plot.get_elementByID(Element.solid, 1).get_type()
        type.solid
        >>> d3plot.get_elementByID(Element.tshell, 1).get_type()
        type.tshell
)qddoc";

const char* element_get_is_rigid_docs = R"qddoc(
    is_rigid()

    Get the status, whether the element is a rigid (flag for shells only).
    Rigid shells have no state data!

    Returns
    -------
    is_rigid : bool
        rigid status of the element

    Examples
    --------
        >>> d3plot = D3plot("path/to/d3plot", read_states="stress_mises max")
        >>> elem1 = d3plot.get_elementByID(Element.shell, 451)
        >>> elem1.is_rigid()
        False
        >>> elem1.get_stress_mises().shape
        (34L,)
        >>> elem2 = d3plot.get_elementByID(Element.shell, 9654)
        >>> elem2.is_rigid()
        True
        >>> elem2.get_stress_mises().shape # rigid shells lack state data
        (0L,)
)qddoc";

/* ----------------------- PART ---------------------- */
const char* part_get_id_docs = R"qddoc(
    get_id()

    Get the id of the part.

    Returns
    -------
    id : int
        id of the part

    Examples
    --------
        >>> d3plot = D3plot("path/to/d3plot")
        >>> part = d3plot.get_partByID(1)
        >>> part.get_id()
        1
)qddoc";

const char* part_get_name_docs = R"qddoc(
    get_name()

    Get the name of the part. It's the same name as in the input deck.

    Return
    -------
    name : str
        name of the part

    Examples
    --------
        >>> d3plot = D3plot("path/to/d3plot")
        >>> part = d3plot.get_partByID(1)
        >>> part.get_name()
        'PLATE_C'
)qddoc";

const char* part_get_nodes_docs = R"qddoc(
    get_nodes()

    Get the nodes of the part. Note that a node may belong to two parts,
    since only the elements are uniquely assignable.

    Returns
    -------
    nodes : list(Node)
        nodes belonging to the elements of the part

    Examples
    --------
        >>> d3plot = D3plot("path/to/d3plot")
        >>> part = d3plot.get_partByID(1)
        >>> len( part.get_nodes() )
        52341
)qddoc";

const char* part_get_elements_docs = R"qddoc(
    get_elements(element_filter=Element.none)

    Get the elements of the part.

    Parameters
    ----------
    element_filter : Element.type
        Optional element type filter. May be beam, shell or solid.

    Returns
    -------
    elements : list(Element)
        list of Elements

    Examples
    --------
        >>> d3plot = D3plot("path/to/d3plot")
        >>> part = d3plot.get_partByID(1)
        >>> len( part.get_elements() )
        49123
        >>> len( part.get_elements(Element.shell) )
        45123
)qddoc";

/* ----------------------- DB_NODES ---------------------- */
const char* dbnodes_description = R"qddoc(

    This class is managing the nodes internally in the
    background of a FEMFile. It can never be constructed
    or attained individually.

    Examples
    --------
        >>> # DB_Nodes is inherited by FEMFile
        >>> issubclass(FEMFile, DB_Nodes)
        True
        >>> # Use DB_Nodes functions
        >>> femfile = KeyFile("path/to/keyfile")
        >>> femfile.get_nNodes()
        45236

)qddoc";

const char* dbnodes_get_nodeByID_docs = R"qddoc(
    get_nodeByID(id)

    Parameters
    ----------
    id : int or list(int)
        node id or list of node ids

    Raises
    ------
    ValueError
        if ``idx`` does not exist.

    Returns
    -------
    node : Node or list(Node)
        requested Node(s)

    Examples
    --------
        >>> # get by single id
        >>> node = femfile.get_nodeByID(1)
        >>> # get a list of nodes at once
        >>> list_of_nodes = femfile.get_nodeByID( [1,2,3] )
)qddoc";

const char* dbnodes_get_nodeByIndex_docs = R"qddoc(
    get_nodeByIndex(index)

    Parameters
    ----------
    index : int or list(int)
        internal node index or list inf indexes

    Raises
    ------
    ValueError
        if ``index`` larger ``femfile.get_nNodes()``.

    Returns
    -------
    node : Node or list(Node)
        Node(s)

    Notes
    -----
        The internal index starts at 0 and ends at
        ``femfile.get_nNodes()``.

    Examples
    --------
        >>> # single index
        >>> node = femfile.get_nodeByIndex(1)
        >>> # get a list of nodes at once
        >>> list_of_nodes = femfile.get_nodeByIndex( [1,2,3] )
)qddoc";

const char* dbnodes_get_nodes_docs = R"qddoc(
    get_nodes()

    Returns
    -------
    nodes : list(Node)
        list of node objects

    Examples
    --------
        >>> list_of_nodes = femfile.get_nodes()
    )qddoc";

const char* dbnodes_get_nNodes_docs = R"qddoc(
    get_nNodes()

    Returns
    -------
    nNodes : int
        number of nodes in the file

    Examples
    --------
        >>> femfile.get_nNodes()
        43145
)qddoc";

/* ----------------------- DB_ELEMENTS ---------------------- */
const char* dbelems_description = R"qddoc(

    This class is managing the elements internally in the
    background of a FEMFile. It can never be constructed
    or attained individually.

    Examples
    --------
        >>> # FEMFile is a database of elements
        >>> issubclass(FEMFile, DB_Elements)
        True
        >>> # brief usage example
        >>> femfile = D3plot("path/to/d3plot")
        >>> femfile.get_nElements()
        45236
        >>> list_of_shells = femfile.get_elements(Element.shell)

)qddoc";

const char* get_elements_docs = R"qddoc(
    get_elements(element_filter=Element.none)

    Parameters
    ----------
    element_filter : Element.type
        Optional element type filter. May be beam, shell or solid.

    Returns
    -------
    elements : list(Element)
        list of Elements

    Raises
    ------
    ValueError
        if invalid ``element_filter`` is given.

    Notes
    -----
    Get the elements of the femfile. One may use a filter by type.

    Examples
    --------
        >>> all_elements = femfile.get_elements()
        >>> shell_elements = femfile.get_elements(Element.shell)
)qddoc";

const char* dbelems_get_nElements_docs = R"qddoc(
    get_nElements(element_filter=Element.none)

    Parameters
    ----------
    element_filter : Element.type
        Optional element type filter. May be beam, shell or solid.

    Raises
    ------
    ValueError
        if invalid ``element_filter`` is given.

    Returns
    -------
    nElements : int
        number of elements

    Examples
    --------
        >>> femfile.get_nElements()
        43156
)qddoc";

const char* dbelems_get_elementByID_docs = R"qddoc(
    get_elementByID(element_type, id)

    Parameters
    ----------
    element_type : Element.type
        type of the element. Must be beam, shell or solid.
    id : int or list(int)
        element id or list of ids

    Raises
    ------
    ValueError
        if invalid ``element_type`` is given
        or an ``id`` does not exist.

    Returns
    -------
    element : Element
        Element(s) depending on the arguments

    Notes
    -----
        Since ids in the dyna file are non unique for
        different element types, one has to specify the
        type too.

    Examples
    --------
        >>> # single element
        >>> elem = femfile.get_elementByID(Element.shell, 1)
        >>> # multiple elements
        >>> list_of_shells = femfile.get_elementByID(Element.shell, [1,2,3])
        >>> # whoever had the great id of non unique ids for elements ...
        >>> femfile.get_elementByID(Element.beam, 1).get_type()
        type.beam
        >>> femfile.get_elementByID(Element.solid, 1).get_type()
        type.solid
        >>> femfile.get_elementByID(Element.tshell, 1).get_type()
        type.solid
)qddoc";

const char* dbelems_get_elementByIndex_docs = R"qddoc(
    get_elementByIndex(element_type, index)

    Parameters
    ----------
    element_type : Element.type
        type of the element. Must be beam, shell or solid.
    index : int or list(int)
        element index or list of indexes

    Raises
    ------
    ValueError
        if invalid ``element_type`` is given
        or ``index`` does not exist

    Returns
    -------
    element : Element
        Element(s) depending on the arguments

    Examples
    --------
        >>> # single element
        >>> elem = femfile.get_elementByIndex(Element.shell, 0)
        >>> # multiple elements
        >>> list_of_shells = femfile.get_elementByID(Element.shell, [0,1,2])
        >>> 
        >>> femfile.get_elementByIndex(Element.beam, 1).get_type()
        type.beam
        >>> femfile.get_elementByIndex(Element.solid, 1).get_type()
        type.solid
        >>> femfile.get_elementByIndex(Element.tshell, 1).get_type()
        type.tshell
)qddoc";

/* ----------------------- DB_PARTS ---------------------- */
const char* dbparts_description = R"qddoc(

    This class is managing the parts internally in the
    background of a FEMFile. It can never be constructed
    or attained individually.

    Examples
    --------
        >>> # FEMFile is a database of parts
        >>> issubclass(FEMFile, DB_Parts)
        True
        >>> # brief usage example
        >>> femfile = D3plot("path/to/d3plot")
        >>> femfile.get_nParts()
        7
        >>> list_of_parts = femfile.get_parts()

)qddoc";

const char* dbparts_get_nParts_docs = R"qddoc(
    get_nParts()

    Returns
    -------
    nParts : int
        number of parts in the database

    Examples
    --------
        >>> femfile.get_nParts()
        7
)qddoc";

const char* dbparts_get_parts_docs = R"qddoc(
    get_parts()

    Returns
    -------
    parts : list(Part)
        list of all parts in the file

    Examples
    --------
        >>> list_of_all_parts = femfile.get_parts()
)qddoc";

const char* dbparts_get_partByID_docs = R"qddoc(
    get_partByID(id)

    Parameters
    ----------
    id : int or list of int
        id or ids of the part in the file

    Raises
    ------
    ValueError
        if some ``id`` does not exist.

    Returns
    -------
    parts : Part or list of Parts
        output depending on arguments

    Examples
    --------
        >>> part = femfile.get_partByID(1)
)qddoc";

const char* dbparts_get_partByIndex_docs = R"qddoc(
    get_partByIndex(index)

    Parameters
    ----------
    index : int or list of int
        index or indexes of the part in the file

    Raises
    ------
    ValueError
        if some ``index`` does not exist.

    Returns
    -------
    parts : Part or list of Parts
        output depending on arguments


    Examples
    --------
        >>> part = femfile.get_partByIndex(0)
)qddoc";

const char* dbparts_get_partByName_docs = R"qddoc(
    get_partByName(name)

    Parameters
    ----------
    name : str
        name of the part

    Raises
    ------
    ValueError
        if a part with ``name`` does not exist.

    Returns
    -------
    parts : Part
        part with the given name

    Examples
    --------
        >>> part = femfile.get_partByName("Lower Bumper")
)qddoc";

/* ----------------------- FEMFILE ---------------------- */
const char* femfile_get_filepath_docs = R"qddoc(
    get_filepath()

    Returns
    -------
    filepath : str
        Filepath of the femfile.

    Examples
    --------
        >>> femfile.get_filepath()
        "path/to/femfile"
)qddoc";

/* ----------------------- D3PLOT ---------------------- */
const char* d3plot_description = R"qddoc(

    A D3plot is a binary result file from LS-Dyna, a 
    commercial FEM-Solver from LSTC. The class reads
    the mesh and result data and makes it available 
    to the user.

    Notes
    -----
        The library focuses entirely on structural
        simulation results and does not support CFD
        or something else.
        A lot of checks take place during reading,
        therefore the library will complain if it does
        not support something (or it will fatally crash
        and kill your computer forever).

)qddoc";

const char* d3plot_constructor = R"qddoc(
    __init__(filepath, use_femzip=False, read_states=[])

    Parameters
    ----------
    filepath : str
        path to the d3plot
    use_femzip : bool
        whether to use femzip for decompression
    read_states : str or list of str
        read state information directly (saves time), 
        see the function ``read_states``

    Raises
    ------
    ValueError
        in case of an invalid filepath or locked file
    RuntimeError
        if anything goes wrong (internal checks) during reading

    Returns
    -------
        D3plot d3plot : instance

    Notes
    -----
        If LS-Dyna writes multiple files (one for each timestep),
        give the filepath to the first file. The library finds all
        other files.
        Please read state information with the read_states flag 
        in the constructor or with the member function.

    Examples
    --------
        Read the plain geometry data

        >>> d3plot = D3plot("path/to/d3plot")
        
        Read a compressed d3plot

        >>> d3plot = D3plot("path/to/d3plot.fz", use_femzip=True)
        
        Read d3plot with state data at once.

        >>> d3plot = D3plot("path/to/d3plot", read_states=["mises_stress max"])

)qddoc";

const char* d3plot_info_docs = R"qddoc(
    info()

    Prints a summary of the header data of the D3plot, which
    involves node info, element info, written state data and
    so forth.


    Examples
    --------
        >>> d3plot = D3plot("path/to/d3plot")
        >>> d3plot.info()
)qddoc";

const char* d3plot_get_title_docs = R"qddoc(
    get_title()

    Get the title of the d3plot, which is part
    of the header data.

    Returns
    -------
    title : str
        the title of the d3plot


    Examples
    --------
        >>> d3plot = D3plot("path/to/d3plot")
        >>> d3plot.get_title()
        "Barrier Impact"
)qddoc";

const char* d3plot_get_timesteps_docs = R"qddoc(
    get_timesteps()

    Get the simulation time of the written states.

    Returns
    -------
    timesteps : np.ndarray
        state timesteps of the D3plot

    Examples
    --------
        >>> d3plot = D3plot("path/to/d3plot")
        >>> time = d3plot.get_timesteps()
)qddoc";

const char* d3plot_get_nTimesteps_docs = R"qddoc(
    get_nTimesteps()

    Get the number of timesteps of the d3plot.

    Returns
    -------
    nTimesteps : int
        number of timesteps

    Examples
    --------
        >>> d3plot = D3plot("path/to/d3plot")
        >>> d3plot.get_nTimesteps()
        32
)qddoc";

const char* d3plot_read_states_docs = R"qddoc(
    read_states(vars)

    Parameters
    ----------
    vars : str or list(str)
        variable or list of variables to read (see Notes below)


    Notes
    -----
        Read a variable from the state files. If this is not done, the nodes
        and elements return empty vectors when requesting a result. The
        variables available are:
        
        * disp (displacement)
        * vel (velocity)
        * accel (acceleration)
        * strain [(optional) mode]
        * stress [(optional) mode]
        * stress_mises [(optional) mode]
        * plastic_strain [(optional) mode]
        * history [id1] [id2] [shell or solid] [(optional) mode]
        
        There is an optional mode for the element results. The results are 
        only given at the center of an element. Since shell elements
        have multiple layers of results, the optional mode determines the
        treatment of these layers:
        
        * inner (first layer)
        * mid (middle)
        * outer (last layer)
        * mean
        * max
        * min

    Examples
    --------
        >>> d3plot = D3plot("path/to/d3plot") # just geometry
        >>> node = d3plot.get_nodeByID(1)
        >>> len( node.get_disp() ) # no disps loaded
        0
        >>> # Read displacements
        >>> d3plot.read_states("disp")
        >>> len( node.get_disp() ) # here they are
        31
        >>> # multi-loading, already loaded will be skipped
        >>> d3plot.read_states(["disp","vel","stress_mises max","shell history 1 mean"])
        >>> # most efficient way, load the results directly when opening
        >>> D3plot("path/to/d3plot", read_states=["disp","vel","plastic_strain max"])
)qddoc";

const char* d3plot_clear_docs = R"qddoc(
    clear(vars)

    Parameters
    ----------
    vars : str or list(str)
        variable or list of variables to delete

    Notes
    -----
        This function may be used if one wants to clear certain state data
        from the memory. Valid variable names are:
        
        * disp
        * vel
        * accel
        * strain
        * stress
        * stress_mises
        * plastic_strain
        * history [(optional) shell or solid]
        
        The specification of shell or solid for history is optional. Deletes
        all history variables if none given.

    Examples
    --------
        >>> d3plot = D3plot("path/to/d3plot", read_states="strain inner")
        >>> elem = d3plot.get_elementByID(Element.shell, 1)
        >>> len( elem.get_strain() )
        34
        >>> # clear specific field
        >>> d3plot.clear("strain")
        >>> len( elem.get_strain() )
        0
        >>> # reread some data
        >>> d3plot.read_states("strain outer")
        >>> len( elem.get_strain() )
        34
        >>> d3plot.clear() # clear all
        >>> len( elem.get_strain() )
        0
)qddoc";

/* ----------------------- KEYFILE ---------------------- */
const char* keyfile_description = R"qddoc(

    A KeyFile is a textual input file for the FEM-Solver
    LS-Dyna from LSTC. The input file contains all the data 
    neccessary, such as nodes, elements, material and so on.

    Notes
    -----
        The library currently supports keyfiles only on a
        very primitive level. The mesh data is available
        in the same way as in a d3plot but everything 
        else is neglected (for now ... I'm lazy).

)qddoc";

const char* keyfile_constructor = R"qddoc(
    KeyFile(filepath, load_includes=True, encryption_detection=0.7)

    Parameters
    ----------
    filepath : str
        path to the keyfile
    load_includes : bool
        load all includes within the file
    encryption_detection : float
        detection threshold for encrypted include files. 
        Must be between 0 and 1.

    Raises
    ------
    ValueError
        in case of a wrong filepath or invalid encryption threshold
    RuntimeError
        if anything goes wrong during reading

    Returns
    -------
    keyfile : KeyFile
        instance

    Notes
    -----
        The argument ``encryption_detection`` is used to skip encrypted 
        include files. It is simply tested against the entropy of every
        include divided by 8 for normalization. Encrypted files usually
        have a very high entropy. The entropy of a file can be obtained 
        through the function ``qd.cae.dyna.get_file_entropy``.

    Examples
    --------
        >>> # load a keyfile
        >>> keyfile = KeyFile("path/to/keyfile")
        >>> # load keyfile without includes
        >>> keyfile = KeyFile("path/to/keyfile", load_includes=False)
        >>> # get some mesh data
        >>> node = keyfile.get_nodeByIndex(0)

)qddoc";

const char* module_get_file_entropy_description = R"qddoc(
    get_file_entropy(filepath)

    Parameters
    ----------
    filepath : str
        path to the file

    Returns
    -------
    entropy : float
        entropy of the file

    Notes
    -----
        The shannon entropy of a file describes the 
        randomness of the bytes in the file. The value is
        limited between 0 and 8, where 0 means 
        entirely structured and 8 means the file 
        is entirely random or encrypted.

    Examples
    --------
        >>> get_file_entropy("path/to/encrypted_file")
        7.64367
        >>> get_file_entropy("path/to/text_file")
        3.12390
)qddoc";

/* ----------------------- RAW D3PLOT ---------------------- */

const char* rawd3plot_constructor_description = R"qddoc(
    RawD3plot(filepath, use_femzip=False)

    Parameters
    ----------
    filepath : str
        path to the file
    use_femzip : bool
        whether to use femzip decompression

    Returns
    -------
    instance : RawD3plot

    Raises
    ------
    ValueError
        in case of an invalid filepath or locked file
    RuntimeError
        if anything goes wrong (internal checks) during reading

    Notes
    -----
        If LS-Dyna writes multiple files (one for each timestep),
        give the filepath to the first file. The library finds all
        other files. The class automatically reads all data!

    Examples
    --------
        >>> from qd.cae.dyna import RawD3plot
        >>> # read an arbitrary d3plot
        >>> raw_d3plot = RawD3plot("path/to/d3plot")
        >>> #read femzip compressed file
        >>> raw_d3plot = RawD3plot("path/to/d3plot.fz",use_femzip=True)
)qddoc";

const char* rawd3plot_get_int_names_docs = R"qddoc(
    get_int_names()

    Returns
    -------
    names : list of str
        names of the integer variables in the d3plot

    Notes
    -----
        The variable arrays themselves can be obtained by
        the member function 'RawD3plot.get_int_data'.

    Examples
    --------
        >>> raw_d3plot.get_int_names();
        ['elem_beam_data', 'elem_beam_ids', 'elem_shell_data', 'elem_shell_ids', ...]
)qddoc";

const char* rawd3plot_get_int_data_docs = R"qddoc(
    get_int_data(name)

    This function is for reading any data, which is saved as an
    integer value.

    Parameters
    ---------
    name : str
        name of data array to request for

    Returns
    -------
    data : numpy.ndarray
        data array

    Examples
    --------
        >>> # check which vars are available
        >>> raw_d3plott.get_int_names();
        ['elem_beam_data', 'elem_beam_ids', 'elem_shell_data', 'elem_shell_ids', ...]
        >>> # request some data
        >>> raw_d3plot.get_int_data("elem_shell_data").shape
        (4969, 5)
        >>> # 4969 shell elements, 4 node ids and 1 material index (not id!)
)qddoc";

const char* rawd3plot_get_string_names_docs = R"qddoc(
    get_string_names()

    Returns
    -------
    names : list of str
        names of all string variables in the d3plot

    Examples
    --------
        >>> raw_d3plot.get_string_names()
        ['part_names']
)qddoc";

const char* rawd3plot_get_string_data_docs = R"qddoc(
    get_string_data(name)

    This function is for reading any data, which is saved as a
    string in the d3plot.

    Parameters
    ---------
    name : str
        name of data array to request for

    Returns
    -------
    data : list of str
        string data list

    Examples
    --------
        >>> raw_d3plot.get_string_names()
        ['part_names']
        >>> # also the part names are raw (untrimmed)
        >>> raw_d3plot.get_string_data("part_names")
        ["SomePart                                                                ']
)qddoc";

const char* rawd3plot_get_float_names_docs = R"qddoc(
    get_float_names()

    Returns
    -------
    names : list of str
        names of all float variables in the d3plot

    Examples
    --------
        >>> raw_d3plot.get_float_names()
        ['elem_shell_results', 'elem_shell_results_layers', 'elem_solid_results', ... ]
)qddoc";

const char* rawd3plot_get_float_data_docs = R"qddoc(
    get_float_data(name)

    This function is for reading any data, which is saved as a
    floating point value.

    Parameters
    ---------
    name : str
        name of data array to request for

    Returns
    -------
    data : numpy.ndarray
        data array

    Examples
    --------
    >>> raw_d3plot.get_float_names()
    ['elem_shell_results', 'elem_shell_results_layers', 'elem_solid_results', ... ]
    >>> raw_d3plot.get_float_data("elem_shell_results").shape
    (12, 4696, 24)
    >>> # 12 timesteps, 4969 elements and 24 variables
)qddoc";

const char* rawd3plot_info_docs = R"qddoc(
    info()

    Prints a summary of the header data of the d3plot, which
    involves node info, element info, written state data and
    so forth.

    Examples
    --------
        >>> raw_d3plot = RawD3plot("path/to/d3plot")
        >>> raw_d3plot.info()
)qddoc";
