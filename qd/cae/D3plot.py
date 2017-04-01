

from ._dyna_utils import plot_parts, _parse_element_result, _extract_elem_coords
from .dyna_cpp import QD_D3plot, QD_Part
from .Part import Part

import os
import numpy as np

class D3plot(QD_D3plot):
    '''
    Notes
    -----
        Class for reading a D3plot. A D3plot is a binary result file 
        written from LS-Dyna and contains the simulation mesh, as well
        as the time step data.
    '''

    def __init__(self, *args, **kwargs):
        '''Constructor for a D3plot.

        If LS-Dyna writes multiple files (one for each timestep),
        give the filepath to the first file. The library finds all
        other files.
        Please read state information with the read_states flag 
        in the constructor or with the member function.
            

        Parameters
        ----------
        filepath : str
            path to the d3plot
        use_femzip : bool
            whether to use femzip for decompression
        read_states : str/list(str)
            read state information directly (saves time), 
            see the function read_states

        Returns
        -------
            D3plot d3plot : instance

        '''
        super(D3plot, self).__init__(*args, **kwargs)


    def get_parts(self):
        '''
        Notes
        -----
            Get the parts of the D3plot
        
        Returns
        -------
        parts : list(Part)
            parts within the D3plot

        Examples
        --------
            >>> list_of_all_parts = femfile.get_parts()
        '''

        # TODO rewrite as c functions
        part_ids = [_part.get_id() for _part in super(D3plot, self).get_parts() ]
        return [ Part(self, part_id) for part_id in part_ids ]


    def get_partByID(self, part_id):
        '''Get the part by its id
        
        Parameters
        ----------
        part_id : int or list(int)
            id or list of ids of parts

        Returns
        -------
        part_id : int or list(int)
            id of the part or list of ids
        
        Raises
        ------
        ValueError
            if `part_id` does not exist.

        Examples
        --------
            >>> part = femfile.get_partByID(1)
            >>> list_of_parts = femfile.get_partByID( [1,45,33] )
        '''

        # TODO rewrite as c functions
        if isinstance(part_id, (list,tuple,np.ndarray)):
            assert all( isinstance(entry,int) for entry in part_id)
            return [ Part(self, entry) for entry in part_id ]
        else:
            assert isinstance(part_id, int)
            return Part(self, part_id)
    

    @staticmethod
    def _compare_scatter(base_filepath, 
                         filepath_list, 
                         element_result,
                         pid_filter_list = None,
                         kMappingNeighbors = 4,
                         export_filepath = None,
                         **kwargs):
        '''Compare the scatter between mutliple d3plot
        ! UNFINISHED !

        Parameters
        ----------
        base_filepath : str
            filepath to the base D3plot, which will also be the base mesh
        filepath_list : list(str)
            list of filepaths of d3plot for comparison
        element_result : str or function(element)
            element results to compare. Either specify a user defined
            function or use predefined results. Available are
            disp, plastic_strain or energy.
        pid_filter_list : list(int)
            list of pids to filter for optionally
        kMappingNeighbors : int
            number of neighbors used for nearest neighbor mapping
        export_filepath : str
            optional filepath for saving. If none, the model
            is exported to a temporary file and shown in the
            browser.
        **kwargs : further arguments 
            additional arguments passed on to d3plot constructor

        Notes
        -----
            The file calling the function will be the basis for the comparison. 
            The other files results will be mapped onto this mesh. The scatter is 
            computed between all runs as maximum difference. 
        '''

        from sklearn.neighbors import KDTree

        if pid_filter_list == None:
            pid_filter_list = []

        # yay checks :)
        assert isinstance(base_filepath, str)
        assert os.path.isfile(base_filepath)
        assert isinstance(filepath_list, (list,tuple,np.ndarray))
        assert all( isinstance(entry, str) for entry in filepath_list )
        assert all( os.path.isfile(filepath) for filepath in filepath_list )
        assert isinstance(element_result, str) or callable(element_result)
        assert all( isinstance(entry, int) for entry in pid_filter_list )
        assert kMappingNeighbors > 0

        # prepare evaluation
        read_vars_str, eval_function =  _parse_element_result(element_result)
        if read_vars_str != None:
            kwargs['read_states'] = read_vars_str

        # base run element coords
        base_d3plot = D3plot(base_filepath, **kwargs)
        if not pid_filter_list:
            pid_filter_list = [ part.get_id() for part in base_d3plot.get_parts() ]
        base_mesh_coords, base_mesh_results = _extract_elem_coords( base_d3plot.get_partByID(pid_filter_list), element_result=eval_function )
        del base_d3plot
        del part

        # init vars for comparison
        element_result_max = base_mesh_results
        element_result_min = base_mesh_results

        # loop other files
        for _filepath in filepath_list:

            # new mesh with results
            _d3plot = D3plot(_filepath, **kwargs) 
            _d3plot_elem_coords, _d3plot_elem_results = _extract_elem_coords( _d3plot.get_partByID(pid_filter_list), element_result=eval_function, iTimestep=0)
            del _d3plot
            
            # compute mapping
            _tree = KDTree(_d3plot_elem_coords)
            distances, mapping_indexes = _tree.query(base_mesh_coords, 
                                                     return_distance=True,
                                                     sort_results=False,
                                                     k=kMappingNeighbors)
            distances = np.exp(distances) / np.sum( distances, axis=1)[:,None] # softmax weights

            # map results 
            _d3plot_elem_results = np.sum( distances*_d3plot_elem_results[mapping_indexes] ,axis=1)

            # update min and max
            element_result_max = np.maximum(_d3plot_elem_results, element_result_max)
            element_result_min = np.minimum(_d3plot_elem_results, element_result_min)

        # compute scatter
        element_result_max = element_result_max-element_result_min

        # plot scatter
        np.save("element_scatter.npy",element_result_max)
            

    def plot(self, iTimestep=0, element_result=None, fringe_bounds=[None,None], export_filepath=None):
        '''Plot the D3plot, currently shells only!

        Parameters
        ----------
        iTimestep : int
            timestep at which to plot the D3plot
        element_result : str or function
            which type of results to use as fringe
            None means no fringe is used.
            When using string as arg you may use plastic_strain or energy.
            Function shall take elem as input and return a float value (for fringe)
        fringe_bounds : list(float,float) or tuple(float,float)
            bounds for the fringe, default will use min and max value
        export_filepath : str
            optional filepath for saving. If none, the model
            is exported to a temporary file and shown in the
            browser.
        '''

        plot_parts(self.get_parts(), 
                   iTimestep=iTimestep, 
                   element_result=element_result, 
                   fringe_bounds=fringe_bounds, 
                   export_filepath=export_filepath)


    @staticmethod
    def plot_parts(parts, iTimestep=0, element_result=None, fringe_bounds=[None,None], export_filepath=None):
        '''Plot a selected group of parts
        
        Parameters:
        -----------
        parts : Part or list(Part)
            parts to plot. Must not be of the same file!
        iTimestep : int
            timestep at which to plot the D3plot
        element_result : str or function
            which type of results to use as fringe
            None means no fringe is used
            When using string as arg you may use plastic_strain or energy.
            Function shall take elem as input and return a float value (for fringe)
        fringe_bounds : list(float,float) or tuple(float,float)
            bounds for the fringe, default will use min and max value
        export_filepath : str
            optional filepath for saving. If none, the model
            is exported to a temporary file and shown in the
            browser.
        '''

        if not isinstance(parts, (tuple,list)):
            parts = [parts]

        assert all( isinstance(part,QD_Part) for part in parts ), "At least one list entry is not a part"

        plot_parts(parts, 
                   iTimestep=iTimestep, 
                   element_result=element_result, 
                   fringe_bounds=fringe_bounds, 
                   export_filepath=export_filepath)