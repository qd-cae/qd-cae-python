

from ._dyna_utils import plot_parts, _parse_element_result, _extract_elem_coords
from .dyna_cpp import QD_D3plot, QD_Part

import os
import numpy as np


class D3plot(QD_D3plot):
    __doc__ = QD_D3plot.__doc__

    def compare_scatter(self,
                        filepath_list,
                        element_result,
                        pid_filter_list=None,
                        kMappingNeighbors=4,
                        export_filepath=None,
                        **kwargs):
        '''Compare the scatter between mutliple d3plot files

        Parameters
        ----------
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
            additional arguments passed on to d3plot constructor (e.g. read_states)

        Notes
        -----
            The file calling the function will be the basis for the comparison. 
            The other files results will be mapped onto this mesh. The scatter is 
            computed between all runs as maximum difference. 

        Examples
        --------
            >>> # Settings (don't forget to load the correct vars!)
            >>> state_vars = ["stress_mises max"]
            >>> other_files = ["path/to/d3plot_2", "path/to/d3plot_3"]
            >>>
            >>> # Element eval function (used for scatter computation)
            >>> def elem_eval_fun(elem):
            >>>     result = elem.get_stress_mises()
            >>>     if len(result):
            >>>         return result[-1]
            >>>     return 0.
            >>>
            >>> # load base file
            >>> d3plot = D3plot("path/to/d3plot", read_states=state_vars)
            >>> 
            >>> # compute and plot scatter
            >>> d3plot.compare_scatter(other_files, elem_eval_fun, read_states=state_vars)
        '''

        from sklearn.neighbors import KDTree

        if pid_filter_list == None:
            pid_filter_list = []

        # yay checks :)
        assert isinstance(filepath_list, (list, tuple, np.ndarray))
        assert all(isinstance(entry, str) for entry in filepath_list)
        assert all(os.path.isfile(filepath) for filepath in filepath_list)
        assert isinstance(element_result, str) or callable(element_result)
        assert all(isinstance(entry, int) for entry in pid_filter_list)
        assert kMappingNeighbors > 0

        # prepare evaluation
        read_vars_str, eval_function = _parse_element_result(element_result)
        if read_vars_str != None:
            kwargs['read_states'] = read_vars_str

        # base run element coords
        if not pid_filter_list:
            pid_filter_list = [part.get_id() for part in self.get_parts()]
        base_mesh_coords, base_mesh_results = _extract_elem_coords(
            self.get_partByID(pid_filter_list),
            element_result=eval_function,
            element_type="shell")

        # init vars for comparison
        element_result_max = base_mesh_results
        element_result_min = base_mesh_results
        del base_mesh_results

        # loop other files
        for _filepath in filepath_list:

            # new mesh with results
            _d3plot = D3plot(_filepath, **kwargs)
            _d3plot_elem_coords, _d3plot_elem_results = _extract_elem_coords(
                _d3plot.get_partByID(pid_filter_list),
                element_result=eval_function,
                iTimestep=0,
                element_type="shell")
            del _d3plot  # deallocate c++ stuff

            # compute mapping
            _tree = KDTree(_d3plot_elem_coords)
            distances, mapping_indexes = _tree.query(base_mesh_coords,
                                                     return_distance=True,
                                                     sort_results=False,
                                                     k=kMappingNeighbors)
            distances = np.exp(distances)
            distances = distances / \
                np.sum(distances, axis=1)[:, None]  # softmax weights

            # map results
            _d3plot_elem_results = np.sum(
                distances * _d3plot_elem_results[mapping_indexes], axis=1)

            # update min and max
            element_result_max = np.max(
                _d3plot_elem_results, element_result_max)
            element_result_min = np.min(
                _d3plot_elem_results, element_result_min)

        # compute scatter
        element_result_max = element_result_max - element_result_min
        del element_result_min

        # plot scatter, sometimes I like it dirty
        data = [0]  # does not work otherwise ...

        def eval_scatter(elem):
            data[0] = data[0] + 1
            return element_result_max[data[0] - 1]

        self.plot(element_result=eval_scatter, export_filepath=export_filepath)

    def plot(self, iTimestep=0, element_result=None, fringe_bounds=[None, None], export_filepath=None):
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

        Examples
        --------
            Load a d3plot and plot it's geometry

            >>> d3plot = D3plot("path/to/d3plot")
            >>> d3plot.plot() # just geometry

            Read the state data and plot in deformed state

            >>> # read state data
            >>> d3plot.read_states(["disp","plastic_strain max"])
            >>> d3plot.plot(iTimestep=-1) # last state

            Use a user-defined element evaluation function for fringe colors.

            >>> # User defined evaluation function
            >>> def eval_fun(element):
            >>>     res = element.get_plastic_strain()
            >>>     if len(res): # some elements may miss plastic strain
            >>>         return res[-1] # last timestep
            >>> 
            >>> d3plot.plot(iTimestep=-1, element_result=eval_fun, fringe_bounds=[0, 0.05])
        '''

        plot_parts(self.get_parts(),
                   iTimestep=iTimestep,
                   element_result=element_result,
                   fringe_bounds=fringe_bounds,
                   export_filepath=export_filepath)

    @staticmethod
    def plot_parts(parts, iTimestep=0, element_result=None, fringe_bounds=[None, None], export_filepath=None):
        '''Plot a selected group of parts. 

        Parameters
        ----------
        parts : Part or list(Part)
            parts to plot. May be from different files.
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

        Notes
        -----
            Can be applied to parts, coming from different files.

        Examples
        --------
            For a full description of plotting functionality, see `d3plot.plot`.
            Load a d3plot and plot a part from it:

            >>> d3plot_1 = D3plot("path/to/d3plot_1")
            >>> part_1 = d3plot_1.get_partByID(1)
            >>> D3plot.plot_parts( [part_1] ) # static function!

            Read a second d3plot and plot both parts at once

            >>> d3plot_2 = D3plot("path/to/d3plot_2") # different file!
            >>> part_2 = d3plot_2.get_partByID(14)
            >>> D3plot.plot_parts( [part_1, part_2] )

        '''

        if not isinstance(parts, (tuple, list)):
            parts = [parts]

        assert all(isinstance(part, QD_Part)
                   for part in parts), "At least one list entry is not a part"

        plot_parts(parts,
                   iTimestep=iTimestep,
                   element_result=element_result,
                   fringe_bounds=fringe_bounds,
                   export_filepath=export_filepath)
