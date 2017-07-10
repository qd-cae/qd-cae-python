
#from ._dyna_utils import plot_parts
from .dyna_cpp import QD_Part
from .D3plot import plot_parts


class Part(QD_Part):
    __doc__ = QD_Part.__doc__

    def __init__(self, *args, **kwargs):
        super(Part, self).__init__(*args, **kwargs)
    __init__.__doc__ = QD_Part.__init__.__doc__

    def plot(self, iTimestep=0, element_result=None, fringe_bounds=[None, None], export_filepath=None):
        '''Plot the Part, currently shells only!

        Parameters
        ----------
        iTimestep : int
            timestep at which to plot the D3plot
        element_result : str or function
            which type of results to use as fringe
            None means no fringe is used
            Function shall take elem as input and return a float value (for fringe)
        fringe_bounds : list(float,float) or tuple(float,float)
            bounds for the fringe, default will use min and max value
        export_filepath : str
            optional filepath for saving. If none, the model
            is exported to a temporary file and shown in the
            browser.

        Examples
        --------

            Load a d3plot and plot a part

            >>> d3plot = D3plot("path/to/d3plot")
            >>> part = d3plot.get_partByID(1)
            >>> part.plot() # just geometry

            Read the state data and plot in deformed state

            >>> # read state data
            >>> d3plot.read_states(["disp","stress_mises max"])
            >>> part.plot(iTimestep=-1) # last state

            Use a user-defined element evaluation function for fringe colors.

            >>> # User defined evaluation function
            >>> def eval_fun(element):
            >>>     res = element.get_stress_mises()
            >>>     if len(res): # some elements may miss stresses
            >>>         return res[-1] # last timestep
            >>> part.plot(iTimestep=-1, element_result=eval_fun)
        '''

        plot_parts(self,
                   iTimestep=iTimestep,
                   element_result=element_result,
                   fringe_bounds=fringe_bounds,
                   export_filepath=export_filepath)
