
#from ._dyna_utils import plot_parts
from .dyna_cpp import QD_Part
from .D3plot import plot_parts

class Part(QD_Part):
    '''Part of a D3plot. 

    Notes
    -----
        The part specific mesh data may be accessed by this container.
        It is recommended to get parts by:
        >>> part = d3plot.get_parts()
        >>> part_id = 13
        >>> part = d3plot.get_partByID(part_id)

        In case one needs it, the constructor has the signature:
        __init__(femfile, part_id)
    '''

    def __init__(self, *args, **kwargs):
        '''Constructor

        Parameters
        ----------
        femfile : FEMFile
            femfile of which to get the part from
        part_id : int
            id of the part

        Examples
        --------
            >>> femfile = KeyFile('path/to/keyfile') # or D3plot
            >>> part = Part(femfile, part_id=1)
        '''
        super(Part, self).__init__(*args, **kwargs)


    def plot(self, iTimestep=0, element_result=None, fringe_bounds=[None,None], export_filepath=None):
        '''Plot the Part, currently shells only!

        Parameters:
        -----------
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
        '''

        plot_parts(self, 
                   iTimestep=iTimestep, 
                   element_result=element_result, 
                   fringe_bounds=fringe_bounds, 
                   export_filepath=export_filepath)
