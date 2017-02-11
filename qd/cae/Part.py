
#from ._dyna_utils import plot_parts
from .dyna_cpp import QD_Part
from .D3plot import plot_parts

class Part(QD_Part):
    '''Part of a D3plot

    It is recommended to get parts by:
    d3plot.get_parts()
    d3plot.get_partByID(...)
    '''

    def __init__(self, *args, **kwargs):
        '''Constructor

        Parameters:
        -----------
        D3plot : d3plot
            d3plot of which to get the part from
        partID : int
            id of the part in the D3plot
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
