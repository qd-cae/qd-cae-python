

from ._dyna_utils import plot_parts
from .dyna_cpp import QD_D3plot, QD_Part
from .Part import Part

class D3plot(QD_D3plot):
    '''
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
            

        Parameters:
        -----------
        filepath : str
            path to the d3plot
        use_femzip : bool
            whether to use femzip for decompression
        read_states : str/list(str)
            read state information directly (saves time), 
            see the function read_states

        Returns:
        --------
            D3plot d3plot : instance

        '''
        super(D3plot, self).__init__(*args, **kwargs)

    

    def plot(self, iTimestep=0, element_result=None, fringe_bounds=[None,None], export_filepath=None):
        '''Plot the D3plot, currently shells only!

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


    def get_parts(self):
        '''Get parts of the D3plot
        
        Returns:
        --------
        parts : list(Part)
            parts within the D3plot

        Overwritten function.
        '''

        part_ids = [_part.get_id() for _part in super(D3plot, self).get_parts() ]
        return [ Part(self, part_id) for part_id in part_ids ]


    def get_partByID(self, *args, **kwargs):
        '''Get the part by its id
        
        Returns:
        --------
        part_id : int
            id of the part
        '''

        part_id = super(D3plot, self).get_partByID(*args,**kwargs).get_id()
        return Part(self, part_id)