
import os
import tempfile
import webbrowser
from ._dyna_utils import _parts_to_html
from .dyna_cpp import QD_D3plot

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

        _html = _parts_to_html(self.get_parts(), 
                               iTimestep=iTimestep, 
                               element_result=element_result,
                               fringe_bounds=fringe_bounds)
        
        # save if export path present
        if export_filepath:
            with open(export_filepath,"w") as fp:
                fp.write(_html)

        # plot if no export
        else:

            # clean temporary dir first (keeps mem low)
            tempdir = tempfile.gettempdir()
            tempdir = os.path.join(tempdir,"qd_eng")
            if not os.path.isdir(tempdir):
                os.mkdir(tempdir)

            for tmpfile in os.listdir(tempdir):
                tmpfile = os.path.join(tempdir,tmpfile)
                if os.path.isfile(tmpfile):
                    os.remove(tmpfile)
            
            # create new temp file
            with tempfile.NamedTemporaryFile(dir=tempdir,suffix=".html", mode="w", delete=False) as fp:
                fp.write(_html)
                webbrowser.open(fp.name)


        