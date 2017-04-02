

from .D3plot import plot_parts
from .dyna_cpp import QD_KeyFile, QD_Part
from .Part import Part

class KeyFile(QD_KeyFile):
    '''  Read a .key file (solver input file)

    Parameters
    ----------
    filepath : str
        filepath to the keyfile
    
    Notes
    -----
        Currently only the mesh is used from the file!

    Examples
    --------
        >>> kf = KeyFile("path/to/keyfile")
    '''

    def __init__(self, *args, **kwargs):
        ''' Read a .key file (solver input file)

        Parameters
        ----------
        filepath : str
            filepath to the keyfile
        
        Notes
        -----
            Currently only the mesh is used from the file!

        Examples
        --------
            >>> kf = KeyFile("path/to/keyfile")
        '''
        super(KeyFile, self).__init__(*args, **kwargs)
    

    def plot(self, export_filepath=None):
        '''Plot the KeyFile geometry.

        Parameters:
        -----------
        export_filepath : str
            optional filepath for saving. If none, the model
            is exported to a temporary file and shown in the
            browser.
        '''

        plot_parts(self.get_parts(),  
                   export_filepath=export_filepath)


    @staticmethod
    def plot_parts(parts, export_filepath=None):
        '''Plot a selected group of parts
        
        Parameters:
        -----------
        parts : Part or list(Part)
            parts to plot. Must not be of the same file!
        export_filepath : str
            optional filepath for saving. If none, the model
            is exported to a temporary file and shown in the
            browser.
        '''

        if not isinstance(parts, (tuple,list)):
            parts = [parts]

        assert all( isinstance(part,QD_Part) for part in parts ), "At least one list entry is not a part"

        plot_parts(parts, 
                   export_filepath=export_filepath)


    def get_parts(self):
        '''Get parts of the KeyFile
        
        Returns:
        --------
        parts : list(Part)
            parts of the file

        Overwritten function.
        '''

        part_ids = [_part.get_id() for _part in super(KeyFile, self).get_parts() ]
        return [ Part(self, part_id) for part_id in part_ids ]


    def get_partByID(self, *args, **kwargs):
        '''Get the part by its id
        
        Returns:
        --------
        part_id : int
            id of the part
        '''

        part_id = super(KeyFile, self).get_partByID(*args,**kwargs).get_id()
        return Part(self, part_id)