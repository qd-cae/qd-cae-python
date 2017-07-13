

from .D3plot import plot_parts
from .dyna_cpp import QD_KeyFile, QD_Part


class KeyFile(QD_KeyFile):
    # copy class docs
    __doc__ = QD_KeyFile.__doc__

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

        if not isinstance(parts, (tuple, list)):
            parts = [parts]

        assert all(isinstance(part, QD_Part)
                   for part in parts), "At least one list entry is not a part"

        plot_parts(parts,
                   export_filepath=export_filepath)
