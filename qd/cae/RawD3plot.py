
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import h5py
warnings.resetwarnings()
import numpy as np
from qd.cae.dyna import QD_RawD3plot


class RawD3plot(QD_RawD3plot):
    __doc__ = QD_RawD3plot.__doc__

    def __init__(self, filepath):
        ''' Create a RawD3plot file object

        Parameters
        ----------
        filepath : str
            path to either the (first) d3plot or a d3plot in hdf5 format

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
            The constructor first checks, if the file is a hdf5 file and
            then tries to open it as a casual D3plot.
            If LS-Dyna writes multiple files (one for each timestep),
            give the filepath to the first file. The library finds all
            other files. The class automatically reads all data!

        Examples
        --------
            >>> from qd.cae.dyna import RawD3plot
            >>> # read an arbitrary d3plot
            >>> raw_d3plot = RawD3plot("path/to/d3plot")
            >>> #read femzip compressed file
            >>> raw_d3plot = RawD3plot("path/to/d3plot.fz")
            >>> # save file as HDF5
            >>> raw_d3plot.save_hdf5("path/to/d3plot.h5")
            >>> # open HDF5 d3plot
            >>> raw_d3plot = RawD3plot("path/to/d3plot.h5")

        '''
        if self._is_hdf5(filepath):
            super(RawD3plot, self).__init__()
            self._load_hdf5(filepath)
        else:
            super(RawD3plot, self).__init__(filepath)

    def get_raw_keys(self):
        ''' Get the names of the raw data fields

        Returns
        -------
        names : list of str

        Exmaple
        -------
            >>> raw_d3plot.get_raw_keys()
            ['elem_shell_data', 'elem_beam_data', ...]
        '''
        return self._get_int_names() + self._get_float_names() + \
            self._get_string_names()

    def get_raw_data(self, key):
        '''Get a variable from its name

        Parameters
        ----------
        key : str
            name of the array or variable to get

        Returns
        -------
        data : np.ndarray or list
            data array or list

        Raises
        ------
        ValueError
            In case that the key can not be found or the 
            arugment is not a str.

        Examples
        --------
            >>> shell_geom_data = raw_d3plot.get_raw_data("elem_shell_data")
        '''

        if not isinstance(key, str):
            raise ValueError("The argument is not a string.")

        # search variable in maps
        elif key in self._get_int_names():
            return self._get_int_data(key)
        elif key in self._get_float_names():
            return self._get_float_data(key)
        elif key in self._get_string_names():
            return self._get_string_data(key)
        else:
            raise ValueError("Can not find key:" + str(key))

    def save_hdf5(self, filepath, overwrite=True, compression="gzip"):
        '''Save the raw d3plot to HDF5

        Paramters
        ---------
        filepath : str
            path for the hdf5 file
        overwrite : bool
            whether to overwrite an existing file
        compression : str
            compression technique (see h5py docs)

        Notes
        -----
            Saves the data arrays from a D3plot into an HDF5 file.

        Raises
        ------
        IOError
            In case anything related to IO goes wrong.

        Examples
        --------
            >>> raw_d3plot = D3plot("path/to/d3plot")
            >>> raw_d3plot.save_hdf5("path/to/d3plot.h5)
        '''

        if os.path.isfile(filepath) and overwrite:
            os.remove(filepath)

        # open file
        with h5py.File(filepath, "w") as fh:

            # int data
            int_grp = fh.create_group("int_data")
            for name in self._get_int_names():
                data = self.get_raw_data(name)
                dset = int_grp.create_dataset(
                    name, data.shape, dtype='i', data=data, compression=compression)

            # float data
            float_grp = fh.create_group("float_data")
            for name in self._get_float_names():
                data = self.get_raw_data(name)
                dset = float_grp.create_dataset(
                    name, data.shape, dtype='f', data=data, compression=compression)
                #dset[...] = data

            # string data
            str_grp = fh.create_group("string_data")
            for name in self._get_string_names():
                data = self.get_raw_data(name)

                # convert utf8 to ascii
                data = [entry.encode('ascii', 'ignore') for entry in data]
                max_len = np.max([len(entry) for entry in data])

                dset = str_grp.create_dataset(
                    name, (len(data), 1), 'S' + str(max_len), data=data, compression=compression)

    def _is_hdf5(self, filepath):
        ''' Check if a file is a HDF5 file

        Parameters
        ----------
        filepath : str
            path to the file

        Returns
        -------
        is_hdf5 : bool
        '''

        try:
            with h5py.File(filepath, "r") as fh:
                pass
            return True
        except:
            return False

    def _load_hdf5(self, filepath):
        ''' Load a d3plot, which was saved as an HDF5 file

        Parameters
        ----------
        filepath : str
            path to the d3plot in HDF5 format
        '''

        dataset_counter = 0
        with h5py.File(filepath, "r") as fh:

            if "int_data" in fh:
                dataset_counter += 1
                for name, data in fh["int_data"].items():
                    data = np.array(data[:], copy=True)
                    self._set_int_data(name, data)

            if "float_data" in fh:
                dataset_counter += 1
                for name, data in fh["float_data"].items():
                    self._set_float_data(name, data[:])

            if "string_data" in fh:
                dataset_counter += 1
                for name, data in fh["string_data"].items():
                    data = [str(entry, "utf-8")
                            for entry in data[:].flatten()]
                    self._set_string_data(name, data)

        if dataset_counter == 0:
            raise RuntimeError(
                "The groups int_data, float_data and string_data were not found in the file and thus no data was loaded.")
