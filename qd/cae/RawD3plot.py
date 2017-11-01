
import os
import h5py
import numpy as np
from qd.cae.dyna import QD_RawD3plot


class RawD3plot(QD_RawD3plot):
    __doc__ = QD_RawD3plot.__doc__

    def raw_keys():
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
            >>> shell_geom_data = raw_d3plot["elem_shell_data"]
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

    def save_hdf5(self, filepath, overwrite=True):
        '''Save the raw d3plot to HDF5

        Paramters
        ---------
        filepath : str
            path for the hdf5 file
        overwrite : bool
            whether to overwrite an existing file

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
        fh = h5py.File(filepath, "w")

        # int data
        int_grp = fh.create_group("int_data")
        for name in self._get_int_names():
            data = self.get_raw_data(name)
            dset = int_grp.create_dataset(
                name, data.shape, dtype='i', data=data)

        # float data
        float_grp = fh.create_group("float_data")
        for name in self._get_float_names():
            data = self.get_raw_data(name)
            dset = float_grp.create_dataset(
                name, data.shape, dtype='f', data=data)
            #dset[...] = data

        # string data
        str_grp = fh.create_group("string_data")
        for name in self._get_string_names():
            data = self.get_raw_data(name)

            # convert utf8 to ascii
            data = [entry.encode('ascii', 'ignore') for entry in data]
            max_len = np.max([len(entry) for entry in data])

            dset = str_grp.create_dataset(
                name, (len(data), 1), 'S' + str(max_len), data=data)

        fh.flush()
        fh.close()
