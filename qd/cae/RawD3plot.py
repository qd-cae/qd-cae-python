
from qd.cae.dyna import QD_RawD3plot


class RawD3plot(QD_RawD3plot):
    __doc__ = QD_RawD3plot.__doc__

    def keys(self):
        '''Get the variable key names contained in this file

        Returns
        -------
        variables : list of str
            names of the variables in this file

        Examples
        --------
            >>> raw_d3plot.get_keys()
            ['elem_shell_results', 'elem_shell_results_layers', 'elem_solid_results', ... ]
        '''

        return self.get_int_names() + self.get_float_names() + self.get_string_names()

    def __getitem__(self, key):
        '''Get a variable from its name

        Parameters
        ----------
        key : str
            name of the variable to get

        Returns
        -------
        data : np.ndarray or list
            data array or list

        Raises
        ------
        ValueError
            In case that the key can not be found or the 
            arugment is not a string

        Examples
        --------
            >>> shell_geom_data = raw_d3plot["elem_shell_data"]
        '''

        if not isinstance(key, str):
            raise ValueError("The argument is not a string.")

        if key in self.get_int_names():
            return self.get_int_data(key)
        elif key in self.get_float_names():
            return self.get_float_data(key)
        elif key in self.get_string_names():
            return self.get_string_data(key)
        else:
            raise ValueError("Can not find key:" + str(key))
