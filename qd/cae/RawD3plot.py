
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

        names = self._get_int_names() + self._get_float_names() + self._get_string_names()

        if "elem_beam_data" in names:
            del names[names.index("elem_beam_data")]
            names.append("elem_beam_nodes")
            names.append("elem_beam_material_ids")
        if "elem_shell_data" in names:
            del names[names.index("elem_shell_data")]
            names.append("elem_shell_nodes")
            names.append("elem_shell_material_ids")
        if "elem_tshell_data" in names:
            del names[names.index("elem_tshell_data")]
            names.append("elem_tshell_nodes")
            names.append("elem_tshell_material_ids")
        if "elem_solid_data" in names:
            del names[names.index("elem_solid_data")]
            names.append("elem_solids_nodes")
            names.append("elem_solids_material_ids")

        return names

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

        # element nodes
        if key == "elem_beam_nodes":
            return self._get_int_data("elem_beam_data")[:,:4]
        elif key == "elem_shell_nodes":
            return self._get_int_data("elem_shell_data")[:,:4]
        elif key == "elem_tshell_nodes":
            return self._get_int_data("elem_tshell_data")[:,:8]
        elif key == "elem_solid_nodes":
            return self._get_int_data("elem_solid_data")[:,:8]
        
        # element material ids
        elif key == "elem_beam_material_ids":
            return self._get_int_data("elem_beam_data")[:,-1]
        elif key == "elem_shell_material_ids":
            return self._get_int_data("elem_shell_data")[:,-1]
        elif key == "elem_tshell_material_ids":
            return self._get_int_data("elem_tshell_data")[:,-1]
        elif key == "elem_solid_material_ids":
            return self._get_int_data("elem_solid_data")[:,-1]
        
        # search variable in maps
        elif key in self._get_int_names():
            return self._get_int_data(key)
        elif key in self._get_float_names():
            return self._get_float_data(key)
        elif key in self._get_string_names():
            return self._get_string_data(key)
        else:
            raise ValueError("Can not find key:" + str(key))
