
import numpy as np
from .RawD3plot import RawD3plot


class ArrayD3plot(RawD3plot):
    ''' Makes the data in a D3plot directly accessible via arrays

    This class is a usability wrapper for the RawD3plot, which gives access
    to the raw data arrays in a d3plot file.
    '''

    def keys(self):
        '''Get the variable key names contained in this file

        Returns
        -------
        variables : list of str
            names of the variables in this file

        Notes
        -----
            The given variable names are not the pure data arrays from memory.
            If one needs access to the pure, raw data, then use the RawD3plot
            class and despair.

        Examples
        --------
            >>> raw_d3plot.get_keys()
            ['elem_shell_results', 'elem_shell_results_layers',
                'elem_solid_results', ... ]
        '''

        names = self.get_raw_keys()

        # elem data wrapper
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

        # elem deletion wrapper
        if "elem_beam_deletion_info" in names:
            del names[names.index("elem_beam_deletion_info")]
            names.append("elem_beam_is_alive")
        if "elem_shell_deletion_info" in names:
            del names[names.index("elem_shell_deletion_info")]
            names.append("elem_shell_is_alive")
        if "elem_tshell_deletion_info" in names:
            del names[names.index("elem_tshell_deletion_info")]
            names.append("elem_tshell_is_alive")
        if "elem_solid_deletion_info" in names:
            del names[names.index("elem_solid_deletion_info")]
            names.append("elem_solid_is_alive")

        # option to get elem results filled in with rigid shells
        if "elem_shell_results" in names:
            names.append("elem_shell_results_filled")

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
            arugment is not a string.

        Examples
        --------
            >>> shell_geom_data = raw_d3plot["elem_shell_data"]
        '''

        if not isinstance(key, str):
            raise ValueError("The argument is not a string.")

        raw_keys = self.get_raw_keys()

        # element nodes
        if key == "elem_beam_nodes":
            return self.get_raw_data("elem_beam_data")[:, :4]
        elif key == "elem_shell_nodes":
            return self.get_raw_data("elem_shell_data")[:, :4]
        elif key == "elem_tshell_nodes":
            return self.get_raw_data("elem_tshell_data")[:, :8]
        elif key == "elem_solid_nodes":
            return self.get_raw_data("elem_solid_data")[:, :8]

        # element material ids
        elif key == "elem_beam_material_ids":
            return self.get_raw_data("elem_beam_data")[:, -1]
        elif key == "elem_shell_material_ids":
            return self.get_raw_data("elem_shell_data")[:, -1]
        elif key == "elem_tshell_material_ids":
            return self.get_raw_data("elem_tshell_data")[:, -1]
        elif key == "elem_solid_material_ids":
            return self.get_raw_data("elem_solid_data")[:, -1]

        # element deletion
        if key == "elem_beam_is_alive":
            return np.array(self.get_raw_data("elem_beam_deletion_info"), dtype=bool)
        if key == "elem_shell_is_alive":
            return np.array(self.get_raw_data("elem_shell_deletion_info"), dtype=bool)
        if key == "elem_tshell_is_alive":
            return np.array(self.get_raw_data("elem_tshell_deletion_info"), dtype=bool)
        if key == "elem_solid_is_alive":
            return np.array(self.get_raw_data("elem_solid_deletion_info"), dtype=bool)

        # rigid shell treatment (fill zeros in)
        if (key == "elem_shell_results_filled"
            or key == "elem_shell_results_layers_filled") \
           and ("material_type_numbers" in raw_keys) \
           and (20 in self["material_type_numbers"]):

            # find rigid shells with mat20
            rigid_shells = self._get_rigid_shells()

            # allocate new array
            if key == "elem_shell_results_filled":
                shell_results = self["elem_shell_results"]
                shape = [shell_results.shape[0],
                         shell_ids.shape[0],
                         shell_results.shape[2]]
                shell_results_filled = np.empty(shape, dtype=np.float32)
                # shell_results_filled[:, :, :] = np.nan # debug

                # fill it
                shell_results_filled[:, ~rigid_shells, :] = shell_results
                del shell_results
                shell_results_filled[:, rigid_shells, :] = 0.

                return shell_results

            elif key == "elem_shell_results_layers_filled":
                shell_results = self["elem_shell_results_layers"]
                shape = [shell_results.shape[0],
                         shell_ids.shape[0],
                         shell_results.shape[2],
                         shell_results.shape[3]]
                shell_results_filled = np.empty(shape, dtype=np.float32)
                # shell_results_filled[:, :, :] = np.nan # debug

                # fill it
                shell_results_filled[:, ~rigid_shells, :, :] = shell_results
                del shell_results
                shell_results_filled[:, rigid_shells, :, :] = 0.

                return shell_results

        else:
            key == "elem_shell_results"

        # search variable in maps
        return self.get_raw_data(key)

    def _get_rigid_shells(self):
        ''' Find out which shell elements are rigid

        Returns
        ----------
        rigid_shells : np.ndarray
            boolean vector

        '''

        raw_keys = self.get_raw_keys()

        # check requirements
        if "material_type_numbers" not in raw_keys \
           or "elem_shell_data" not in raw_keys:
            return None

        shell_ids = self["elem_shell_ids"]
        # mat ids start at 1 not 0, thus -1
        shell_material_ids = self["elem_shell_material_ids"] - 1
        material_types = self["material_type_numbers"]

        # find rigid shells with mat20
        return material_types[shell_material_ids] == 20
