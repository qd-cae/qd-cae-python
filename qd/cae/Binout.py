
import os
import re
import sys
import glob
import struct
import ntpath
import numpy as np

import sys
# if sys.version_info[0] < 3:
#    from .lsda_py2 import Lsda
# else:
#    from .lsda_py3 import Lsda

from .lsda_py3 import Lsda


'''
## Recoded stuff from lsda from LSTC, but much more readable and quoted ...
#
class Diskfile:

    # symbol = binary variable / python translation
    # note: bolded means unsigned
    #
    # b = char / int
    # h = short / int
    # i = int   / int
    # q = long long / int
    # f = float  / float
    # d = double / float
    # s = char[] / string

    packsize = [0,"b","h",0,"i",0,0,0,"q"]
    packtype = [0,"b","h","i","q","B","H","I","Q","f","d","s"]
    sizeof = [0,1,2,4,8,1,2,4,8,4,8,1] # packtype

    ##
    #
    #
    def __init__(self,filepath,mode="r"):

        # This opens a file and mainly treats the header
        #
        # header[0] == header length & header offset ?!? | default=8 byte
        # header[1] == lengthsize  | default = 8 byte
        # header[2] == offsetsize  | default = 8 byte
        # header[3] == commandsize | default = 1 byte
        # header[4] == typesize    | default = 1 byte
        # header[5] == file endian | default = 1 byte
        # header[6] == ?           | default = 0 byte
        # header[7] == ?           | default = 0 byte

        # start init
        self.filepath = filepath
        self.mode = mode
        self.file_ends = False

        # open file ...
        self.fp = open(filepath,mode+"b")
        # ... in read mode yay
        if mode == "r":
            header = struct.unpack("BBBBBBBB",self.fp.read(8))
            if header[0] > 8: #?!? some kind of header offset ?!?
                self.fp.seek(header[0])
        # ... in write mode
        else:
            header = [8,8,8,1,1,0,0,0]
            if sys.byteorder == "big":
                header[5] = 0
            else:
                header[5] = 1

        # fetch byte length of several ... I honestly don't know what exactly
        self.lengthsize    = header[1]
        self.offsetsize    = header[2]
        self.commandsize   = header[3]
        self.typesize      = header[4]
        self.ordercode = ">" if header[5] == 0 else '<' # endian

        # again I have no idea what is going on ...
        # these are some data unpacking format codes
        self.ounpack  =  self.ordercode+Diskfile.packsize[self.offsetsize]
        self.lunpack  =  self.ordercode+Diskfile.packsize[self.lengthsize]
        self.lcunpack = (self.ordercode+
                        Diskfile.packsize[self.lengthsize]+
                        Diskfile.packsize[self.commandsize])
        self.tolunpack = (self.ordercode+
                        Diskfile.packsize[self.typesize]+
                        Diskfile.packsize[self.offsetsize]+
                        Diskfile.packsize[self.lengthsize])
        self.comp1 = self.typesize+self.offsetsize+self.lengthsize
        self.comp2 = self.lengthsize+self.commandsize+self.typesize+1

        # write header if write mode
        if mode == "w":
            header_str = ''
            for value in header:
                s += struct.pack("B",value) # convert to unsigned char
            self.fp.write(s)
            #self.writecommand(17,Lsda.SYMBOLTABLEOFFSET)
            #self.writeoffset(17,0)
            #self.lastoffset = 17

    # UNFINISHED
'''


class Binout:
    '''This class is meant to read binouts from LS-Dyna

    Parameters
    ----------
    filepath : str
        path to the binout

    Notes
    -----
        This class is only a utility wrapper for Lsda from LSTC.

    Examples
    --------
        >>> binout = Binout("path/to/binout")
    '''

    def __init__(self, filepath):
        '''Constructor for a binout

        Parameters
        ----------
        filepath : str
            path to the binout or pattern

        Notes
        -----
            The class loads the file given in the filepath. By giving a 
            search pattern such as: "binout*", all files with that 
            pattern will be loaded.

        Examples
        --------
            >>> # reads a single binout
            >>> binout = Binout("path/to/binout0000")
            >>> binout.filelist
            ['path/to/binout0000']

            >>> # reads multiple files
            >>> binout = Binout("path/to/binout*")
            >>> binout.filelist
            ['path/to/binout0000','path/to/binout0001']
        '''

        self.filelist = glob.glob(filepath)

        # check file existance
        if not self.filelist:
            raise IOError("No file was found.")

        # open lsda buffer
        self.lsda = Lsda(self.filelist, "r")
        self.lsda_root = self.lsda.root

        # if sys.version_info[0] < 3:
        #    self.lsda_root = self.lsda.root
        # else:
        #    self.lsda_root = self.lsda.root.children[""]
        #self.lsda_root = self.lsda.root

    ##
    #
    # @param str/list(str) *path : path to read within binout. Leave empty fir top level.
    # @return see description
    #
    # @example read()/read("swforc")/read("swforc","time")/read("swforc/time")
    #
    # Since this command reads everything, the following return types are possible:
    #   - path/to/dir : list(str) names of directory content
    #   - path/to/ids : np.array(int) ids
    #   - path/to/variable : np.array(float) vars
    #
    def read(self, *path):
        '''read(path)
        Read all data from Binout (top to low level)

        Parameters
        ----------
        path : list(str) or str
            internal path in the folder structure of the binout

        Returns
        -------
        ret : list(str) or np.ndarray
            list of subdata within the folder or data itself

        Notes
        -----
            This function is used to read any data from the binout. It has been used
            to make the access to the data more comfortable. The return type depends 
            on the given path:

             - `binout.read()` : list(str) names of directories (in binout)
             - `binout.read(dir)` : list(str) names of variables or subdirs
             - `binout.read(dir1, ..., variable)` : np.array(float/int) data

            If you have multiple outputs with different ids (e.g. in nodout for 
            multiple nodes) then don't forget to read the ids array for 
            identification or id-labels.

        Examples
        --------
            >>> from qd.cae.dyna import Binout
            >>> binout = Binout("test/binout")
            >>> # get top dirs
            >>> binout.read()
            ['swforc']
            >>> binout.read("swforc")
            ['title', 'failure', 'ids', 'failure_time', ...]
            >>> binout.read("swforc","shear").shape
            (321L, 26L)
            >>> binout.read("swforc","ids").shape
            (26L,)
            >>> binout.read("swforc","ids")
            array([52890, 52891, 52892, ...])
            >>> # strings are just numbers, see Binout.to_string
            >>> binout.read("swforc","typenames")
            array([99, 111, 110, ...]) 
        '''

        return self._decode_path(path)

        '''
        iLevel = len(path)

        if iLevel == 0:
            return self._bstr_to_str(list(self.lsda_root.children.keys()))
        elif ((path[0] == "jntforc")
              or (path[0] == "rwforc")
              or (path[0] == "elout")):
            return self._decode_three_levels(path)
        else:
            return self._decode_two_levels(path)
        '''

    def _decode_path(self, path):
        '''Decode a path and get whatever is inside.

        Parameters
        ----------
        path : list(str)
            path within the binout

        Notes
        -----
            Usually returns the folder children. If there are variables in the folder
            (usually also if a subfolder metadata exists), then the variables will
            be printed from these directories.

        Returns
        -------
        ret : list(str) or np.ndarray
            either subfolder list or data array
        '''

        iLevel = len(path)

        if iLevel == 0:  # root subfolders
            return self._bstr_to_str(list(self.lsda_root.children.keys()))

        # some subdir
        else:

            # try if path can be resolved (then it's a dir)
            # in this case print the subfolders or subvars
            try:

                dir_symbol = self._get_symbol(self.lsda_root, path)

                if 'metadata' in dir_symbol.children:
                    return self._collect_variables(dir_symbol)
                else:
                    return self._bstr_to_str(list(dir_symbol.children.keys()))

            # an error is risen, if the path is not resolvable
            # this could be, because we want to read a var
            except ValueError as err:

                return self._get_variable(path)

    def _decode_two_levels(self, path):
        '''Decode a path, which has depth 2 (default)

        Parameters
        ----------
        path : list(str)

        Returns
        -------
        ret : list(str)/np.array(int)/np.array(float)
            either path children or data
        '''

        iLevel = len(path)

        # LEVEL 0 : no args (top)
        if iLevel == 0:
            return self._bstr_to_str(list(self.lsda_root.children.keys()))

        # LEVEL 1 : variable names
        elif iLevel == 1:

            # subdir
            dir_symbol = self._get_symbol(self.lsda_root, path)

            # collect variable names
            return self._collect_variables(dir_symbol)

        # LEVEL 2 : read variable data
        elif iLevel == 2:
            return self._get_variable(path)

        # LEVEL 3+ : Error
        else:
            raise ValueError("Invalid path depth of %d > 2" % iLevel)

    def _decode_three_levels(self, path):
        '''Decode a path, which has depth 3 (default)

        Parameters
        ----------
        path : list(str)

        Returns
        -------
        ret : list(str)/np.array(int)/np.array(float)
            either path children or data

        Notes
        -----
            Level 3 Files are:
             - rwforc
             - jntforc
        '''

        iLevel = len(path)

        # LEVEL 0 : no args (top)
        if iLevel == 0:
            return self._bstr_to_str(list(self.lsda_root.children.keys()))

        # LEVEL 1 : categories
        elif iLevel == 1:

            # subdir
            dir_symbol = self._get_symbol(self.lsda_root, path)

            # search subsubdir vars (metadata + states)
            return self._bstr_to_str(list(dir_symbol.children.keys()))

        # LEVEL 2 : variable names
        elif iLevel == 2:

            # subdir
            dir_symbol = self._get_symbol(self.lsda_root, path)

            # collect variable names
            return self._collect_variables(dir_symbol)

        # LEVEL 3 : read variable data
        elif iLevel == 3:
            return self._get_variable(path)

        # LEVEL 4+ : Error
        else:
            raise ValueError("Invalid path depth \"%d > 3\"." % len(iLevel))

    def _get_symbol(self, symbol, path):
        '''Get a symbol from a path via lsda

        Parameters
        ----------
        symbol : Symbol
            current directory which is a Lsda.Symbol

        Returns
        -------
        symbol : Symbol
            final symbol after recursive search of path
        '''

        # check
        if symbol == None:
            raise ValueError("Symbol may not be none.")

        # no further path, return current symbol
        if len(path) == 0:
            return symbol
        # more subsymbols to search for
        else:

            sub_path = list(path)  # copy
            next_symbol_name = sub_path.pop(0)

            next_symbol = symbol.get(next_symbol_name)
            if next_symbol == None:
                raise ValueError("Can not find: %s" % next_symbol_name)

            return self._get_symbol(next_symbol, sub_path)

    def _get_variable(self, path):
        '''Read a variable from a given path

        Parameters
        ----------
        path : list(str)
            path to the variable

        Returns
        -------
        data : np.ndarray of int or float
        '''

        dir_symbol = self._get_symbol(self.lsda_root, path[:-1])
        # variables are somehow binary strings ... dirs not
        variable_name = self._str_to_bstr(path[-1])

        # var in metadata
        if ("metadata" in dir_symbol.children) and (variable_name in dir_symbol.get("metadata").children):
            return np.asarray(dir_symbol.get("metadata").get(variable_name).read())
        # var in state data ... hopefully
        else:

            time = []
            data = []
            for subdir_name, subdir_symbol in dir_symbol.children.items():

                # skip metadata
                if subdir_name == "metadata":
                    continue

                # read data
                if variable_name in subdir_symbol.children:
                    state_data = subdir_symbol.get(variable_name).read()
                    if len(state_data) == 1:
                        data.append(state_data[0])
                    else:  # more than one data entry
                        data.append(state_data)
                    time += subdir_symbol.get(b"time").read()
                    #data += subdir_symbol.get(variable_name).read()

            # return sorted by time
            return np.array(data)[np.argsort(time)]

        raise ValueError("Could not find and read: %s" % str(path))

    def _collect_variables(self, symbol):
        '''Collect all variables from a symbol

        Parameters
        ----------
        symbol : Symbol

        Returns
        -------
        variable_names : list(str)

        Notes
        -----
            This function collect all variables from the state dirs and metadata.
        '''

        var_names = set()
        for subdir_name, subdir_symbol in symbol.children.items():
            var_names = var_names.union(subdir_symbol.children.keys())

        return self._bstr_to_str(list(var_names))

    @staticmethod
    def to_string(data_array):
        '''Convert a data series of numbers (usually ints) to a string

        Parameters
        ----------
        data : np.array of int
            some data array

        Returns
        -------
        string : str
            data array converted to characters

        Notes
        -----
            This is needed for the reason that sometimes the binary data
            within the files are strings. 

        Examples
        --------
            >>> # strings also are just plain numbers
            >>> binout.read("swforc","typenames")
            array([99, 111, 110, ...]) 
            >>> Binout.to_string( binout.read("swforc","typenames") )
            'constraint,weld,beam,solid,non nodal, ,solid assembly'
        '''

        return "".join([chr(entry) for entry in data_array])

    def _bstr_to_str(self, arg):
        '''Encodes or decodes a string correctly regarding python version

        Parameters
        ----------
        string : str/unicode/bytes

        Returns
        -------
        string : str
            converted to python version
        '''

        # in case of a list call this function with its atomic strings
        if isinstance(arg, (list, tuple)):
            return [self._bstr_to_str(entry) for entry in arg]

        # convert a string (dependent on python version)
        if not isinstance(arg, str):
            return arg.decode("utf-8")
        else:
            return arg

    def _str_to_bstr(self, string):
        '''Convert a string to a binary string python version independent

        Parameters
        ----------
        string : str

        Returns
        -------
        string : binary str
        '''

        if not isinstance(string, bytes):
            return string.encode("utf-8")
        else:
            return string

    def _remove_trailing_numbers(self, filepath):
        '''Find the additional binouts belonging to a base binout

        Parameters
        ----------
        filepath : str
            path to the base binout

        Returns
        -------
        binouts : list(str)
            list of filepaths to the additional binouts
        '''

        # filesystem variables
        binout_dir = os.path.dirname(filepath)
        binout_filename = ntpath.basename(filepath)
        binout_filename_numberless = binout_filename.rstrip("0123456789")
        filepath_numberless = os.path.join(
            binout_dir, binout_filename_numberless)

        return filepath_numberless

        # get all possible files
        filelist = glob.glob(filepath_numberless + "*")

        # filter those with number at the end
        cleaned_list = [entry.replace(filepath_numberless, "")
                        for entry in filelist]
        indexes = [ii for ii in range(len(cleaned_list))
                   if cleaned_list[ii].isnumeric()]
        filelist = [filelist[ii] for ii in indexes]
        filelist.append(binout_filename_numberless)

        # throw out non-existing files
        filelist = [entry for entry in filelist if os.path.isfile(entry)]

        return filelist
