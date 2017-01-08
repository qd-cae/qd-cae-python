
import os
import sys
import struct
import numpy as np

import sys
if sys.version_info[0] < 3:
    from .lsda_py2 import Lsda
else:
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


## This class is meant to read binouts from LS-Dyna
#
# This class is only a utility wrapper for Lsda from LSTC.
class Binout:


    ## Constructor for a binout
    #
    # @param str filepath
    def __init__(self,filepath):

        # check
        if not os.path.isfile(filepath):
            raise IOError("File %s does not exist." % filepath)

        self.filepath = filepath
        self.lsda = Lsda(filepath,"r")

        #if sys.version_info[0] < 3:
        #    self.lsda_root = self.lsda.root
        #else:
        #    self.lsda_root = self.lsda.root.children[""]
        self.lsda_root = self.lsda.root
        

    ## Get the labels of the file
    #
    # @param str folder_name = None : subdirectory to investigate. None is root
    #
    # Get the labels of either the top directories in the file (folder_name=None)
    # or the labels of data in the subdirectories, e.g  folder_name="matsum".
    # This routine is meant for looking into the file.
    def get_labels(self,folder_name=None):

        # highest level info
        if folder_name == None:

            return self._bStrToStr_list( list( self.lsda_root.children.keys() ) )

        # subdir info
        else:

            name_symbol = self.lsda_root.get(folder_name)
            if not name_symbol:
                raise ValueError("%s does not exist." % folder_name)

            # search vars ... a bit complicated
            #
            # each subdir is data written at a timestep so we need to iterate
            # through all in order to catch all vars
            var_names = set()
            for subdir_name, subdir_symbol in name_symbol.children.items():
                
                subdir_name = self._bStrToStr(subdir_name)

                # metadata
                if subdir_name == "metadata":
                    # nodout metadata contains node ids
                    if folder_name == "nodout" or folder_name == "swforc":
                        if 'ids' in self._bStrToStr_list(list(subdir_symbol.children.keys())):
                            var_names.add('ids')
                    continue

                for subsubdir_name in subdir_symbol.children.keys():
                    var_names.add(subsubdir_name)

            var_names = self._bStrToStr_list( list(var_names) ) 

            # remove time, since we add it anyways for every var
            if "time" in var_names:
                del var_names[var_names.index("time")]

            return var_names


    ## Get some data from the Binout
    #
    # @param str folder_name : name of top level folder e.g. "matsum"
    # @param str variable_name : name of variable e.g. "internal_energy"
    #
    # Fetch data from the binout. If you have no idea what data is inside and
    # how it is called, use get_labels() first.
    def get_data(self, folder_name, variable_name):

        folder_link = self.lsda_root.get(folder_name)
        if not folder_link:
            raise Exception("%s does not exist." % folder_name)

        # special node ids in nodout treatment
        elif (folder_name == "nodout" or folder_name == "swforc") and (variable_name == "ids"):

            return np.asarray(folder_link.children['metadata'].children[self._str_to_bstr('ids')].read())

        # treatment of arbitrary data
        else:

            # collect
            time, data = [], []
            for subfolder_name, subfolder_link in folder_link.children.items():

                if variable_name in self._bStrToStr_list(list(subfolder_link.children)):
                    time += subfolder_link.children[self._str_to_bstr("time")].read()
                    _tmp = subfolder_link.children[self._str_to_bstr(variable_name)].read()
                    if len(_tmp) == 1:
                        data += _tmp
                    else:
                        data.append(_tmp)

            # convert
            time, data = np.asarray(time),np.asarray(data)

            # sort after time ... binout is not very structured ...
            indexes = time.argsort()
            time = time[indexes]
            data = data[indexes]

        return time, data

    
    ## Encodes or decodes a list of string correctly regarding python version
    #
    # @param list(str/unicode/bytes) string
    # @return list(str) string : converted to python version
    #
    def _bStrToStr_list(self, strings):

        return [ self._bStrToStr(string) for string in strings ]


    ## Encodes or decodes a string correctly regarding python version
    #
    # @param str/unicode/bytes string
    # @return str string : converted to python version
    #
    def _bStrToStr(self, string):

        if sys.version_info[0] < 3:

            if not isinstance(string, bytes):
                return string.encode("utf-8")
            else:
                return string

        else:

            if not isinstance(string, str):
                return string.decode("utf-8")
            else:
                return string

    ## Convert a string to a binary string python version independent
    #
    # @param str string
    # @param bstr string
    def _str_to_bstr(self,string):

        if sys.version_info[0] < 3:
            return string
        else:
            return string.encode("utf-8")

    ## Developers only: Scan the file subdirs.
    #
    # @param int nMaxChildren = 10 : limit of children to abort (e.g. we hit data)
    def _scan_file(self,nMaxChildren=10):
        self._print_tree_item(self.lsda_root,0,nMaxChildren=nMaxChildren)


    ## Developers only: print a lsda symbol item recursively
    #
    # @param Symbol item : symbol instance
    # @param int level : current depth level
    # @param int nMaxChildren = 10 : limit of children to abort (e.g. we hit data)
    # @param bool stop_level : level at which we stop looking deeper
    def _print_tree_item(self,item,level,nMaxChildren=10,stop_level=None):

        if item.type == 0: # dir
            print("%s> %s" % ("-"*level,item.name))
            if len(item.children) < nMaxChildren and level < stop_level:
                for child_name in item.children:
                    self._print_tree_item(item.children[child_name],level+1,nMaxChildren,stop_level=stop_level)
