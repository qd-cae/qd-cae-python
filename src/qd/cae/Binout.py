
import os
import sys
import struct
import numpy as np
from lsda import Lsda

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


    ## Get the labels of the file
    #
    # @param str folder_name = None : subdirectory to investigate. None is root
    #
    # Get the labels of either the top directories in the file (folder_name=None)
    # or the labels of data in the subdirectories, e.g  folder_name="matsum".
    # This routine is meant for looking into the file.
    def get_labels(self,folder_name=None):

        if folder_name == None:

            var_names = self.lsda.root.children.keys()
            return var_names

        else:

            name_symbol = self.lsda.root.get(folder_name)
            if not name_symbol:
                raise Exception("%s does not exist." % folder_name)

            # search vars ... a bit complicated
            #
            # each subdir is data written at a timestep so we need to iterate
            # through all in order to catch all vars
            var_names = set()
            for subdir_name,subdir_symbol in name_symbol.children.iteritems():

                # skip metadata
                if subdir_name == "metadata":
                    continue

                for subsubdir_name in subdir_symbol.children.keys():
                    var_names.add(subsubdir_name)

            var_names = list(var_names)

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
    def get_data(self,folder_name,variable_name):

        folder_link = self.lsda.root.get(folder_name)
        if not folder_link:
            raise Exception("%s does not exist." % folder_name)

        # collect
        time, data = [], []
        for subfolder_name, subfolder_link in folder_link.children.iteritems():

            if variable_name in subfolder_link.children:
                time += subfolder_link.children["time"].read()
                data += subfolder_link.children[variable_name].read()

        # convert
        time, data = np.asarray(time),np.asarray(data)

        # sort after time ... binout is not very structured ...
        indexes = time.argsort()
        time = time[indexes]
        data = data[indexes]

        return time, data


    ## Developers only: Scan the file subdirs.
    #
    # @param int nMaxChildren = 10 : limit of children to abort (e.g. we hit data)
    def _scan_file(self,nMaxChildren=10):
        self._print_tree_item(self.lsda.root,0,nMaxChildren=nMaxChildren)


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
