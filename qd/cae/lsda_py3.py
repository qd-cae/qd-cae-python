#!/usr/bin/python
#
# Module of code to read/write LSDA binary files.
#
import glob
import string
import struct
import sys
#
##################################################################
#
class LsdaError(Exception):
  '''This is only here so I can raise an error in case the data type
  sizes are not what I expect'''
  pass
#
##################################################################
#
# Handles all the low level file I/O.  Nothing here should be
# called directly by a user.
#
class _Diskfile:
  packsize = [0,"b","h",0,"i",0,0,0,"q"]
  packtype = [0,"b","h","i","q","B","H","I","Q","f","d","s"]
  sizeof = [0,1,2,4,8,1,2,4,8,4,8,1]
  def __init__(self,name,mode):
    self.mode = mode             # file open mode (r,r+,w,w+)
    self.name = name             # file name
    self.ateof = 0               # 1 if the file pointer is at EOF
    self.fp = open(name,mode+'b')
    if(mode[0] == 'r'):
      s = self.fp.read(8)
      header = struct.unpack("BBBBBBBB",s)
      if(header[0] > 8):
         self.fp.seek(header[0])
    else:
      header = [8,8,8,1,1,0,0,0]
#
# Determine if my native ordering is big or little endian....
#
      b = struct.unpack("bbbb",struct.pack("i",1))
      if(b[0]):
        header[5]=1
      else:
        header[5]=0
    self.lengthsize    = header[1]
    self.offsetsize    = header[2]
    self.commandsize   = header[3]
    self.typesize      = header[4]
    if(header[5] == 0):
      self.ordercode='>'
    else:
      self.ordercode='<'
    self.ounpack  =  self.ordercode+_Diskfile.packsize[self.offsetsize]
    self.lunpack  =  self.ordercode+_Diskfile.packsize[self.lengthsize]
    self.lcunpack = (self.ordercode+
                    _Diskfile.packsize[self.lengthsize]+
                    _Diskfile.packsize[self.commandsize])
    self.tolunpack = (self.ordercode+
                    _Diskfile.packsize[self.typesize]+
                    _Diskfile.packsize[self.offsetsize]+
                    _Diskfile.packsize[self.lengthsize])
    self.comp1 = self.typesize+self.offsetsize+self.lengthsize
    self.comp2 = self.lengthsize+self.commandsize+self.typesize+1
    if(mode[0] != 'r'):
#
# Write initial header and ST offset command.
#
      s=bytes('','UTF-8')
      for h in header:
        s=s+struct.pack("B",h)
      self.fp.write(s)
      self.writecommand(17,Lsda.SYMBOLTABLEOFFSET)
      self.writeoffset(17,0)
      self.lastoffset = 17
#
  def readcommand(self):
    '''Read a LENGTH,COMMAND pair from the file at the current location'''
    s = self.fp.read(self.lengthsize+self.commandsize)
    return struct.unpack(self.lcunpack,s)
  def writecommand(self,length,cmd):
    '''Write a LENGTH,COMMAND pair to the file at the current location'''
    s = struct.pack(self.lcunpack,length,cmd)
    self.fp.write(s)
  def readoffset(self):
    '''Read an OFFSET from the file at the current location'''
    s = self.fp.read(self.offsetsize)
    return struct.unpack(self.ounpack,s)[0]
  def writeoffset(self,offset,value):
    '''Write an OFFSET to the file at the given location'''
    self.fp.seek(offset,0)
    s = struct.pack(self.ounpack,value)
    self.fp.write(s)
    self.ateof=0
  def writelength(self,length):
    '''Write a LENGTH to the file at the current location'''
    s = struct.pack(self.lunpack,length)
    self.fp.write(s)
  def writecd(self,dir):
    '''Write a whole CD command to the file at the current location'''
    length = self.lengthsize+self.commandsize+len(dir)
    s = struct.pack(self.lcunpack,length,Lsda.CD)
    self.fp.write(s)
	# TODO
    if type(dir) is str:
      self.fp.write(bytes(dir,'utf-8'))
    else:
      self.fp.write(dir)
	  
  def writestentry(self,r):
    '''Write a VARIABLE command (symbol table entry) to the file at
    the current location'''
    length = 2*self.lengthsize+self.commandsize+len(r.name)+self.typesize+self.offsetsize
    s = struct.pack(self.lcunpack,length,Lsda.VARIABLE)
    self.fp.write(s)
    if type(r.name) is str:
      self.fp.write(bytes(r.name,'utf-8'))
    else:
      self.fp.write(r.name)
    s = struct.pack(self.tolunpack,r.type,r.offset,r.length)
    self.fp.write(s)
  def writedata(self,sym,data):
    '''Write a DATA command to the file at the current location'''
    nlen = len(sym.name)
    length = self.lengthsize+self.commandsize+self.typesize+1+nlen+self.sizeof[sym.type]*sym.length
    sym.offset = self.fp.tell()
    self.fp.write(struct.pack(self.lcunpack,length,Lsda.DATA))
    self.fp.write(struct.pack("bb",sym.type,nlen)+bytes(sym.name,'utf-8'))
#    fmt=self.ordercode+self.packtype[sym.type]*sym.length
    fmt="%c%d%c" % (self.ordercode,sym.length,self.packtype[sym.type])
    self.fp.write(struct.pack(fmt,*data))
    sym.file = self
#
##################################################################
#
#  A directory tree structure.  A Symbol can be a directory (type==0)
#  or data
#
class Symbol:
  def __init__(self,name="",parent=0):
    self.name = name      # name of var or directory
    self.type = 0         # data type
    self.offset = 0       # offset of DATA record in file
    self.length = 0       # number of data entries, or # of children
    self.file = 0         # which file the data is in
    self.children = {}    # directory contents
    self.parent = parent  # directory that holds me
    if(parent):
      parent.children[name] = self
      parent.length = len(parent.children)
  def path(self):
    '''Return absolute path for this Symbol'''
    if(not self.parent):
      return "/"
    sym=self
    ret='/'+sym.name
    while(sym.parent and sym.parent.name != '/'):
      sym=sym.parent
      ret='/'+sym.name+ret
    return ret
  def get(self,name):
    '''Return the Symbol with the indicated name.  The name can be
    prefixed with a relative or absolute path'''
# If I am just a variable, let my parent handle this
    if(self.type != 0):
      return self.parent.get(name)
# If I have this variable, return it
    if(name in self.children):
      return self.children[name]
# If name has a path component, then look for it there
    if(name[0]=="/"):  # absolute path
      parts = name.split("/")[1:]
      sym=self
      while (sym.parent):
        sym=sym.parent
      for i in range(len(parts)):
        if(parts[i] in sym.children):
          sym=sym.children[parts[i]]
        else:
          return None
      return sym
    if(name[0]=="."):  # relative path
      parts = name.split("/")[1:]
# Throw out any "." in the path -- those are just useless....
      parts = filter(lambda p: p != '.', parts)
      if(len(parts)==0):
        return self
      sym=self
      for i in range(parts):
        if(parts[i] == '..'):
          if(sym.parent):
            sym=sym.parent
        elif(parts[i] in sym):
          sym=sym.children[parts[i]]
        else:
          return None
      return sym
# Not found
    return None
  def lread(self,start=0,end=2000000000):
    '''Read data from the file.
    If this symbol is a DIRECTORY, this returns a sorted list of the
    contents of the directory, and "start" and "end" are ignored.
    Otherwise, read and return data[start:end] (including start but
    not including end -- standard Python slice behavior).
    This routine does NOT follow links.'''
    if(self.type == 0):  # directory -- return listing
      return sorted(self.children.keys())
    if(end > self.length):
      end = self.length
    if(end < 0):
      end = self.length+end
    if(start > self.length):
      return ()
    if(start < 0):
      start = self.length+start
    if(start >= end):
      return ()
    size=_Diskfile.sizeof[self.type]
    pos = self.offset+self.file.comp2+len(self.name)+start*size
    self.file.fp.seek(pos)
    self.file.ateof=0
#    format = self.file.ordercode + _Diskfile.packtype[self.type]*(end-start)
#    return struct.unpack(format,self.file.fp.read(size*(end-start)))
    format = "%c%d%c" % (self.file.ordercode,(end-start),_Diskfile.packtype[self.type])
    if(self.type == Lsda.LINK):
      return struct.unpack(format,self.file.fp.read(size*(end-start)))[0]
    else:
      return struct.unpack(format,self.file.fp.read(size*(end-start)))
  def read(self,start=0,end=2000000000):
    '''Read data from the file.  Same as lread, but follows links'''
    return _resolve_link(self).lread(start,end)
  def read_raw(self,start=0,end=2000000000):
    '''Read data from the file and return as bytestring'''
    if(self.type == 0):  # directory -- return listing
      return sorted(self.children.keys())
    if(end > self.length):
      end = self.length
    if(end < 0):
      end = self.length+end
    if(start > self.length):
      return ()
    if(start < 0):
      start = self.length+start
    if(start >= end):
      return ()
    size=_Diskfile.sizeof[self.type]
    pos = self.offset+self.file.comp2+len(self.name)+start*size
    self.file.fp.seek(pos)
    self.file.ateof=0
    size=size*(end-start)
    return self.file.fp.read(size)
#
##################################################################
#
# Follow a link to find what it finally resolves to
#
def _resolve_link(var):
  ret = var
  while(ret.type == Lsda.LINK):
    ret = ret.get(ret.lread())
  return ret
#
##################################################################
#
# Read a VARIABLE record from the file, and construct the proper Symbol
# Users should never call this.
#
def _readentry(f,reclen,parent):
  s = f.fp.read(reclen)
  n = reclen-f.comp1
  name = s[:n]
# If parent already has a symbol by this name, orphan it....
  #if(parent.children.has_key(name)):
  if(name in parent.children):
    var = parent.children[name]
  else:
    var = Symbol(name,parent)
  (var.type,var.offset,var.length) = struct.unpack(f.tolunpack,s[n:])
  var.file = f
#
##################################################################
#
# Read all the SYMBOLTABLEs in the current file
# Users should never call this.
#
#
def _readsymboltable(lsda,f):
  f.ateof=0
  while 1:
    f.lastoffset = f.fp.tell()
    offset = f.readoffset()
    if(offset == 0): return
    f.fp.seek(offset)
    (clen,cmd) = f.readcommand()
    if(cmd != Lsda.BEGINSYMBOLTABLE): return
    while 1:
      (clen,cmd) = f.readcommand()
      clen = clen - f.commandsize - f.lengthsize
      if(cmd == Lsda.CD):
        path = f.fp.read(clen)
        ss=lsda.cd(path,1)
      elif(cmd == Lsda.VARIABLE):
        _readentry(f,clen,lsda.cwd)
      else:   # is end of symbol table...get next part if there is one
        break
#
#
##################################################################
#
# Flush all dirty symbols out to the file.
# Users should never call this.
#
#def keyfunction(item):
#  return item[1]
def _writesymboltable(lsda,f):
#
# Collect all the symbols we want to write out, and sort
# them by path.  This is a bit strange: the symbols don't store
# the path, but build it when needed.  So build it, and store
# (symbol,path) pairs, then sort by path.  "path" returns the full
# path to the symbol, and we only want the directory it is in, so
# get the path of its parent instead.
#
  if(len(lsda.dirty_symbols)==0):
    return
#
  slist=[]
  for s in lsda.dirty_symbols:
    p=s.parent.path()
    slist.append((s,p))
  #slist.sort(key = lambda r1,r2: cmp(r1[1],r2[1]))
  slist.sort(key = lambda x: x[1])
  lsda.dirty_symbols=set()
#
# Move to end of the file and write the symbol table
#
  if(not f.ateof):
    f.fp.seek(0,2)
    f.ateof=1
  start_st_at = f.fp.tell()
  f.writecommand(0,Lsda.BEGINSYMBOLTABLE)
  cwd = None
# lsda.lastpath=None
#
# Write all records
#
  for (s,path) in slist:
    if(path != cwd):
      cdcmd = _get_min_cd(cwd,path)
      f.writecd(cdcmd)
      cwd=path
    f.writestentry(s)
#
# Finish ST: write END record, and patch up ST length
#
  cmdlen = f.offsetsize+f.lengthsize+f.commandsize
  f.writecommand(cmdlen,Lsda.ENDSYMBOLTABLE)
  nextoffset=f.fp.tell()
  f.writeoffset(nextoffset,0)
  cmdlen = nextoffset+f.offsetsize-start_st_at
  f.fp.seek(start_st_at)
  f.writelength(cmdlen)
#
# Purge symbol table, if we are only writing
#
  if(f.mode == "w"):
    cwd=lsda.cwd
    cwd.children = {}
    while(cwd.parent):
      cwd.parent.children = {}
      cwd.parent.children[cwd.name] = cwd
      cwd = cwd.parent
#
# And add link from previous ST
#
  f.writeoffset(f.lastoffset,start_st_at)
  f.lastoffset = nextoffset
  f.ateof=0
#
##################################################################
#
# Given two absolute paths, return the shortest "cd" string that
# gets from the first (cwd) to the second (cd)
#
def _get_min_cd(cwd,cd):
  if(cwd == None):
    return cd
#
# Find common part of path
#
  have = cwd.split("/")[1:]
  want = cd.split("/")[1:]
  nhave = len(have)
  nwant = len(want)
  n=min(nhave,nwant)
  head=0
  headlength=0
  for i in range(n):
    if(have[i] != want[i]):
      break
    head=i+1
    headlength=headlength+len(have[i])
  if(head == 0):
    return cd
#
# head = # of common components.
# headlength = string length of common part of path (sans "/" separators)
# tail1 = # components we would need ".." leaders for
#
  tail1 = nhave - head
#
# Now see if "cd" is shorter than "../../tail_part"
#
  if(2*tail1 >= headlength):
    return cd
#
# nope, the ".." version is shorter....
#
  return tail1*"../"+"/".join(want[head:])
#
#
##################################################################
#
# Main class: holds all the Symbols for an LSDA file, and has methods
# for reading data from and writing data to the file
#
class Lsda:
  CD=2
  DATA=3
  VARIABLE=4
  BEGINSYMBOLTABLE=5
  ENDSYMBOLTABLE=6
  SYMBOLTABLEOFFSET=7
  I1=1
  I2=2
  I4=3
  I8=4
  U1=5
  U2=6
  U4=7
  U8=8
  R4=9
  R8=10
  LINK=11
  
  def __init__(self,files,mode="r"):
    '''Creates the LSDA structure, opens the file and reads the
    SYMBOLTABLE (if reading), or creates the initial file contents
    (if writing).  "files" is a tuple of file names to be opened
    and treated as a single file.  All the %XXX continuation files
    will be automatically included.  "mode" is the file open mode:
    "r", "r+", "w", or "w+".  If a "w" mode is selected, "files"
    must contain only a single file name'''
#
# If they only input a single name, put it in a tuple, so I can
# accept input of either kind
#
    if(not types_ok):
      raise LsdaError
    if type(files) != type((1,)) and type(files) != type([1]):
      files=(files,)
    self.files = []
#
    if(mode[0] == 'r'):
#
# Open all the files in the list that is input, and anything
# that looks like a continuation of one of them.
#
      nameset = set()
      for name in files:
        nameset.add(name)
        nameset=nameset.union(set(glob.glob(name+"%[0-9][0-9]*")))
#
# Convert to a list and sort, because if I'm going to be writing,
# I want the last one in the list to be the last one of its family
#
      namelist = list(nameset)
      namelist.sort()
      for file in namelist:
        self.files.append(_Diskfile(file,mode))
      self.root = Symbol("/")
      for f in self.files:
#
# We are already positioned to read the SYMBOLTABLEOFFSET record
#
        (clen,cmd) = f.readcommand()
        self.cwd = self.root
        if(cmd == Lsda.SYMBOLTABLEOFFSET):
          _readsymboltable(self,f)
    else:
      if(len(files) > 1):
        return None    # can't open multiple files for WRITING
      self.files.append(_Diskfile(files[0],mode))
      self.root = Symbol("/")
    self.cwd = self.root
    self.dirty_symbols = set()
    self.lastpath = None
    self.mode = mode
#
# writing will always be to the last one of the files
#
    if(mode == "r"):
      self.fw = None
      self.make_dirs = 0
    else:
      self.fw = self.files[-1]
      self.make_dirs = 1
  def __del__(self):  # close files
    self.flush()
    for f in self.files:
      if(not f.fp.closed):
        f.fp.close()
  def cd(self,path,create=2):      # change CWD
    '''Change the current working directory in the file.  The optional
    argument "create" is for internal use only'''
    # DEBUG FIX qd-codie:
    # Someone forgot to decode bytes into str here. Due to this the 
    # following comparisons always went wrong (of course they were)
    # not similar ...
    if isinstance(path,bytes):
      path = path.decode("utf-8")
    if(path == "/"):
      self.cwd = self.root
      return self.root
    if(path[-1] == "/"):       # remove trailing /
      path=path[:-1]
    if(path[0] == "/"):        # absolute path
      path=path[1:]
      self.cwd = self.root
    #path = string.split(path,"/")
    #print(type(path))
    if type(path) is bytes:
      path = str(path,'utf-8').split("/")
    else:
      path = path.split('/')
    
    for part in path:
      if(part == ".."):
        if(self.cwd.parent):
          self.cwd = self.cwd.parent
      else:
        #if(self.cwd.children.has_key(part)):
        if(part in self.cwd.children):
          self.cwd = self.cwd.children[part]
          if(self.cwd.type != 0):  # component is a variable, not a directory!
            self.cwd = self.cwd.parent
            break
        elif(create == 1 or (create == 2 and self.make_dirs == 1)):
          self.cwd = Symbol(part,self.cwd)  # Create directory on the fly
        else:    # component in path is missing
          break
    return self.cwd
  def write(self,name,type,data):
    '''Write a new DATA record to the file.  Creates and returns
    the Symbol for the data written'''
    if(self.fw == None):
      return None
# want a tuple, but if they hand us a single value that should work too...
    try:
      x=data[0]
    except TypeError:
      data=(data,)
    pwd = self.cwd.path()
    if(not self.fw.ateof):
      self.fw.fp.seek(0,2)
      self.fw.ateof=1
    if(pwd != self.lastpath):
      cdcmd = _get_min_cd(self.lastpath,pwd)
      self.fw.writecd(cdcmd)
      self.lastpath=pwd
# Overwrite existing symbol if there is one
    if(name in self.cwd.children):
      sym=self.cwd.children[name]
    else:
      sym=Symbol(name,self.cwd)
    sym.type = type
    sym.length = len(data)
    self.fw.writedata(sym,data)
    self.dirty_symbols.add(sym)
    return sym
  def close(self):
    '''Close the file'''
    self.flush()
    for f in self.files:
      if(not f.fp.closed):
       f.fp.close()
    self.files=[]
  def get(self,path):
    '''Return the Symbol with the indicated name.  The name can be
    prefixed with a relative or absolute path'''
    return self.cwd.get(path)
  def flush(self):  # write ST and flush file
    '''Write a SYMBOLTABLE as needed for any new DATA, and flush the file'''
    if(self.fw == None or self.fw.fp.closed):
      return
    _writesymboltable(self,self.fw)
    self.fw.fp.flush()
  def filesize(self):
    '''Returns the current size, on disk, of the file we are currently
    writing to.  Returns 0 for files that are opened readonly'''
    if(self.fw == None):
      return 0
    if(not self.fw.ateof):
      self.fw.fp.seek(0,2)
      self.fw.ateof=1
    return self.fw.fp.tell()
  def nextfile(self):  # Open next file in sequence
    '''Flush the current output file and open the next file in the
    sequence'''
    if(self.fw == None):
      return None
    if(not self.fw.fp.closed):
       _writesymboltable(self,self.fw)
       self.fw.fp.flush()
    parts=self.fw.name.split("%")
    if(len(parts) == 1):
      ret=1
      newname = parts[0]+"%001"
    else:
      ret=int(parts[1])+1
      newname = "%s%%%3.3d" % (parts[0],ret)
    if(self.mode == "w"):
      self.fw = _Diskfile(newname,"w")
    else:
      self.fw = _Diskfile(newname,"w+")
    self.files.append(self.fw)
    self.lastpath = None
    return ret
#
# def testit():
#   files = ["binout0000"]
#   file=Lsda(files)
#   file.cd("/matsum/d000120")
#   top=file.cwd
#   var = top.get("internal_energy")
#   print var.path(),"=",var.read()
# 
# if __name__ == "__main__":
#   testit()
#
# Do sanity check of type lengths.  types_ok will be checked whenever
# a new file is opened, and if things don't match then an exception will
# be raised -- this should prevent unknown errors due to type size problems
#
types = [("b",1),("h",2),("i",4),("q",8),("f",4),("d",8)]
x=17
types_ok = 1
for (a,b) in types:
  s=struct.pack(a,x)
  if(len(s) != b):
    print( "LSDA: initialization error")
    print( "Data type %s has length %d instead of %d" % (a,len(s),b))
    types_ok = 0
