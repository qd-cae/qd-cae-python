import subprocess
# import logging
from Mesh import Mesh as Mesh
import os
import numpy as np
from datetime import datetime
from numpy import cross, eye, dot
from scipy.linalg import expm3, norm
from getpass import getuser

class MeshReaders:
    def readDynaMesh(self, file):
        nodesection = False
        elemsection = False
        elemthicksection = False
        partsection = False
        matsection1 = False
        matsection2 = False
        propsection1 = False
        propsection2 = False

        i=0
        # self.logger.info("LS-Dyna Reader Started: "+file)

        _Mesh=Mesh(file,"LS-Dyna")

        with open(file, "r") as f:
            for line in f:
                # if "*KEYWORD" in line:
                    # self.logger.debug("*KEYWORD found")

                if line.strip() == "*NODE":
                    # self.logger.debug("*NODE keyword found")
                    nodesection = True
                elif (nodesection == True) and (not("$" in line or "*" in line)):
                    _nodeID = int(line[0:8])
                    _x = float(line[9:24])
                    _y = float(line[25:40])
                    _z = float(line[41:56])
                    _Mesh.addNode(_nodeID, _x, _y, _z)
                elif "*" in line:
                    nodesection = False

                if line.strip() == "*ELEMENT_SHELL":
                    # self.logger.debug("*ELEMENT_SHELL keyword found")
                    elemsection = True
                elif (elemsection == True) and (not("$" in line or "*" in line)):
                    _elID = int(line[0:8])
                    _partID = int(line[9:16])
                    _n1 = int(line[17:24])
                    _n2 = int(line[25:32])
                    _n3 = int(line[33:40])
                    _n4 = int(line[41:48])
                    _Mesh.addElem(_elID, _partID, _n1, _n2, _n3, _n4)
                elif "*" in line:
                    elemsection = False

                if line.strip() == "*ELEMENT_SHELL_THICKNESS":
                    # self.logger.debug("*ELEMENT_SHELL_THICKNESS keyword found")
                    elemthicksection = True
                    i=0
                elif (elemthicksection == True) and (not("$" in line or "*" in line)):
                    if i == 0:
                        _elID = int(line[0:8])
                        _partID = int(line[9:16])
                        _n1 = int(line[17:24])
                        _n2 = int(line[25:32])
                        _n3 = int(line[33:40])
                        _n4 = int(line[41:48])
                        _Mesh.addElem(_elID, _partID, _n1, _n2, _n3, _n4)
                        i = i+1
                    elif i==1:
                        _Mesh.addNodalThickness(_n1,float(line[0:16]))
                        _Mesh.addNodalThickness(_n2,float(line[17:32]))
                        _Mesh.addNodalThickness(_n3,float(line[33:48]))
                        _Mesh.addNodalThickness(_n4,float(line[49:64]))
                        i = 0
                elif "*" in line:
                    elemthicksection = False

                if line.strip() == "*PART":
                    # self.logger.debug("*PART Keyword Found")
                    partsection = True
                    i=0
                elif (partsection == True) and (not("$" in line or "*" in line)):
                    if i == 0: _title=line.strip()
                    if i == 1:

                        _partID=int(line[0:10])
                        _propID=int(line[11:20])
                        _matID=int(line[21:30])
                        _Mesh.addPart(_partID, _title, _propID, _matID)
                    i=i+1
                elif "*" in line:
                    partsection = False


                if line.strip() == "*MAT_PIECEWISE_LINEAR_PLASTICITY" :
                    # self.logger.debug("*MAT_PIECEWISE_LINEAR_PLASTICITY Keyword Found")
                    matsection1 = True
                    i=0
                elif (matsection1 == True) and (not("$" in line or "*" in line)):
                    if i == 0:
                        _matID=int(line[0:10])
                        _rho=float(line[11:20])
                        _E=float(line[21:30])
                        _Mesh.addMat(_matID, _rho, _E)
                    i=i+1
                elif "*" in line:
                    matsection1 = False

                if line.strip() == "*MAT_PIECEWISE_LINEAR_PLASTICITY_TITLE" :
                    # self.logger.debug("*MAT_PIECEWISE_LINEAR_PLASTICITY_TITLE Keyword Found")
                    matsection2 = True
                    i=0
                elif (matsection2 == True) and (not("$" in line or "*" in line)):
                    if i == 0: title=line.strip()
                    if i == 1:
                        _matID=int(line[0:10])
                        _rho=float(line[11:20])
                        _E=float(line[21:30])
                        _Mesh.addMat(_matID, _rho, _E)
                    i=i+1
                elif "*" in line:
                    matsection2 = False

                if line.strip() == "*SECTION_SHELL":
                    # self.logger.debug("*SECTION_SHELL Keyword Found")
                    propsection1 = True
                    i=0
                elif (propsection1 == True) and (not("$" in line or "*" in line)):
                    if i == 0:
                        _propID=int(line[0:10])
                    if i == 1:
                        _thick=float(line[0:10])
                        _Mesh.addProp(_propID, _thick)
                    i=i+1
                elif "*" in line:
                    propsection1 = False

                if line.strip() == "*SECTION_SHELL_TITLE":
                    # # self.logger.debug("*SECTION_SHELL_TITLE Keyword Found")
                    propsection2 = True
                    i=0
                elif (propsection2 == True) and (not("$" in line or "*" in line)):
                    if i == 0: _title=line.strip()
                    if i == 1: _propID=int(line[0:10])
                    if i == 2:
                        _thick=float(line[0:10])
                        _Mesh.addProp(_propID, _thick)
                    i=i+1
                elif "*" in line:
                    propsection2 = False
        # self.logger.info("LS-Dyna Mesh successfully read")
        # self.logger.info("Number of Nodes: "+str(len(_Mesh.Nodelist)))
        # self.logger.info("Number of Elems: "+str(len(_Mesh.Elemlist)))
        # self.logger.info("Number of Props: "+str(len(_Mesh.Proplist)))
        # self.logger.info("Number of Mater: "+str(len(_Mesh.Matlist)))
        # self.logger.info("Number of Parts: "+str(len(_Mesh.Partlist)))
        # self.logger.info("Number of EThck: "+str(len(_Mesh.Elementalthickness)))
        # self.logger.info("Number of NThck: "+str(len(_Mesh.Nodalthickness)))
        return _Mesh

    def readRadiossMesh(self, file):
        nodesection = False
        SH3Nsection = False
        SHELLsection = False
        partsection = False
        matsection = False
        propsection = False

        i=0
        # self.logger.info("Radioss Reader Started: "+file)

        _Mesh=Mesh(file,"Radioss")

        with open(file, "r") as f:
            for line in f:
                # if "#RADIOSS STARTER" in line:
                    # self.logger.debug("#RADIOSS STARTER found")

                if "/NODE" in line[0:5]:
                    # self.logger.debug("/NODE keyword found")
                    nodesection = True
                elif (nodesection == True) and (not("#" in line or "/" in line)):
                    _nodeID = int(line[0:10])
                    _x = float(line[11:30])
                    _y = float(line[31:50])
                    _z = float(line[51:70])
                    _Mesh.addNode(_nodeID, _x, _y, _z)
                elif nodesection and "/" in line:
                    nodesection = False

                if "/SH3N" in line[0:5]:
                    # self.logger.debug("/SH3N keyword found")
                    _partid=int(line.strip().split("/")[-1])
                    SH3Nsection = True
                elif (SH3Nsection == True) and (not("#" in line or "/" in line)):
                    #print [line[0:10], partid, line[11:20], line[21:30], line[31:40]]
                    _elID = int(line[0:10])
                    _n1 = int(line[11:20])
                    _n2 = int(line[21:30])
                    _n3 = int(line[31:40])
                    _Mesh.addElem(_elID, _partid, _n1, _n2, _n3)
                    if line[91:100].strip() != "": _Mesh.addElementalThickness(line[0:10],line[91:100])
                elif SH3Nsection and "/" in line:
                    SH3Nsection = False

                if "/SHELL" in line[0:6]:
                    # self.logger.debug("/SHELL keyword found")
                    _partid=int(line.strip().split("/")[-1])
                    SHELLsection = True
                elif (SHELLsection == True) and (not("#" in line or "/" in line)):
                    #print [line[0:10], partid, line[11:20], line[21:30], line[31:40], line[41:50]]

                    _elID = int(line[0:10])
                    _n1 = int(line[11:20])
                    _n2 = int(line[21:30])
                    _n3 = int(line[31:40])
                    _n4 = int(line[41:50])
                    _Mesh.addElem(_elID, _partid, _n1, _n2, _n3, _n4)

                    try:
                        _elthick = float(line[91:100].strip())
                        if _elthick != 0.0 :
                            _Mesh.addElementalThickness(_elID,_elthick)
                    except:
                        pass
                elif SHELLsection and "/" in line:
                    SHELLsection = False

                if "/PART" in line[0:5]:
                    # self.logger.debug("/PART Keyword Found")
                    _partid=int(line.strip().split("/")[-1])
                    partsection = True
                    i=0
                elif (partsection == True) and (not("#" in line or "/" in line)):
                    if i == 0: _title=line.strip()
                    elif i == 1:
                        _propID=int(line[0:10])
                        _matID=int(line[11:20])
                        _Mesh.addPart(_partid, _title, _propID, _matID)
                    i=i+1
                elif partsection and "/" in line:
                    partsection = False

                if "/MAT/PLAS_TAB" in line[0:13]:
                    # self.logger.debug("/MAT/PLAS_TAB Keyword Found")
                    _matID=int(line.strip().split("/")[-1])
                    matsection = True
                    i=0
                elif (matsection == True) and (not("#" in line or "/" in line)):
                    if i == 0:
                        _title=line.strip()
                    elif i == 1:
                        _rho=float(line[0:20])
                    elif i == 2:
                        _E=float(line[0:20])
                        _Mesh.addMat(_matID, _rho, _E)
                    i=i+1
                elif matsection and "/" in line:
                    matsection = False

                if "/PROP/SHELL" in line[0:11]:
                    # self.logger.debug("/PROP/SHELL Keyword Found")
                    _propID=float(line.strip().split("/")[-1])
                    propsection = True
                    i=0
                elif (propsection == True) and (not("#" in line or "/" in line)):
                    if i == 0:
                        _title=line.strip()
                    elif i == 3:
                        _thick=float(line[21:40])
                        _Mesh.addProp(_propID, _thick)
                    i=i+1
                elif propsection and "/" in line:
                    propsection = False

        # self.logger.info("Radioss Mesh successfully read")
        # self.logger.info("Number of Nodes: "+str(len(_Mesh.Nodelist)))
        # self.logger.info("Number of Elems: "+str(len(_Mesh.Elemlist)))
        # self.logger.info("Number of Props: "+str(len(_Mesh.Proplist)))
        # self.logger.info("Number of Mater: "+str(len(_Mesh.Matlist)))
        # self.logger.info("Number of Parts: "+str(len(_Mesh.Partlist)))
        # self.logger.info("Number of EThck: "+str(len(_Mesh.Elementalthickness)))
        # self.logger.info("Number of NThck: "+str(len(_Mesh.Nodalthickness)))

        return _Mesh

    def writeDynaMesh(self, _Mesh, file):
        elemsection = False
        elemthicksection = False
        elemshellthicknesswritten = False
        elemshellwritten = False
        nodesection = False
        i=0

        ofile=open(file, "w")
        ofile.write("$ - Mubea TRB CAE - FE_Mesh_Handlers - writeDynaMesh\n")
        ofile.write("$ - Date/Time: "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+"\n")
        ofile.write("$ - User: "+getuser()+"\n")

        # self.logger.info("LS-Dyna Mesh will be written to: "+ file)
        # self.logger.info("Baseline Input is taken from: "+ _Mesh.getMeshFile())
        with open(_Mesh.getMeshFile(), "r") as f:
            for line in f:
                if (elemsection == True) and (not("$" in line or "*" in line)):
                    if (int(line[9:16]) in _Mesh.NUTProps):
                        if not elemshellthicknesswritten:
                            ofile.write("*ELEMENT_SHELL_THICKNESS\n")
                            elemshellthicknesswritten=True
                            elemthicksection=True
                            elemshellwritten=False
                        ofile.write(line)
                        #elem=_Mesh.getElemObjByID(int(line[0:8]))
                        #numbernodes=len(set(_Mesh.getElemlist()[int(line[0:8]][1:]))
                        n1=_Mesh.getNodalThickness(int(line[17:24]))
                        n2=_Mesh.getNodalThickness(int(line[25:32]))
                        n3=_Mesh.getNodalThickness(int(line[33:40]))
                        n4=_Mesh.getNodalThickness(int(line[41:48]))
                        s= "%16.8f"%n1+"%16.8f"%n2+"%16.8f"%n3+"%16.8f"%n4+"\n"
                        ofile.write(s)
                        i=i+1
                    else:
                        if not elemshellwritten:
                            ofile.write("*ELEMENT_SHELL\n")
                            elemthicksection=False
                            elemshellwritten=True
                            elemsection=True
                            elemshellthicknesswritten=False
                        ofile.write(line)
                elif (elemthicksection) and (not("$" in line or "*" in line)):
                    ofile.write(line)
                    #
                elif nodesection and (not("$" in line or "*" in line)):
                    nid=int(line[0:8])
                    x=float(_Mesh.Nodelist[nid][0])
                    y=float(_Mesh.Nodelist[nid][1])
                    z=float(_Mesh.Nodelist[nid][2])
                    s= "%8i"%nid+"%16.10f"%x+"%16.10f"%y+"%16.10f"%z+"\n"
                    ofile.write(s)
                elif line.strip() == "*ELEMENT_SHELL":
                    # self.logger.debug("*ELEMENT_SHELL found")
                    elemsection = True
                    elemthicksection = False
                    elemshellthicknesswritten = False
                    elemshellwritten = True
                    nodesection = False
                    # self.logger.debug("state variables [elemsection,elemshellwritten,elemthicksection,elemshellthicknesswritten,nodesection]"+str([elemsection,elemshellwritten,elemthicksection,elemshellthicknesswritten,nodesection]))
                    ofile.write(line)
                elif line.strip() == "*ELEMENT_SHELL_THICKNESS":
                    i=0
                    # self.logger.debug("*ELEMENT_SHELL_THICKNESS found")
                    elemsection = False
                    elemthicksection = True
                    elemshellthicknesswritten = True
                    elemshellwritten = False
                    nodesection = False
                    # self.logger.debug("state variables [elemsection,elemshellwritten,elemthicksection,elemshellthicknesswritten,nodesection]"+str([elemsection,elemshellwritten,elemthicksection,elemshellthicknesswritten,nodesection]))
                    ofile.write(line)
                elif line.strip() == "*NODE":
                    # self.logger.debug("*NODE found")
                    elemsection = False
                    elemthicksection = False
                    elemshellthicknesswritten = False
                    elemshellwritten = False
                    nodesection = True
                    # self.logger.debug("state variables [elemsection,elemshellwritten,elemthicksection,elemshellthicknesswritten,nodesection]"+str([elemsection,elemshellwritten,elemthicksection,elemshellthicknesswritten,nodesection]))
                    ofile.write(line)
                elif "*" in line and not "*ELEMENT_SHELL" in line and not "*NODE" in line:
                    elemsection = False
                    nodesection = False
                    ofile.write(line)
                else:
                    ofile.write(line)
        ofile.close
        # self.logger.info('Writing to LS-Dyna Mesh file completed')

    def writeRadiossMesh(self, _Mesh, file):
        nodesection = False
        SH3Nsection = False
        SHELLsection = False
        partsection = False
        matsection = False
        propsection = False

        elemthick=False
        partid = 0
        i=0

        ofile=open(file, "w")
        # self.logger.info("Radioss Mesh will be written to: "+ file)
        # self.logger.info("Baseline Input is taken from: "+ _Mesh.getMeshFile())
        with open(_Mesh.getMeshFile(), "r") as f:
            if len(_Mesh.Elementalthickness.keys())!=0: elemthick=True
            for line in f:
                if "#RADIOSS STARTER" in line:
                    ofile.write(line)
                    ofile.write("## - Mubea TRB CAE - FE_Mesh_Handlers - writeRadiossMesh\n")
                    ofile.write("## - Date/Time: "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+"\n")
                    ofile.write("## - User: "+getuser()+"\n")
                elif ((SH3Nsection == True) or (SHELLsection == True)) and (not("#" in line or "/" in line)):
                    elemid=int(line[0:10])
                    if elemthick:
                        if int(partid) in _Mesh.NUTProps:
                            thick = _Mesh.getElementalThickness(elemid)
                        else: thick = 0.0
                    else:
                        thick = 0.0
                    ofile.write(line[0:90]+"%10.8f"%thick+"\n")
                elif nodesection and (not("#" in line or "/" in line)):
                    nid=int(line[0:10])
                    x=float(_Mesh.Nodelist[nid][0])
                    y=float(_Mesh.Nodelist[nid][1])
                    z=float(_Mesh.Nodelist[nid][2])
                    s= "%10i"%nid+"%20.14f"%x+"%20.14f"%y+"%20.14f"%z+"\n"
                    ofile.write(s)
                elif "/SH3N" in line[0:5]:
                    # self.logger.debug("/SH3N found")
                    partid=line.strip().split("/")[2]
                    SH3Nsection = True
                    SHELLsection = False
                    nodesection = False
                    ofile.write(line)
                elif "/SHELL" in line[0:6]:
                    i=0
                    # self.logger.debug("/SHELL found")
                    partid=line.strip().split("/")[2]
                    SH3Nsection = False
                    SHELLsection = True
                    nodesection = False
                    ofile.write(line)
                elif "/NODE" in line[0:5]:
                    # self.logger.debug("/NODE found")
                    SH3Nsection = False
                    SHELLsection = False
                    nodesection = True
                    ofile.write(line)
                elif "/" in line:
                    SH3Nsection = False
                    SHELLsection = False
                    nodesection = False
                    ofile.write(line)
                else:
                    ofile.write(line)
        ofile.close
        # self.logger.info('Writing Radioss Mesh file completed')
