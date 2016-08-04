from Element cimport Element
from Node cimport Node
from Part cimport Part
cdef class Mesh:
    cdef public dict Nodelist, Nodalthickness, Elementalthickness, Elemlist, PartElemlist, Partlist, Matlist, Proplist, PartObjList
    cdef public list NUTProps
    cdef public str Meshfile, Meshformat

    cpdef addNode(self, int NodeID, double x, double y, double z)
    # cpdef addElem(self, int ElemID, int PartID, *Nodes)
    cpdef addNodalThickness(self, int NodeID, double thickness)
    cpdef addElementalThickness(self, int ElemID, double thickness)
    cpdef addPart(self, int PartID, str title, int PropID, int MatID)
    cpdef addMat(self, int MatID, double Rho, double E)
    cpdef addProp(self, int PropID, double Thickness)

    cpdef double getNodalThickness(self, int NodeID)
    cpdef double getElementalThickness(self, int ElemID)

    cpdef InitPartObj(self, int PartID)
    # def InitAllObj(self)

    cpdef setMeshFile(self, str _file)
    cpdef setMeshFormat(self, str _format)
    cpdef str getMeshFormat(self)
    cpdef str getMeshFile(self)

    cpdef list getNodelist(self)
    cpdef list getElemlist(self)
    cpdef double getMassByPartID(self, int PartID)
    cpdef double getVolumeByPartID(self, int PartID)
    cpdef double getAreaByPartID(self, int PartID)
    cpdef double getAreaByElemID(self, int ElemID)
    cpdef double getVolumeByElemID(self, int ElemID)

    cpdef Node getNodeObjByID(self, int NodeID)
    cpdef Element getElemObjByID(self, int ElemID)

    cpdef list getNodeByID(self, int NodeID)
    cpdef list getNodesByElemID(self, int ElemID)
    cpdef list getElemByID(self, int ElemID)

    cpdef dict getPartlistObj(self)
    cpdef Part getPartByID(self, int PartID)
    cpdef list getRectangleBounds(self)
