#from Node cimport Node
cdef class Element:
    cdef public int ElemID,PartID
    cdef public double thickness
    cdef public list Nodes

    cpdef setID(self, int ElemID)
    cpdef setPartID(self, int PartID)
    # def setNodes(self, list *Nodes)

    cpdef int getID(self)
    cpdef int getPartID(self)
    cpdef list getNodes(self)
    cpdef double getThickness(self)
    cpdef double getVolume(self)
