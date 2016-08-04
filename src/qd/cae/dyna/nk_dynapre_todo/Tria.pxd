from Element cimport Element
cdef class Tria(Element):
    #cdef public int ElemID,PartID
    #cdef public double thickness
    #cdef public list Nodes
    
    cpdef double UpdateThicknessFromNodes(self)
    cpdef double getArea(self)
