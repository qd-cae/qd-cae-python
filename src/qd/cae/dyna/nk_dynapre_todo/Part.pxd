from Element cimport Element
from Material cimport Material
from Property cimport Property
# from libcpp cimport bool
cdef class Part:
    cdef public int PartID
    cdef public str title
    cdef public bint nonUniformThickness
    cdef public Material Mat
    cdef public Property Prop
    cdef public list ElemObj
    cdef public dict Elemlist,Nodelist

    cpdef addElem(self, Element Elem)

    cpdef setNonUniformThickness(self, bint _TRBbool)
    cpdef bint isNonUniformThickness(self)
    # cpdef list of Element getElemObj(self)
    cpdef int getNumElem(self)
    cpdef list getElemlist(self)
    cpdef list getNodelist(self)
    cpdef double getPartArea(self)
    cpdef double getPartVolume(self)
    cpdef double getPartMass(self)
    cpdef int getPartID(self)
    cpdef char getPartname(self)
