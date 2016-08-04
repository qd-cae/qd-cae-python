cdef class Node:
    cdef public int NodeID
    cdef public double x,y,z,thickness

    cpdef setCoord(self, double x, double y, double z)
    cpdef setThickness(self, double thick)

    cpdef int getID(self)
    cpdef list getCoord(self)
    cpdef double getThickness(self)
