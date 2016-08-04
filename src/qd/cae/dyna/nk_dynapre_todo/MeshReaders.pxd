from Mesh cimport Mesh
cdef class MeshReaders:
    cpdef Mesh readDynaMesh(self, str file)
    cpdef Mesh readRadiossMesh(self, str file)
    cpdef writeDynaMesh(self, Mesh _Mesh, str file)
    cpdef writeRadiossMesh(self, Mesh _Mesh, str file)
