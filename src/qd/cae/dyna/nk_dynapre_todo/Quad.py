from Element import Element
import numpy as np
class Quad(Element):
    def __init__(self, ElemID, PartID, Node1, Node2, Node3, Node4):
        self.thickness = 0.0
        self.ElemID = ElemID
        self.PartID = PartID
        self.Nodes = [Node1, Node2, Node3, Node4]

    def UpdateThicknessFromNodes(self):
        tempthick=0.0
        for node in self.getNodes():
            tempthick=tempthick+node.getThickness()
        self.thickness = tempthick/4
        return self.thickness

    def getArea(self):
        a = np.array([self.Nodes[0].x, self.Nodes[0].y, self.Nodes[0].z])
        b = np.array([self.Nodes[1].x, self.Nodes[1].y, self.Nodes[1].z])
        c = np.array([self.Nodes[2].x, self.Nodes[2].y, self.Nodes[2].z])
        d = np.array([self.Nodes[3].x, self.Nodes[3].y, self.Nodes[3].z])
        return 0.5 * np.linalg.norm(np.cross(c-a, d-b ))