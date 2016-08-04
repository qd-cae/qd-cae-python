from Element import Element
import numpy as np
class Tria(Element):
    def __init__(self, ElemID, PartID, Node1, Node2, Node3):
        self.thickness = 0.0
        self.ElemID = ElemID
        self.PartID = PartID
        self.Nodes = [Node1, Node2, Node3]

    def UpdateThicknessFromNodes(self):
        tempthick=0.0
        for node in self.getNodes():
            tempthick=tempthick+node.getThickness()
        self.thickness = tempthick/3

    def getArea(self):
        a = np.array([self.Nodes[0].x, self.Nodes[0].y, self.Nodes[0].z])
        b = np.array([self.Nodes[1].x, self.Nodes[1].y, self.Nodes[1].z])
        c = np.array([self.Nodes[2].x, self.Nodes[2].y, self.Nodes[2].z])
        return 0.5 * np.linalg.norm(np.cross(b-a, c-a ))