class Node:
    def __init__(self, NodeID, x, y, z):
        self.NodeID = NodeID
        self.thickness = 0.0
        self.x = x
        self.y = y
        self.z = z

    def setCoord(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def setThickness(self, thick):
        self.thickness=thick

    def getID(self):
        return self.NodeID

    def getCoord(self):
        return [self.x, self.y, self.z]

    def getThickness(self):
        return self.thickness