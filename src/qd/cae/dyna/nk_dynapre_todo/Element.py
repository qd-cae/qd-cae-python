class Element:
    def __init__(self, ElemID, PartID, *Nodes):
        self.thickness= 0.0
        self.ElemID = ElemID
        self.PartID = PartID
        self.Nodes = Nodes

    def setID(self, ElemID):
        self.ElemID = ElemID

    def setPartID(self, PartID):
        self.PartID = PartID

    def setNodes(self, *Nodes):
        self.Nodes = Nodes

    def getID(self):
        return self.ElemID

    def getPartID(self):
        return self.PartID

    def getNodes(self):
        return self.Nodes

    def getThickness(self):
        return self.thickness

    def getVolume(self):
        return self.getArea()*self.thickness
