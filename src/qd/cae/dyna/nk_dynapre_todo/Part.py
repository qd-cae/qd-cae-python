class Part:
    def __init__(self, PartID, title, Mat, Prop):
        self.nonUniformThickness=False
        self.PartID=PartID
        self.Mat=Mat
        self.Prop=Prop
        self.title=title
        self.ElemObj=[]
        self.Elemlist={}
        self.Nodelist={}


    def addElem(self, Elem):
        self.ElemObj.append(Elem)
        self.Elemlist[Elem.getID()]=Elem
        for node in Elem.getNodes():
            self.Nodelist[node.getID()]=node

    def getNumElem(self):
        return len(self.ElemObj)

    def getElemObj(self):
        return self.ElemObj

    def getElemlist(self):
        return self.Elemlist

    def getNodelist(self):
        return self.Nodelist

    def getPartArea(self):
        Area=0.0
        for elem in self.ElemObj:
            Area=Area+elem.getArea()
        return Area

    def getPartVolume(self):
        Volume=0.0
        if self.isNonUniformThickness():
            for elem in self.ElemObj:
                Volume=Volume+elem.getVolume()
        else:
            Volume = self.Prop.thickness*self.getPartArea()
        return Volume

    def getPartMass(self):
        return self.Mat.Rho*self.getPartVolume()

    def getPartID(self):
        return self.PartID

    def getPartname(self):
        return self.title

    def setNonUniformThickness(self, nonUniformThickness):
        self.nonUniformThickness=nonUniformThickness

    def isNonUniformThickness(self):
        return self.nonUniformThickness
