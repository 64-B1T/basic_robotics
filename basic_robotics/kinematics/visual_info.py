class vis_info:
    def __init__(self):
        # Universal
        self.geo_type = None
        self.origin = None

        #Mesh
        self.scale = 1,0
        self.file_name = None

        #Cylinder and Sphere
        self.radius = None 
        self.length = None #Cylinder only

        #Box 
        self.box_size = None 

    def setOrigin(self, origin):
        if origin is not None:
            self.origin = origin

    def setScale(self, scale):
        if scale is not None:
            self.scale = scale