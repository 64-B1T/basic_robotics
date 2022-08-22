"""Holding File for Visual Info."""
from ..general import tm

class vis_info:
    """Holding Class with better organization than a simple list."""

    def __init__(self):
        """Initialize a new VisInfo object."""
        # Universal
        self.geo_type = None
        self.origin = tm()

        #Mesh
        self.scale = 1,0
        self.file_name = None

        #Cylinder and Sphere
        self.radius = None 
        self.length = None #Cylinder only

        #Box 
        self.box_size = None 

    def setOrigin(self, origin):
        """
        Set origin of the visual object.

        Args:
            origin (tm): New Origin
        """
        if origin is not None:
            self.origin = origin

    def setScale(self, scale):
        """
        Set New Scale.

        Args:
            scale (list[float]): New Scale Factor
        """
        if scale is not None:
            self.scale = scale

    def __str__(self):
        """
        Get String representation.

        Returns:
            str: String representation.
        """
        if self.geo_type == 'mesh':
            return('Mesh: ' + str(self.file_name))
        if self.geo_type == 'cyl':
            return('Cylinder: ' + str(self.radius) + " " + str(self.length))
        if self.geo_type == 'spr':
            return('Sphere: ' + str(self.radius) + " " + str(self.length))
        if self.geo_type == 'box':
            return('Box: ' + str(self.box_size))
        return("Undefined")