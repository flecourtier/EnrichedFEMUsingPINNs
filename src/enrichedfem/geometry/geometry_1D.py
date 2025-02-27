PI = 3.14159265358979323846

class Line:
    """Represents a 1D line segment.

    This class defines a 1D line segment using its endpoints 'a' and 'b'.
    The line segment is stored as a list containing a single tuple (a, b).
    
    Args:
        a (float): The left endpoint of the line segment.
        b (float): The right endpoint of the line segment.
    """
    def __init__(self,a,b):
        self.box = [[a,b]]
        
class Line1(Line):
    """Represents a 1D line segment.

    This class defines the 1D line segment [0,1].    
    """
    def __init__(self):
        super().__init__(0.0,1.0)