PI = 3.14159265358979323846

class Square:
    """Represents a square.

    This class defines a square [a,b]x[a2,b2] using its endpoints 'a', 'b', 'a2', and 'b2'.
    
    Args:
        a (float): The left endpoint of the square.
        b (float): The right endpoint of the square.
        a2 (float): The lower endpoint of the square.
        b2 (float): The upper endpoint of the square.
    """
    def __init__(self,a,b,a2,b2):
        self.box = [[a,b],[a2,b2]]
        self.center = [(a+b)/2,(a2+b2)/2]
        self.radius = (b-a)/2
        
class Square1(Square):
    """Represents a square.

    This class defines the square [-0.5PI,0.5PI]x[-0.5PI,0.5PI].
    """
    def __init__(self):
        super().__init__(-0.5*PI, 0.5*PI, -0.5*PI, 0.5*PI)
        
class UnitSquare(Square):
    """Represents a square.

    This class defines the square [0,1]x[0,1].
    """
    def __init__(self):
        super().__init__(0.0, 1.0, 0.0, 1.0)
    
class Circle:
    """Represents a circle.

    This class defines a circle using its center (a, b) and radius r.
    It also calculates a bounding box for the circle and defines a starting default
    mesh size H_start.

    Args:
        a (float): The x-coordinate of the circle's center.
        b (float): The y-coordinate of the circle's center.
        r (float): The radius of the circle.
    """
    def __init__(self,a,b,r):
        self.center = [a,b]
        self.radius = r
        eps = 0.1
        self.box = [[a-r-eps,a+r+eps],[b-r-eps,b+r+eps]]
        self.H_start = 320 # To construct more quickly mesh_ex (with FEniCS)
        
class Circle2(Circle):
    """Represents a circle.

    This class defines the circle with center (0.5, 0.5) and radius 1.
    """
    def __init__(self):
        super().__init__(0.5, 0.5, 1.0)
        
class UnitCircle(Circle):
    """Represents a circle.

    This class defines the circle with center (0, 0) and radius 1.
    It also defines a bounding box for the circle : [-1,1]x[-1,1].
    """
    def __init__(self):
        super().__init__(0.0, 0.0, 1.0)
        self.box = [[-1,1],[-1,1]]
        
class Donut:
    """Represents a donut (annulus) shape.

    This class defines a donut shape using two circles: a larger outer circle
    and a smaller inner circle (the hole). It also calculates a bounding box for
    the donut and defines a starting default mesh size H_start.

    Args:
        a (float): x-coordinate of the center of both circles.
        b (float): y-coordinate of the center of both circles.
        bigr (float): Radius of the outer circle.
        smallr (float): Radius of the inner circle (hole).
    """
    def __init__(self,a,b,bigr,smallr):
        self.bigcircle = Circle(a,b,bigr)
        self.hole = Circle(a,b,smallr)
        self.box = self.bigcircle.box
        self.H_start = 350 # To construct more quickly mesh_ex (with FEniCS)
        
class Donut1(Donut):
    """Represents a donut (annulus) shape.

    This class defines the donut shape of center (0,0) using two circles: a larger outer circle of radius 1
    and a smaller inner circle of radius 0.5 (the hole). It also defines a bounding box for the circle : [-1,1]x[-1,1].
    """
    def __init__(self):
        super().__init__(0.0, 0.0, 1.0, 0.5)
        self.box = [[-1,1],[-1,1]]
        
class Donut2(Donut):
    """Represents a donut (annulus) shape.

    This class defines the donut shape of center (0,0) using two circles: a larger outer circle of radius 1
    and a smaller inner circle of radius 0.25 (the hole). It also defines a bounding box for the circle : [-1,1]x[-1,1].
    """
    def __init__(self):
        super().__init__(0.0, 0.0, 1.0, 0.25)
        self.box = [[-1,1],[-1,1]]