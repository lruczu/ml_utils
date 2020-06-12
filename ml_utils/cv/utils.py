from typing import List, Tuple, Optional

from matplotlib.patches import Rectangle as MatplotlibRectangle
import matplotlib.pyplot as plt
import numpy as np


class Point:
    """
        remember:
            x ---->  
    """
    def __init__(
        self, 
        x: float,
        y: float,
    ):
        """
        Args:
            x: normalized to [0, 1]
            y: normalized to [0, 1]
        """
        self.x = x
        self.y = y

    def __repr__(self):
        return f"(x={self.x}, y={self.y})"


class Rectangle(Point):
    
    def __init__(
        self,
        x: float, 
        y: float, 
        width: float, 
        height: float,
    ):
        """
        Args:
            x: normalized to [0, 1], corresponds to upper lower point (if rectangle)
            y: normalized to [0, 1], corresponds to upper lower point (if rectangle)
            width: normalized to [0, 1]
            height normalized to [0, 1]
        """
        super().__init__(x, y)
        self.width = width
        self.height = height

    @property
    def area(self) -> float:
        return self.width * self.height

    def get_intersection_area(self, other: "Rectangle") -> float:
        if self._is_inside(other):
            return (self.x + self.width - other.x) * (self.y + self.height - other.y)
        elif other._is_inside(self):
            return (other.x + other.width - self.x) * (other.y + other.height - self.y)

        return 0

    def _is_inside(self, other: Point) -> bool:
        """If given point is inside rectangle.
        
        Args:
            other: treated as point (width and height are ignored)
        """
        if not self.x < other.x < self.x + self.width:
            return False

        if not self.y < other.y < self.y + self.height:
            return False

        return True

    def __repr__(self):
        return f"(x={self.x}, y={self.y}, w={self.width}, h={self.height})"



def draw_image_with_rectangle(
    image: np.ndarray,
    rectangles: List[Rectangle], 
    figsize: Tuple[int, int]=(10, 10)
):
    """
    Args:
        image: 3-dim array (N, N, 3)
        rectangles:
        figsize:
    """
    h, w, _ = image.shape

    _, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    for r in rectangles:
        rectangle = MatplotlibRectangle(
            (r.x * w, r.y * h),
            r.width * w,
            r.height * h, 
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rectangle)
    plt.show()


def iou(r1: Point, r2: Point):
    """
    Args:
        p1: treated as Rectangle
        p2: 
    """
    if not (r1.width * r1.height > 0 or r2.width * r2.height):
        raise ValueError('At least one rectangle must be nonempty.')

    area_overlap = r1.get_intersection_area(r2)
    if area_overlap == 0:
        return 0

    area_union = r1.area + r2.area - 2 * area_overlap

    return area_overlap / area_union
