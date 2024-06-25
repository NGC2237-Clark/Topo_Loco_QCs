import numpy as np
from typing import Tuple, List

class quasicrystal:

    def __init__(self) -> None:
        pass

    def triangle(coordinate: Tuple[float, float], l_side):
        vertex = [tuple(coordinate), (coordinate[0]+l_side*np.sqrt(2),coordinate[1]), (coordinate[0]+l_side/np.sqrt(2), coordinate[1]+l_side/np.sqrt(2))]
        return vertex

    def rhombus(coordinate: Tuple[float, float], l_side):
        vertex = [tuple(coordinate), 
                (coordinate[0]+l_side, coordinate[1]), 
                (coordinate[0]+l_side/np.sqrt(2)+l_side,coordinate[1]+l_side/np.sqrt(2)), 
                (coordinate[0]+l_side/np.sqrt(2),coordinate[1]+l_side/np.sqrt(2))]
        return vertex
    
    def transform_point_ini(vertices, theta, trans_x, trans_y, origin=(0,0), invert_y=False):
        # Convert tuple to list for modification
    vertices = list(vertices)
    #print(origin)
    # Translate the point to the origin
    vertices[0] -= origin[0]
    vertices[1] -= origin[1]

    # Inversion or not?
    if invert_y:
        vertices[0] = -vertices[0]

    # Rotation matrix
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotated_x = cos_theta * vertices[0] - sin_theta * vertices[1]
    rotated_y = sin_theta * vertices[0] + cos_theta * vertices[1]

    # Translate the rotated point
    translated_x = rotated_x + trans_x + origin[0]
    translated_y = rotated_y + trans_y + origin[1]
    
    return (translated_x, translated_y)