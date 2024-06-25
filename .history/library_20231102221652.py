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
