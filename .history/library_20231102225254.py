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
    
    def transform_point(vertices, theta, trans_x, trans_y, origin=(0,0), invert_y=False):
        # Convert tuple to list for modification
        vertices = list(vertices)
        #print(origin)
        # Translate the point to the origin
        #vertices[0] -= origin[0]
        #vertices[1] -= origin[1]

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
    

    def reverse_transform_point(vertices, theta, trans_x, trans_y, origin=(0,0), invert_y=False):
        # Convert tuple to list for modification
        vertices = list(vertices)
        #print(vertices)
        #print(origin)
        # Translate the point to the origin
        vertices[0] -= origin[0]
        vertices[1] -= origin[1]

        # Reverse the translation
        vertices[0] -= trans_x
        vertices[1] -= trans_y
        #print(vertices)
        # Reverse the rotation
        cos_theta = np.cos(-theta)
        sin_theta = np.sin(-theta)
        rotated_x = cos_theta * vertices[0] - sin_theta * vertices[1]
        rotated_y = sin_theta * vertices[0] + cos_theta * vertices[1]

        # Inversion or not?
        if invert_y:
            rotated_x = -rotated_x

        # Translate back to the original position
        #translated_x = rotated_x + origin[0]
        translated_x = rotated_x
        #translated_y = rotated_y + origin[1]
        translated_y = rotated_y
        
        return (translated_x, translated_y)
    
    def is_inverted_tri(triangle_vertices):
        x1, y1 = triangle_vertices[0]
        x2, y2 = triangle_vertices[1]
        x3, y3 = triangle_vertices[2]

        area = 0.5 * (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

        return area < 0

    def compute_rotation_and_scale_tri(triangle1, triangle2):
        # Check if triangle2 is inverted
        if is_inverted_tri(triangle2):
            # Invert triangle2 back
            triangle2 = [triangle2[1], triangle2[0], triangle2[2]]
        
        # Compute vectors for the sides of the triangles
        vec1_triangle1 = np.array(triangle1[1]) - np.array(triangle1[0])
        vec1_triangle2 = np.array(triangle2[1]) - np.array(triangle2[0])
        
        # Normalize the vectors
        vec1_triangle1 = vec1_triangle1 / np.linalg.norm(vec1_triangle1)
        vec1_triangle2 = vec1_triangle2 / np.linalg.norm(vec1_triangle2)
        
        # Compute the dot product and cross product of the normalized vectors
        dot_product = np.dot(vec1_triangle1, vec1_triangle2)
        cross_product = np.cross(vec1_triangle1, vec1_triangle2)
        
        # Compute the angle between the vectors using the dot and cross products
        angle = np.arctan2(cross_product, dot_product)
        
        return angle, triangle2
    
    def is_inverted_r(vertices):
        n = len(vertices)
        area = 0
        for i in range(n):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % n]
            area += (x1*y2 - x2*y1)
        return area < 0

    def compute_rotation_and_scale_rhombus(rhombus1, rhombus2):
        # Check if rhombus2 is inverted
        if is_inverted_r(rhombus2):
            # Invert rhombus2 back
            rhombus2 = [rhombus2[0], rhombus2[3], rhombus2[2], rhombus2[1]]
        
        # Compute vectors for the sides of the rhombuses
        vec1_rhombus1 = np.array(rhombus1[1]) - np.array(rhombus1[0])
        vec1_rhombus2 = np.array(rhombus2[1]) - np.array(rhombus2[0])
        
        # Normalize the vectors
        vec1_rhombus1 = vec1_rhombus1 / np.linalg.norm(vec1_rhombus1)
        vec1_rhombus2 = vec1_rhombus2 / np.linalg.norm(vec1_rhombus2)
        
        # Compute the dot product and cross product of the normalized vectors
        dot_product = np.dot(vec1_rhombus1, vec1_rhombus2)
        cross_product = np.cross(vec1_rhombus1, vec1_rhombus2)
        
        # Compute the angle between the vectors using the dot and cross products
        angle = np.arctan2(cross_product, dot_product)
        
        # Compute the scaling factor
        scale_factor = np.linalg.norm(np.array(rhombus2[1]) - np.array(rhombus2[0])) / np.linalg.norm(np.array(rhombus1[1]) - np.array(rhombus1[0]))
        
        return angle, scale_factor
    
    def inflate_triangle(coordinate: Tuple[float, float], l_side)-> List[Tuple[str, List[Tuple[float, float]]]]:
        """according to arXiv:math/0203252, we dissect the triangle into 3 triangles and two rhombus with side length = l_side"""
        vertice_tri = triangle(coordinate, l_side/(1+np.sqrt(2)))
        vertice_rhom = rhombus(coordinate, l_side/(1+np.sqrt(2)))
        #vert_enlarge = [((1+np.sqrt(2))*x, (1+np.sqrt(2))*y) for x, y in vertice_tri] 
        tri1 = [transform_point_ini(i, 5*np.pi/4, l_side/(1+np.sqrt(2)), l_side/(1+np.sqrt(2)), origin=coordinate) for i in vertice_tri]
        tri2 = [transform_point_ini(x, 0, (1+np.sqrt(2))*l_side/(1+np.sqrt(2)), 0, origin=coordinate, invert_y=True) for x in vertice_tri]
        tri3 = [transform_point_ini(y, 3*np.pi/4, (2+np.sqrt(2)/2)*l_side/(1+np.sqrt(2)), (np.sqrt(2)/2)*l_side/(1+np.sqrt(2)), origin=coordinate) for y in vertice_tri]
        r1 = [transform_point_ini(i, 5*np.pi/4, (1+np.sqrt(2)/2)*l_side/(1+np.sqrt(2)), (1+np.sqrt(2)/2)*l_side/(1+np.sqrt(2)), origin=coordinate) for i in vertice_rhom]
        r2 = [transform_point_ini(x, 3*np.pi/4, (2+np.sqrt(2))*l_side/(1+np.sqrt(2)), 0, origin=coordinate) for x in vertice_rhom]
        return [[tri1], [tri2], [tri3], [r1] ,[r2]]
    
    def inflate_rhombus(coordinate: Tuple[float, float], l_side) -> List[Tuple[str, List[Tuple[float, float]]]]:
        """according to arXiv:math/0203252, we dissect the rhombus into 4 triangles and 3 rhombus with side length = l_side"""
        vertice_tri = triangle(coordinate, l_side/(1+np.sqrt(2)))
        vertice_rhom = rhombus(coordinate, l_side/(1+np.sqrt(2)))
        tri1 = [transform_point_ini(i, 0, l_side/(1+np.sqrt(2)), 0) for i in vertice_tri]
        tri2 = [transform_point_ini(i, np.pi, (1+3*np.sqrt(2)/2)*l_side/(1+np.sqrt(2)), (1+np.sqrt(2)/2)*l_side/(1+np.sqrt(2)), origin=coordinate) for i in vertice_tri]
        tri3 = [transform_point_ini(i, np.pi/4, (2+np.sqrt(2))*l_side/(1+np.sqrt(2)), l_side/(1+np.sqrt(2)), origin=coordinate, invert_y=True) for i in vertice_tri]
        tri4 = [transform_point_ini(i, 5*np.pi/4, (np.sqrt(2)/2)*l_side/(1+np.sqrt(2)), (np.sqrt(2)/2)*l_side/(1+np.sqrt(2)), origin=coordinate, invert_y=True) for i in vertice_tri]
        r1 = [transform_point_ini(i, 0, 0, 0, origin=coordinate) for i in vertice_rhom]
        r2 = [transform_point_ini(i, np.pi/2, (1+np.sqrt(2))*l_side/(1+np.sqrt(2)), 0, origin=coordinate) for i in vertice_rhom]
        r3 = [transform_point_ini(i, 0, (1+np.sqrt(2))*l_side/(1+np.sqrt(2)), l_side/(1+np.sqrt(2)), origin=coordinate) for i in vertice_rhom]
        return [[tri1], [tri2], [tri3], [tri4], [r1], [r2], [r3]]