import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# scipy.sparse import lil_matrix
# from scipy.sparse.linalg import eigsh
from typing import Tuple, List
import networkx as nx
# from IPython.display import HTML
# import matplotlib.animation as animation
from concurrent.futures import ThreadPoolExecutor


# Rule: the length of shorter side of the right trangle is equal to
# the length of any side of a rhombus.
def triangle(coordinate: Tuple[float, float], l_side):
    vertex = [tuple(coordinate), (coordinate[0]+l_side*np.sqrt(2),coordinate[1]), (coordinate[0]+l_side/np.sqrt(2), coordinate[1]+l_side/np.sqrt(2))]
    return vertex

def rhombus(coordinate: Tuple[float, float], l_side):
    vertex = [tuple(coordinate), 
              (coordinate[0]+l_side, coordinate[1]), 
              (coordinate[0]+l_side/np.sqrt(2)+l_side,coordinate[1]+l_side/np.sqrt(2)), 
              (coordinate[0]+l_side/np.sqrt(2),coordinate[1]+l_side/np.sqrt(2))]
    return vertex

#### To exactly match the dissected tiles with the original sized tile with matched all the markings, we need to define a function to transform the tiles.
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

### For the transformation of the triangle
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
    
    return angle, triangle2 ## The triangle2 is reflected back if it is inverted

### For the transformation of the rhombus
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
    '''The reverse of the transform_point function. 
    It is used to find the original position of the tile before transformation.'''
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

def plot_shapes(shapes, title):
    """
    Plots a list of shapes using matplotlib.
    
    Args:
    - shapes (list): A list of shapes, where each shape is a list of (x, y) coordinates.
    - title (str): The title of the plot.
    """
    fig, ax = plt.subplots()
    ax.set_title(title)
    
    for shape in shapes:
        polygon = patches.Polygon(shape, closed=True, fill=None, edgecolor='r')
        ax.add_patch(polygon)
    
    ax.set_aspect('equal')
    ax.autoscale_view()
    plt.show()

def inflate_iterate_final(tile, ls_initial, number_iterate) -> List[List[Tuple[float, float]]]:
    kk = 0
    vertice_tri = triangle((0,0), ls_initial/(1+np.sqrt(2)))
    vertice_rhom = rhombus((0,0), ls_initial/(1+np.sqrt(2)))
    
    while kk < number_iterate:
        kk += 1
        #print(f"Starting iteration {kk}")

        if isinstance(tile[0], tuple):
            tile = [tile]
        
        inflate_shapes = [([(x*(1+np.sqrt(2))*ls_initial, y*(1+np.sqrt(2))*ls_initial) for x, y in vertices]) for vertices in tile]
        all_tiles = []

        for i, j in enumerate(inflate_shapes):
            #print(f"Processing tile: {j}")  # Debug print
            if len(j) == 3:
                angletri, triangle_c = compute_rotation_and_scale_tri(vertice_tri, j)
                newtri = [reverse_transform_point(p, angletri, 0, 0, triangle_c[0]) for p in j]
                if is_inverted_tri(j):
                    inflatetri = inflate_triangle(newtri[1], ls_initial)
                    inflatetri1 = [[transform_point(p, angletri, 0, 0, j[0], True) for p in shape] for shape_list in inflatetri for shape in shape_list]
                else:
                    inflatetri = inflate_triangle(newtri[0], ls_initial)
                    inflatetri1 = [[transform_point(p, angletri, 0, 0, j[0]) for p in shape] for shape_list in inflatetri for shape in shape_list]
                #print("just after each inflation-triangle(5)",inflatetri)
                #print(len(inflatetri))
                all_tiles.extend(inflatetri1)
            elif len(j) == 4:
                angler, _ = compute_rotation_and_scale_rhombus(vertice_rhom, j)
                newr = [reverse_transform_point(p, angler, 0, 0, j[0]) for p in j]
                inflater = inflate_rhombus(newr[0], ls_initial)
                #print("just after each inflation-rhombus(7)", inflater)
                #print(len(inflater))
                inflater1 = [[transform_point(p, angler, 0, 0, j[0]) for p in shape] for shape_list in inflater for shape in shape_list]
                all_tiles.extend(inflater1)
        
        # Set the tile for the next iteration to the result of the current iteration
        tile = all_tiles
        print("all tiles in each iteration are", tile)
    
    return all_tiles

def inflate_triangle_iterate_final(tile, ls_initial, number_iterate) -> List[List[Tuple[float, float]]]:
    kk = 0
    vertice_tri = triangle((0,0), ls_initial/(1+np.sqrt(2)))
    vertice_rhom = rhombus((0,0), ls_initial/(1+np.sqrt(2)))
    
    while kk < number_iterate:
        kk += 1
        #print(f"Starting iteration {kk}")

        if isinstance(tile[0], tuple):
            tile = [tile]
        
        inflate_shapes = [([(x*(1+np.sqrt(2))*ls_initial, y*(1+np.sqrt(2))*ls_initial) for x, y in vertices]) for vertices in tile]
        all_tiles = []

        for i, j in enumerate(inflate_shapes):
            #print(f"Processing tile: {j}")
            #plot_shapes([j], f"Tile before inflation: {i}")
            
            if len(j) == 3:
                angletri, triangle_c = compute_rotation_and_scale_tri(vertice_tri, j)
                #print(angletri)
                newtri = [reverse_transform_point(p, angletri, 0, 0, triangle_c[0]) for p in j]
                #plot_shapes([newtri], f"New Triangle: {i}")
                #print(newtri[0])
                if is_inverted_tri(j):
                    inflatetri = inflate_triangle(newtri[1], ls_initial)
                else:
                    inflatetri = inflate_triangle(newtri[0], ls_initial)
                    
                inflatetri_shapes = [shape[0] for shape in inflatetri]  # Extract the shapes
                #plot_shapes(inflatetri_shapes, f"Inflated Triangle: {i}")
                if is_inverted_tri(j):
                    inflatetri1 = [[transform_point(p, angletri, 0, 0, j[0], True) for p in shape] for shape in inflatetri_shapes]
                else:
                    inflatetri1 = [[transform_point(p, angletri, 0, 0, j[0]) for p in shape] for shape in inflatetri_shapes]
                #plot_shapes(inflatetri1, f"Transformed Inflated Triangle: {i}")
                
                all_tiles.extend(inflatetri1)
            elif len(j) == 4:
                angler, _ = compute_rotation_and_scale_rhombus(vertice_rhom, j)
                newr = [reverse_transform_point(p, angler, 0, 0, j[0]) for p in j]
                #plot_shapes([newr], f"New Rhombus: {i}")
                
                inflater = inflate_rhombus(newr[0], ls_initial)
                inflater_shapes = [shape[0] for shape in inflater]  # Extract the shapes
                #plot_shapes(inflater_shapes, f"Inflated Rhombus: {i}")
                
                inflater1 = [[transform_point(p, angler, 0, 0, j[0]) for p in shape] for shape in inflater_shapes]
                #plot_shapes(inflater1, f"Transformed Inflated Rhombus: {i}")
                
                all_tiles.extend(inflater1)
        
        plot_shapes(all_tiles, f"All tiles after iteration {kk}")
        tile = all_tiles
    
    return all_tiles


### Create the QCs tight-binding Hamiltonian with constant nearest neighbour hopping.
def vertices_coordinates(tiles, precision=12):
    # Flatten the list of lists of tuples
    flattened = [coordinate for sublist in tiles for coordinate in sublist]
    # Round the coordinates to the specified precision to avoid floating-point precision issues
    rounded = [(round(x, precision), round(y, precision)) for x, y in flattened]
    # Remove duplicates by converting the list to a set, then back to a list
    unique = list(set(rounded))
    # Sort the list of tuples if needed
    unique.sort(key=lambda x: (x[0], x[1]))
    # Convert the list to a numpy array
    unique_array = np.array(unique)
    return unique_array

def distance(coordinate_a: Tuple[float, float], coordinate_b: Tuple[float, float]):
    x_a, y_a = coordinate_a
    x_b, y_b = coordinate_b
    return ((x_b - x_a) ** 2 + (y_b - y_a) ** 2) ** 0.5

def hamiltonian(hopping_parameter, all_tiles, hopping_distance_max, hopping_distance_min, E_onsite):
    h = np.zeros([len(all_tiles), len(all_tiles)])
    np.fill_diagonal(h, E_onsite)
    for i, coordinate_1 in enumerate(all_tiles):
        for j, coordinate_2 in enumerate(all_tiles):
            if i != j:  # Ensure we're not calculating the distance from a point to itself
                d = distance(coordinate_1, coordinate_2)
                if hopping_distance_min < d <= hopping_distance_max:
                    h[i][j] = hopping_parameter
    return h

def plot_connectivity(all_tiles, hopping_parameter, hopping_distance_min, hopping_distance_max, E_onsite, save_path=None):
    # Create the Hamiltonian matrix
    h = hamiltonian(hopping_parameter, all_tiles, hopping_distance_max, hopping_distance_min, E_onsite)
    
    # Create a graph object
    G = nx.Graph()

    # Add nodes to the graph
    for i, coordinate in enumerate(all_tiles):
        G.add_node(i, pos=coordinate)

    # Add edges to the graph based on the Hamiltonian
    for i, coordinate_1 in enumerate(all_tiles):
        for j, coordinate_2 in enumerate(all_tiles):
            if i != j:  # Ensure we're not calculating the distance from a point to itself
                d = distance(coordinate_1, coordinate_2)
                if hopping_distance_min < d <= hopping_distance_max:
                    if h[i][j] != 0:
                        G.add_edge(i, j)
    # Create a figure before drawing the graph
    plt.figure()
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=False, node_size=10)
    # Adjust the plot
    plt.margins(0.1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x')
    plt.ylabel('y')
    # Show the plot
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()
    
def compute_mu(args):
    x, y, a, b, theta = args
    u = x * np.cos(theta) - y * np.sin(theta)
    v = x * np.sin(theta) + y * np.cos(theta)
    phi1 = np.pi * (a * np.cos(theta) - b * np.sin(theta))
    phi2 = np.pi * (a * np.sin(theta) + b * np.cos(theta))
    return np.sin(np.pi * u - phi1) ** 2 + np.sin(np.pi * v - phi2) ** 2

def quasip_continuum(x, y, a, b, theta):
    '''Here the inputs x and y are arrays of x and y coordinates'''
    lx = len(x)
    ly = len(y)
    mu = np.zeros((lx, ly), dtype=float)
    
    # Create a list of arguments for each combination of x[i] and y[j]
    args_list = [(x[i], y[j], a, b, theta) for i in range(lx) for j in range(ly)]
    
    # Use ThreadPoolExecutor to parallelize the computation, limiting the number of threads
    max_threads = min(32, len(args_list))  # Adjust the number of threads as needed
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        results = list(executor.map(compute_mu, args_list))
    
    # Reshape the results back into the mu matrix
    for index, value in enumerate(results):
        i = index % lx
        j = index // lx
        mu[i, j] = value
    return mu

