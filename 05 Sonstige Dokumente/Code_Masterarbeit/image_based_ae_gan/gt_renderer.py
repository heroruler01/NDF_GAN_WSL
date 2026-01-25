import open3d as op
import numpy as np
from matplotlib import pyplot as plt
import copy

def view_render(objpath):

    """
    arguments:
    - filepath
    - mesh rotation parameters (np.pi/4 = 45 Grad)
    :return: numpy array: 3 channel image
    """
    current_view = [0, -np.pi/2, 0]
    rot_x = current_view[0]
    rot_y = current_view[1]
    rot_z = current_view[2]

    vis = op.visualization.Visualizer()
    #vis.create_window() # we need window instance on server --> specify DISPLAY
    vis.create_window(width=128, height=128, visible=True)

    # read mesh and compute normals
    mesh = op.io.read_triangle_mesh(objpath)
    mesh.compute_vertex_normals()

    # create a copy of the mesh
    mesh_copy = copy.deepcopy(mesh)

    # rotate mesh
    mesh_copy.rotate(mesh.get_rotation_matrix_from_xyz((rot_x, rot_y, rot_z)))

    vis.add_geometry(mesh_copy)
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()

    # Buffer takes the image as np array
    buffer = vis.capture_screen_float_buffer(True)
    #print(np.max(buffer), np.min(buffer))
    #vis.capture_screen_image("test.png", do_render=True)

    # convert buffer data to numpy array
    image = np.asarray(buffer)
    #print(image)
    #mean = np.mean(image)
    #median = np.median(image)
    # print(buffer)
    #plt.imshow(image, interpolation='nearest')
    #plt.show()
    return image
