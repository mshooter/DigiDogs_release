import torch
from digidogs.configs.defaults import F_PLANE, N_PLANE

def normalise_skeleton(pts, cx, cy, n_w, n_h, focal_length, fy=None, epsilon=1e-8, scale=1):
    """
    Normalise skeleton of 3D camera points
    Params:
        pts (tensor) : 3d camera points (K, 3)
        cx, cy (int) : centre point camera
        n_w, n_h (float) : width, height
        focal_length (int) : focal lenght in pixels
    Return: 
        normalised points
    """
    if fy is not None:
        focal_lengthx = focal_length
        focal_lengthy = fy
    else:
        focal_lengthx = focal_length
        focal_lengthy = focal_length
    curr_x = pts[:,0]
    curr_y = pts[:,1]
    curr_z = pts[:,2]
    term1 = F_PLANE - N_PLANE
    term2 = (F_PLANE+N_PLANE)
    term3 = (-2 * F_PLANE * N_PLANE)
    norm_x3d = (focal_lengthx * curr_x / (curr_z+epsilon) + cx) *scale  / n_w
    norm_y3d = (focal_lengthy * curr_y / (curr_z+epsilon) + cy) *scale  / n_h
    norm_z3d = term2/term1+(1/(curr_z+epsilon))*term3/term1

    # normalise between -1,1
    norm_x3d = 2*norm_x3d - 1
    norm_y3d = 2*norm_y3d - 1

    return norm_x3d, norm_y3d, norm_z3d

def denormalise_skeleton(n_pts, cx, cy, n_w, n_h):
    curr_x = n_pts[:,0]
    curr_y = n_pts[:,1]
    curr_z = n_pts[:,2]

    term1 = F_PLANE - N_PLANE
    term2 = (F_PLANE+N_PLANE)
    term3 = (-2 * F_PLANE * N_PLANE)

    # normalise to -1,1
    x3d = (curr_x+1)/2
    y3d = (curr_y+1)/2
    x3d = x3d * n_w
    y3d = y3d * n_h
    z3d = term3/(term1*(curr_z-term2/term1))
    return x3d, y3d, z3d
