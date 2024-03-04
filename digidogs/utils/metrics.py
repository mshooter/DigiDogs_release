"""
From: https://github.com/anibali/margipose/blob/2933f30203b3cd5c636917a7c9ff107d02434598/src/margipose/dsntnn.py#L133
"""
import numpy as np
import torch
from scipy.spatial import procrustes

def batch_compute_similarity_transform_torch(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    S1 = S1.to(torch.float32)
    S2 = S2.to(torch.float32)

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)

    return S1_hat

def rigid_transform_3D(A, B):
    centroid_A = np.mean(A, axis = 0)
    centroid_B = np.mean(B, axis = 0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B)
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))
    t = -np.dot(R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return R, t

def rigid_align(A, B):
    R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(R, np.transpose(A))) + t
    return A2

def apply_alignment(pred_skel, target_skel):
    """
    Need to apply to 1 sample
    """
    pred_skel = pred_skel.detach().cpu().numpy()
    target_skel = target_skel.detach().cpu().numpy()
    mtx1, mtx2, _ = procrustes(target_skel, pred_skel)
    #mean = np.mean(target_skel,0)
    #std = np.linalg.norm(target_skel - mean)
    #aligned = (mtx2 * std) + mean
    return mtx1, mtx2

def metrics2d(pred,target,included_joints, threshold = 0.15, norm=None, is_3d=False):
    """
    pred [B, K, 2/3]
    target [B, K, 2/3]
    included joints [B,K,1]
    norm [B,1]
    """
    assert pred.size() == target.size(), 'x and y not the same shape'
    assert len(included_joints.shape) == 2, 'size of visibility is not correct' 
    if len(norm.shape) != 2: 
        norm = norm.reshape(norm.shape[0],1)
    assert len(norm.shape) == 2, 'size of norm is not correct' 
    # we take the face out..
    total_joints = torch.sum(included_joints,dim=1)
    errors = torch.sqrt(torch.sum(((pred-target)**2),dim=-1))[:,:26] 
    print(errors, norm)
    errors = errors / (norm)

    mpjpe = errors * included_joints
    mpjpe = torch.sum(mpjpe,dim=1) / (total_joints)
    errors = (errors < threshold)* included_joints
    errors = torch.sum(errors, dim=1) 
    pck = errors / (total_joints) 
    return pck, mpjpe

if __name__ == "__main__":
    x = torch.rand((4,3,2))
    y = torch.rand((4,3,2))
    v = torch.randint(2,(4,3)).float()
    norm = torch.rand((4,1))
    print(v)
    pck2d(x,y,v,norm=norm)
