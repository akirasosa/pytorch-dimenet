import torch
import torch.nn.functional as F


def calculate_interatomic_distances(R, idx_i, idx_j):
    Ri = R[idx_i]
    Rj = R[idx_j]
    # ReLU prevents negative numbers in sqrt
    Dij = torch.sqrt(F.relu(torch.sum((Ri - Rj) ** 2, -1)))
    return Dij


def calculate_neighbor_angles(R, id3_i, id3_j, id3_k):
    """Calculate angles for neighboring atom triplets"""
    Ri = R[id3_i]
    Rj = R[id3_j]
    Rk = R[id3_k]
    R1 = Rj - Ri
    R2 = Rk - Ri
    x = torch.sum(R1 * R2, dim=-1)
    y = torch.cross(R1, R2)
    y = torch.norm(y, dim=-1)
    angle = torch.atan2(y, x)
    return angle
