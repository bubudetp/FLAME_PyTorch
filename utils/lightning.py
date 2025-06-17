import torch

def estimate_lightning(normals, image, mask, reg_weight=0.01):
    """
    Estimate spherical harmonics lighting coefficients using regularized least squares.
    image: (H, W, 3) – linear RGB image
    normals: (H, W, 3) – surface normals
    mask: (H, W) – binary mask of reliable face region (face & ellipse)
    reg_weight: regularization lambda
    """
    def compute_sh_basis(normals):
        x, y, z = normals[..., 0], normals[..., 1], normals[..., 2]
        sh = torch.stack([
            torch.ones_like(x),       # l=0
            x, y, z,                  # l=1
            x * y, x * z, y * z,      # l=2
            x**2 - y**2,
            3 * z**2 - 1              # l=2
        ], dim=-1)  # [N, 9]
        return sh

    image_flat = image[mask].reshape(-1, 3)        # [N, 3]
    normals_flat = normals[mask].reshape(-1, 3)    # [N, 3]
    sh_basis = compute_sh_basis(normals_flat)      # [N, 9]

    # Regularized least squares: L = (AᵀA + λI)⁻¹ AᵀB
    A = sh_basis
    B = image_flat
    I = torch.eye(A.shape[1], device=A.device)
    AtA = A.T @ A
    AtB = A.T @ B
    L = torch.linalg.lstsq(sh_basis, image_flat).solution  # [9, 3]
    
    return L

def compute_shading(normals, lighting):
    """
    Generate shading image from normals and spherical harmonics coefficients.
    normals: (H, W, 3)
    lighting: (9, 3)
    Returns: shading in linear space [H, W, 3]
    """
    def compute_sh_basis(normals):
        x, y, z = normals[..., 0], normals[..., 1], normals[..., 2]
        sh = torch.stack([
            torch.ones_like(x),
            x, y, z,
            x * y, x * z, y * z,
            x**2 - y**2,
            3 * z**2 - 1
        ], dim=-1)  # [H, W, 9]
        return sh

    sh_basis = compute_sh_basis(normals)  # [H, W, 9]
    shading = torch.einsum('hwk,kc->hwc', sh_basis, lighting)  # [H, W, 3]
    return torch.clamp(shading, 0.0, 1.0)

def apply_elliptical_mask(H, W, device="cpu"):
    """
    Generate elliptical mask covering central facial region.
    """
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    cx, cy = W // 2, H // 2
    norm_x = ((x - cx) / (0.4 * W)) ** 2
    norm_y = ((y - cy) / (0.5 * H)) ** 2
    ellipse_mask = (norm_x + norm_y) < 1.0
    return ellipse_mask
