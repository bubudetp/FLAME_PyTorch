import torch
import numpy as np
import cv2
from flame_pytorch.flame import FLAME
from flame_pytorch.config import get_config
from utils.lightning import estimate_lightning, compute_shading
from utils.rasterizer import rasterize_mesh

import os
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings, MeshRasterizer,
    PerspectiveCameras
)

from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    BlendParams,
    PointLights,
    SoftPhongShader,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform
)

def render_mesh_overlay(mesh, image_tensor):
    """
    Renders the mesh onto the image using PyTorch3D.
    mesh: PyTorch3D Meshes
    image_tensor: [H, W, 3] float32 torch Tensor, RGB in [0, 1]
    """
    H, W = image_tensor.shape[:2]
    device = image_tensor.device

    # 🔧 Camera positioned to look at mesh center (FLAME is centered at origin)
    R, T = look_at_view_transform(dist=2.0, elev=0.0, azim=0.0)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    blend_params = BlendParams(background_color=(0.0, 0.0, 0.0))

    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=1
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights, blend_params=blend_params)
    )

    # Dummy color texture to make mesh visible
    verts_rgb = torch.ones_like(mesh.verts_padded())  # white
    mesh.textures = TexturesVertex(verts_features=verts_rgb)

    rendered = renderer(mesh)[0, ..., :3]  # (H, W, 3)

    # Alpha composite with background image
    overlay = 0.5 * rendered + 0.5 * image_tensor
    overlay = (overlay.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    cv2.imwrite("mesh_overlay.png", overlay[..., ::-1])  # RGB to BGR
    print("Saved overlay to mesh_overlay.png")


def compute_vertex_normals(vertices, faces):
    v0 = vertices[:, faces[:, 0]]
    v1 = vertices[:, faces[:, 1]]
    v2 = vertices[:, faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0)
    face_normals = torch.nn.functional.normalize(face_normals, dim=-1)

    vertex_normals = torch.zeros_like(vertices)
    for i in range(faces.shape[0]):
        for j in range(3):
            vertex_normals[:, faces[i, j]] += face_normals[:, i]
    vertex_normals = torch.nn.functional.normalize(vertex_normals, dim=-1)
    return vertex_normals

def apply_elliptical_mask(H, W, device="cpu"):
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    cx, cy = W // 2, H // 2
    norm_x = ((x - cx) / (0.4 * W)) ** 2
    norm_y = ((y - cy) / (0.5 * H)) ** 2
    ellipse_mask = (norm_x + norm_y) < 1.0
    return ellipse_mask

def main():
    image_path = "image.png"
    image = cv2.imread(image_path)[..., ::-1] / 255.0  # BGR to RGB, [0,1]
    image = torch.from_numpy(image).float()
    H, W, _ = image.shape

    # Gamma correction to linear space
    image_linear = image ** 2.2

    config = get_config()
    config.batch_size = 1
    flame = FLAME(config)

    # Generate mesh
    vertices, landmarks = flame(
        shape_params=torch.zeros(1, config.shape_params),
        expression_params=torch.zeros(1, config.expression_params),
        pose_params=torch.zeros(1, config.pose_params)
    )  # vertices: [1, V, 3]

    vertices = vertices[0]
    faces = flame.faces_tensor  # [F, 3]

    # Compute normals
    normals = compute_vertex_normals(vertices.unsqueeze(0), faces)[0]

    # Create rasterizer
    mesh = Meshes(verts=[vertices], faces=[faces])
    render_mesh_overlay(mesh, image)  # image must still be in [0,1] RGB float
    cameras = PerspectiveCameras(device=vertices.device)
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(mesh)

    # Mesh visibility mask
    mesh_mask = fragments.pix_to_face[0, ..., 0] >= 0

    # Elliptical mask
    ellipse_mask = apply_elliptical_mask(H, W, device=image.device)
    mask = mesh_mask & ellipse_mask  # Combined reliable region
    cv2.imwrite("mask.png", mask.cpu().numpy().astype(np.uint8) * 255)

    # Normals to image-space
    normals_img = torch.zeros((H, W, 3), dtype=torch.float32)
    zbuf = fragments.zbuf[0, ..., 0]
    bary_coords = fragments.bary_coords[0, ..., 0]
    pix_to_face = fragments.pix_to_face[0, ..., 0]

    for y in range(H):
        for x in range(W):
            fid = pix_to_face[y, x]
            if fid < 0:
                continue
            inds = faces[fid]
            bary = bary_coords[y, x]
            if bary.shape[0] < 3:
                continue
            normal = bary[0] * normals[inds[0]] + bary[1] * normals[inds[1]] + bary[2] * normals[inds[2]]
            normals_img[y, x] = torch.nn.functional.normalize(normal, dim=0)
    
    normals_vis = (normals_img + 1.0) / 2.0  # from [-1, 1] to [0, 1]
    cv2.imwrite("normals_x.png", (normals_vis[..., 0].numpy() * 255).astype(np.uint8))
    cv2.imwrite("normals_y.png", (normals_vis[..., 1].numpy() * 255).astype(np.uint8))
    cv2.imwrite("normals_z.png", (normals_vis[..., 2].numpy() * 255).astype(np.uint8))


    print("mesh_mask pixels:", mesh_mask.sum().item())
    print("ellipse_mask pixels:", ellipse_mask.sum().item())
    print("combined mask pixels:", mask.sum().item())
    # Estimate lighting from linear image
    lighting = estimate_lightning(normals_img, image_linear, mask)

    # Compute shading
    shading = compute_shading(normals_img, lighting)

    # Compute albedo in linear space
    albedo_linear = image_linear / (shading + 1e-6)
    cv2.imwrite("shading_debug.png", (shading.numpy() * 255).astype(np.uint8)[..., ::-1])

    albedo_linear = torch.clamp(albedo_linear, 0.0, 1.0)

    # Convert back to gamma-corrected sRGB
    albedo = albedo_linear ** (1 / 2.2)
    albedo_np = (albedo.numpy() * 255).astype(np.uint8)

    # Save
    out_path = "albedo_map.png"
    cv2.imwrite(out_path, albedo_np[..., ::-1])  # RGB to BGR
    print(f"Saved albedo map to {out_path}")


if __name__ == "__main__":
    main()