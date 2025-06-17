from pytorch3d.structures import Meshes
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, PerspectiveCameras

def rasterize_mesh(vertices, faces, image_size=512):
    """
    Rasterize mesh to get per-pixel face index and z-buffer.
    """
    device = vertices.device
    mesh = Meshes(verts=[vertices], faces=[faces])
    cameras = PerspectiveCameras(device=device)
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(mesh)
    return fragments.pix_to_face, fragments.zbuf