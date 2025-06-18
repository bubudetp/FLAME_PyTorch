import cv2
import torch
import numpy as np
from tqdm import trange
import face_alignment
from flame_pytorch.flame import FLAME
from flame_pytorch.config import get_config
from pytorch3d.io import save_obj

# ---------- 1. Detect and Resize to 224x224 ----------
def detect_2d_landmarks(image_path, device):
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device='cuda' if device.type == 'cuda' else 'cpu',
        flip_input=False
    )

    img = cv2.imread(image_path)[..., ::-1]  # BGR to RGB
    original_h, original_w = img.shape[:2]

    img_resized = cv2.resize(img, (224, 224))
    scale_x = 224.0 / original_w
    scale_y = 224.0 / original_h

    landmarks = fa.get_landmarks(img_resized)
    if landmarks is None or len(landmarks) == 0:
        raise RuntimeError("No face detected")
    landmarks = landmarks[0]

    # Save resized image (optional)
    cv2.imwrite("aligned_input.png", img_resized[..., ::-1])

    return torch.tensor(landmarks, dtype=torch.float32, device=device).unsqueeze(0)  # [1, 68, 2]

# ---------- 2. Load FLAME model ----------
def get_flame_model(device):
    config = get_config()
    config.batch_size = 1
    return FLAME(config).to(device), config

# ---------- 3. Project 3D points to 2D ----------
def project_vertices(vertices, camera_params, image_size):
    fov = 5.0
    focal = 0.5 * image_size / np.tan(0.5 * fov * np.pi / 180.0)
    cam_t = camera_params[:, :3]  # [B, 3]

    projected = vertices + cam_t.unsqueeze(1)
    x = projected[:, :, 0] * focal / projected[:, :, 2] + image_size / 2
    y = -projected[:, :, 1] * focal / projected[:, :, 2] + image_size / 2
    return torch.stack([x, y], dim=-1)  # [B, N, 2]

# ---------- 4. Optimize FLAME parameters ----------
def optimize_flame(image_path, flame, config, device):
    image_size = 224
    target_2d = detect_2d_landmarks(image_path, device)

    shape = torch.zeros(1, config.shape_params, requires_grad=True, device=device)
    expr  = torch.zeros(1, config.expression_params, requires_grad=True, device=device)
    pose  = torch.zeros(1, config.pose_params, requires_grad=True, device=device)
    cam   = torch.tensor([[0.0, 0.0, 10.0]], requires_grad=True, device=device)

    optimizer = torch.optim.Adam([shape, expr, pose, cam], lr=0.01)
    loss_fn = torch.nn.MSELoss()

    for i in trange(1000, desc="Optimizing"):
        verts, landmarks3d = flame(
            shape_params=shape,
            expression_params=expr,
            pose_params=pose
        )  # landmarks3d: [1, 68, 3]

        lmks_2d = project_vertices(landmarks3d, cam, image_size)  # [1, 68, 2]

        loss = loss_fn(lmks_2d, target_2d) + 0.001 * torch.norm(shape) + 0.001 * torch.norm(expr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"✅ Final landmark loss: {loss.item():.4f}")
    return verts, flame.faces_tensor, cam

# ---------- 5. Main ----------
if __name__ == "__main__":
    image_path = "test5.jpg"  # change to your image file
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flame, config = get_flame_model(device)

    verts, faces, cam = optimize_flame(image_path, flame, config, device)

    save_obj("fitted_face.obj", verts[0], faces)
    print("✅ Saved aligned FLAME mesh to fitted_face.obj")
