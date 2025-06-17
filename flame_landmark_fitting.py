import cv2
import numpy as np
import torch
import mediapipe as mp
from tqdm import trange
from flame_pytorch.flame import FLAME
from flame_pytorch.config import get_config
from utils.landmark_detector import detect_68_landmarks


# These indices map MediaPipe's 468 landmarks to the closest FLAME 68 landmark order
# You MUST use the correct correspondence if you're using FLAME's 68-lmk shape
# This list is approximate and should ideally be replaced with a better 68-point map.
FLAME_LMK_IDS = list(range(68))  # FLAME uses standard iBUG 68-point format


# ---------- Step 1: 2D landmark detection ----------
def detect_2d_landmarks(image_path):
    mp_face = mp.solutions.face_mesh
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    with mp_face.FaceMesh(static_image_mode=True) as face_mesh:
        result = face_mesh.process(img_rgb)
        if not result.multi_face_landmarks:
            raise RuntimeError("No face detected")
        landmarks = result.multi_face_landmarks[0].landmark
        coords = np.array([[l.x * w, l.y * h] for l in landmarks], dtype=np.float32)
        return coords


# ---------- Step 2: Camera projection ----------
def project_vertices(vertices, camera_params, image_size):
    fov = 5.0
    focal = 0.5 * image_size / np.tan(0.5 * fov * np.pi / 180.0)
    cam_t = camera_params[:, :3]  # tx, ty, tz

    projected = vertices + cam_t.unsqueeze(1)
    x = projected[:, :, 0] * focal / projected[:, :, 2] + image_size / 2
    y = -projected[:, :, 1] * focal / projected[:, :, 2] + image_size / 2
    return torch.stack([x, y], dim=-1)


# ---------- Step 3: Get FLAME model ----------
def get_flame_model(device):
    config = get_config()
    config.batch_size = 1
    flame = FLAME(config).to(device)
    return flame


# ---------- Step 4: Optimization ----------
def optimize_flame_to_landmarks(image_path, flame, config, device):
    image_size = 512
    target_2d_full, img = detect_68_landmarks(image_path)
    target_2d = torch.tensor(target_2d_full, dtype=torch.float32, device=device).unsqueeze(0)  # shape [1, 68, 2]


    # Init FLAME params
    shape = torch.zeros(1, config.shape_params, requires_grad=True, device=device)
    expr  = torch.zeros(1, config.expression_params, requires_grad=True, device=device)
    pose  = torch.zeros(1, config.pose_params, requires_grad=True, device=device)
    cam   = torch.tensor([[0.0, 0.0, 10.0]], requires_grad=True, device=device)

    optimizer = torch.optim.Adam([shape, expr, pose, cam], lr=0.01)
    loss_fn = torch.nn.MSELoss()

    for i in trange(200):
        verts, landmarks3d = flame(
            shape_params=shape,
            expression_params=expr,
            pose_params=pose
        )
        lmks_3d = landmarks3d[:, :68, :]
        lmks_2d = project_vertices(lmks_3d, cam, image_size)

        loss = loss_fn(lmks_2d, target_2d)
        prior_loss = 0.001 * torch.norm(shape) + 0.001 * torch.norm(expr)
        total_loss = loss + prior_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print("Final landmark loss:", loss.item())
    return verts, flame.faces_tensor, cam


# ---------- Run ----------
if __name__ == '__main__':
    image_path = 'testa.webp'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_config()
    flame = get_flame_model(device)

    verts, faces, cam = optimize_flame_to_landmarks(image_path, flame, config, device)

    from pytorch3d.io import save_obj
    save_obj("fitted_face.obj", verts[0], faces)
    print("Saved aligned FLAME mesh to fitted_face.obj")
