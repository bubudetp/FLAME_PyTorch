import numpy as np
import torch 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from flame_pytorch.flame import FLAME
from flame_pytorch.config import get_config


# ---- Step 1: Load landmark embedding (from RingNet) ----
embedding = np.load("./model/flame_dynamic_embedding.npy", allow_pickle=True, encoding="latin1").item()
lmk_face_idx = torch.tensor(np.array(embedding['lmk_face_idx']), dtype=torch.long)
lmk_b_coords = torch.tensor(np.array(embedding['lmk_b_coords']), dtype=torch.float32)

num_verts = 5023
num_faces = 9976

config = get_config()
config.batch_size = 1
flame_model = FLAME(config)


shape_params = torch.zeros(1, config.shape_params)
expression_params = torch.zeros(1, config.expression_params)
pose_params = torch.zeros(1, config.pose_params)

vertices, landmarks= flame_model(
    shape_params=shape_params,
    expression_params=expression_params,
    pose_params=pose_params
)

# Normalize landmarks to image scale
def normalize_landmarks(landmarks_2d, image_size=224):
    """
    Scale landmarks to image_size × image_size with center at (image_size/2, image_size/2)
    """
    min_xy = landmarks_2d.min(dim=1, keepdim=True).values
    max_xy = landmarks_2d.max(dim=1, keepdim=True).values
    scale = image_size / (max_xy - min_xy).max(dim=2, keepdim=True).values
    normalized = (landmarks_2d - min_xy) * scale
    return normalized

landmarks_2d = normalize_landmarks(landmarks[:, :, :2])


landmarks_np = landmarks_2d[0].detach().cpu().numpy()

verts = vertices[0].detach().cpu().numpy()

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=0.5, c='salmon')

ax.view_init(elev=20, azim=45)  # Change elevation and angle
ax.set_title("FLAME 3D Mesh View")
plt.show()