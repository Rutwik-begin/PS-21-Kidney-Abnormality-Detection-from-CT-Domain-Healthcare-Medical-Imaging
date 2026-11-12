"""
explainability.py - Grad-CAM Visualizations for Clinical Interpretability of Kidney CT.
Ensures that the model focuses on actual renal lesions (parenchyma, renal pelvis, cysts, calculi)
rather than spurious shortcuts (e.g. vertebral bone density, CT bed, body outline).
"""

import os
from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

class GradCAM:
    """
    Grad-CAM extractor for PyTorch CNN backbones (e.g., ResNet50, EfficientNet, Custom CNN).
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generates Grad-CAM heatmap normalized to [0, 1].
        input_tensor: (1, 3, H, W)
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # Channel-wise pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True) # (1, C, 1, 1)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True) # (1, 1, H', W')
        cam = F.relu(cam) # keep only features that positively correlate

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_cam_on_image(
    img: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = 'jet'
) -> np.ndarray:
    """
    Overlays a Grad-CAM heatmap on an RGB image using pure Pillow and Matplotlib.
    img: float in [0, 1] or uint8 in [0, 255]
    """
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    h, w = img.shape[:2]
    # Resize cam to match image size using PIL
    cam_pil = Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
    cam_resized = np.array(cam_pil) / 255.0

    # Apply colormap
    cmap = plt.get_cmap(colormap_name)
    colored_cam = cmap(cam_resized)[:, :, :3] # RGB in [0, 1]
    colored_cam = (colored_cam * 255).astype(np.uint8)

    # Blend
    overlay = (img.astype(float) * (1.0 - alpha) + colored_cam.astype(float) * alpha).astype(np.uint8)
    return overlay


def plot_gradcam_gallery(
    samples: list,
    save_path: Optional[str] = None
):
    """
    Plots a multi-panel figure showing Original CT, Grad-CAM Heatmap, and Overlay
    across all 4 abnormality classes.
    """
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(10, 3.2 * n), dpi=300)
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for i, s in enumerate(samples):
        # Original
        axes[i, 0].imshow(s['original'])
        axes[i, 0].set_title(f"Ground Truth: {s['true_label']}\n(Original CT Slice)", fontsize=10, fontweight='bold')
        axes[i, 0].axis('off')

        # Heatmap
        axes[i, 1].imshow(s['cam'], cmap='jet')
        axes[i, 1].set_title(f"Predicted: {s['pred_label']}\n(Grad-CAM Saliency)", fontsize=10, fontweight='bold')
        axes[i, 1].axis('off')

        # Overlay
        axes[i, 2].imshow(s['overlay'])
        axes[i, 2].set_title(f"Clinical Attenuation Focus\n(Confidence: {s['confidence']:.2%})", fontsize=10, fontweight='bold')
        axes[i, 2].axis('off')

    plt.suptitle("Radiological Feature Interpretability via Grad-CAM", fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
