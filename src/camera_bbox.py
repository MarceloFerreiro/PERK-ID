import argparse
import time
from pathlib import Path

import numpy as np
import torch

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: opencv-python. Install it with 'pip install opencv-python'.") from exc

from src.features.read_image import read_image
from src.models.net import MaskedAutoencoder

TARGET_SIZE = 256


def _select_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _preprocess_frame(frame_bgr: np.ndarray) -> tuple[torch.Tensor, dict]:
    frame_rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB
    image, meta = read_image(frame_rgb, target_size=TARGET_SIZE, return_meta=True)
    if image is None or meta is None:
        raise ValueError("Invalid frame size")

    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    tensor = tensor.div_(255.0)
    return tensor, meta


def _bbox_to_frame(bbox_norm: np.ndarray, meta: dict) -> tuple[int, int, int, int]:
    x, y, bw, bh = bbox_norm.tolist()

    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    bw = max(0.0, min(1.0, bw))
    bh = max(0.0, min(1.0, bh))

    x1 = x * TARGET_SIZE
    y1 = y * TARGET_SIZE
    x2 = (x + bw) * TARGET_SIZE
    y2 = (y + bh) * TARGET_SIZE

    x1 = (x1 + meta["crop_x"]) / meta["scale_x"]
    x2 = (x2 + meta["crop_x"]) / meta["scale_x"]
    y1 = (y1 + meta["crop_y"]) / meta["scale_y"]
    y2 = (y2 + meta["crop_y"]) / meta["scale_y"]

    x1 = int(max(0, min(meta["frame_w"] - 1, round(x1))))
    y1 = int(max(0, min(meta["frame_h"] - 1, round(y1))))
    x2 = int(max(0, min(meta["frame_w"] - 1, round(x2))))
    y2 = int(max(0, min(meta["frame_h"] - 1, round(y2))))

    return x1, y1, x2, y2


def _confidence_from_recon(recon: torch.Tensor, inp: torch.Tensor, alpha: float) -> float:
    # Heuristic: lower reconstruction error => higher confidence.
    mse = torch.mean((recon - inp) ** 2).item()
    conf = float(np.exp(-alpha * mse))
    return max(0.0, min(1.0, conf))


def _color_from_conf(conf: float) -> tuple[int, int, int]:
    conf = max(0.0, min(1.0, conf))
    r = int(255 * (1.0 - conf))
    g = int(255 * conf)
    return (0, g, r)


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-time bounding box inference from camera")
    parser.add_argument("--params", type=Path, required=True, help="Path to model weights (.pt)")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension used to train the model")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--device", type=str, default=None, help="Force device: cpu, cuda, or mps")
    parser.add_argument("--width", type=int, default=0, help="Capture width (0 keeps default)")
    parser.add_argument("--height", type=int, default=0, help="Capture height (0 keeps default)")
    parser.add_argument("--flip", action="store_true", help="Mirror image horizontally")
    parser.add_argument("--conf-alpha", type=float, default=12.0, help="Confidence sensitivity")
    parser.add_argument("--box-thickness", type=int, default=2, help="Bounding box line thickness")
    parser.add_argument("--show-fps", action="store_true", help="Overlay FPS")
    args = parser.parse_args()

    if not args.params.exists():
        print(f"[ERROR] Weights not found: {args.params}")
        return 1

    device = _select_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = MaskedAutoencoder(in_channels=3, latent_dim=args.latent_dim).to(device)
    model.load_state_dict(torch.load(args.params, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(args.camera)
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {args.camera}")
        return 1

    last_time = time.time()
    fps = 0.0

    with torch.inference_mode():
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] Failed to read frame")
                break

            if args.flip:
                frame = cv2.flip(frame, 1)

            inp_tensor, meta = _preprocess_frame(frame)
            inp_tensor = inp_tensor.to(device)

            recon, latent = model(inp_tensor, return_latent=True)
            bbox = model.predict_bbox(latent).squeeze(0).detach().cpu().numpy()
            conf = _confidence_from_recon(recon, inp_tensor, args.conf_alpha)

            x1, y1, x2, y2 = _bbox_to_frame(bbox, meta)
            color = _color_from_conf(conf)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, args.box_thickness)
            cv2.putText(
                frame,
                f"conf: {conf:.2f}",
                (x1, max(10, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

            if args.show_fps:
                now = time.time()
                dt = max(now - last_time, 1e-6)
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
                last_time = now
                cv2.putText(
                    frame,
                    f"fps: {fps:.1f}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (240, 240, 240),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("PERK-ID Camera", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
