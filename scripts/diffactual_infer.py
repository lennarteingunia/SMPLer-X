"""diffactual inference shim for the SMPLer-X fork.

Contract (see diffactual/external/README.md):

    python scripts/diffactual_infer.py --frames <DIR> --out <FILE.npz> \
        [--config <YAML>] [--overrides <JSON>]

Runs mmdet (largest person box) -> SMPLer-X per frame, and writes one .npz with
per-frame SMPL-X parameters + the predicted pinhole camera, in the schema
`diffactual/external/smplerx.md` documents.

Kept on the fork's `diffactual` branch so the fork stays reusable by other projects.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp
import sys

import numpy as np

FORK_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
MAIN_DIR = osp.join(FORK_ROOT, "main")

DEFAULTS = {
    "model": "smpler_x_h32",       # -> main/config/config_<model>.py + pretrained_models/<model>.pth.tar
    "checkpoint": "",              # optional explicit .pth.tar path (overrides <model> default)
    "device": "cuda:0",
    "bbox_thr_px": 50,             # skip person boxes smaller than this (width); height uses 3x
    "mmdet_config": "",            # default: pretrained_models/mmdet/mmdet_faster_rcnn_r50_fpn_coco.py
    "mmdet_checkpoint": "",        # default: pretrained_models/mmdet/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
}

SMPLX_KEYS = (
    "global_orient", "body_pose", "left_hand_pose", "right_hand_pose",
    "jaw_pose", "leye_pose", "reye_pose", "expression", "transl",
)


def load_cfg(argv_config: str | None, overrides: dict) -> dict:
    cfg = dict(DEFAULTS)
    if argv_config:
        import yaml
        with open(argv_config) as fh:
            cfg.update({k: v for k, v in (yaml.safe_load(fh) or {}).items() if v is not None})
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def list_frames(frames_dir: str) -> list[str]:
    frames = sorted(glob.glob(osp.join(frames_dir, "frame_*.png")))
    if not frames:
        frames = sorted(
            p for ext in ("*.png", "*.jpg", "*.jpeg")
            for p in glob.glob(osp.join(frames_dir, ext))
        )
    if not frames:
        raise SystemExit(f"no frames found in {frames_dir}")
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--overrides", default="{}")
    args = ap.parse_args()

    cfg_d = load_cfg(args.config, json.loads(args.overrides))
    frames = list_frames(args.frames)

    # --- SMPLer-X expects to run from main/ with these on sys.path ---
    os.chdir(MAIN_DIR)
    for sub in ("main", "data", "common"):
        sys.path.insert(0, osp.join(FORK_ROOT, sub))

    import cv2
    import torch
    import torchvision.transforms as transforms
    from config import cfg as sx_cfg

    config_path = osp.join("./config", f"config_{cfg_d['model']}.py")
    ckpt_path = cfg_d["checkpoint"] or osp.join(
        FORK_ROOT, "pretrained_models", f"{cfg_d['model']}.pth.tar"
    )
    sx_cfg.get_config_fromfile(config_path)
    sx_cfg.update_test_config("EHF", "na", shapy_eval_split=None,
                              pretrained_model_path=ckpt_path, use_cache=False)
    sx_cfg.update_config(1, "output/diffactual")
    torch.backends.cudnn.benchmark = True

    from base import Demoer
    from utils.preprocessing import generate_patch_image, load_img, process_bbox

    demoer = Demoer()
    demoer._make_model()
    demoer.model.eval()

    from mmdet.apis import inference_detector, init_detector
    from utils.inference_utils import process_mmdet_results

    det_cfg = cfg_d["mmdet_config"] or osp.join(
        FORK_ROOT, "pretrained_models", "mmdet", "mmdet_faster_rcnn_r50_fpn_coco.py")
    det_ckpt = cfg_d["mmdet_checkpoint"] or osp.join(
        FORK_ROOT, "pretrained_models", "mmdet",
        "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth")
    detector = init_detector(det_cfg, det_ckpt, device=cfg_d["device"])

    to_tensor = transforms.ToTensor()
    accum: dict[str, list] = {k: [] for k in SMPLX_KEYS}
    accum["betas"], accum["focal"], accum["princpt"] = [], [], []

    h0 = w0 = None
    for fp in frames:
        img = load_img(fp)  # RGB float
        H, W = img.shape[:2]
        if h0 is None:
            h0, w0 = H, W

        det = inference_detector(detector, fp)
        boxes = process_mmdet_results(det, cat_id=0, multi_person=False)[0]  # largest first
        if len(boxes) < 1:
            box_xywh = np.array([0.0, 0.0, W, H])  # fallback: whole frame
        else:
            b = boxes[0]
            box_xywh = np.array([b[0], b[1], abs(b[2] - b[0]), abs(b[3] - b[1])])

        bbox = process_bbox(box_xywh, W, H)  # -> x, y, w, h (aspect-fixed)
        patch, _, _ = generate_patch_image(img, bbox, 1.0, 0.0, False, sx_cfg.input_img_shape)
        inp = {"img": to_tensor(patch.astype(np.float32)).cuda()[None] / 255.0}

        with torch.no_grad():
            out = demoer.model(inp, {}, {}, "test")

        accum["global_orient"].append(out["smplx_root_pose"].reshape(-1).cpu().numpy())      # 3
        accum["body_pose"].append(out["smplx_body_pose"].reshape(-1).cpu().numpy())          # 63
        accum["left_hand_pose"].append(out["smplx_lhand_pose"].reshape(-1).cpu().numpy())    # 45
        accum["right_hand_pose"].append(out["smplx_rhand_pose"].reshape(-1).cpu().numpy())   # 45
        accum["jaw_pose"].append(out["smplx_jaw_pose"].reshape(-1).cpu().numpy())            # 3
        accum["leye_pose"].append(np.zeros(3, np.float32))
        accum["reye_pose"].append(np.zeros(3, np.float32))
        accum["expression"].append(out["smplx_expr"].reshape(-1).cpu().numpy())              # 10
        accum["transl"].append(out["cam_trans"].reshape(-1).cpu().numpy())                   # 3
        accum["betas"].append(out["smplx_shape"].reshape(-1).cpu().numpy())                  # 10

        fx = sx_cfg.focal[0] / sx_cfg.input_body_shape[1] * bbox[2]
        fy = sx_cfg.focal[1] / sx_cfg.input_body_shape[0] * bbox[3]
        cx = sx_cfg.princpt[0] / sx_cfg.input_body_shape[1] * bbox[2] + bbox[0]
        cy = sx_cfg.princpt[1] / sx_cfg.input_body_shape[0] * bbox[3] + bbox[1]
        accum["focal"].append([fx, fy])
        accum["princpt"].append([cx, cy])

    n = len(frames)
    payload = {k: np.asarray(v, dtype=np.float32) for k, v in accum.items()}
    payload["image_size"] = np.array([w0, h0], dtype=np.int64)
    payload["source_frame_indices"] = np.arange(n, dtype=np.int64)
    payload["hand_pca"] = np.array(False)
    payload["model_gender"] = np.array("neutral")

    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **payload)
    print(f"wrote {args.out}  ({n} frames)")


if __name__ == "__main__":
    main()
