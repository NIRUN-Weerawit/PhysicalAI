"""
Grounded SAM 2 wrapper: open-vocabulary 2D object detection + segmentation.

Uses HuggingFace Transformers for Grounding DINO (no custom CUDA compilation)
and SAM 2 for mask segmentation. This avoids the need for python3.11-dev.
"""
import os
import cv2
import json
import torch
import numpy as np
from pathlib import Path
import pycocotools.mask as mask_util

# SAM 2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
# Grounding DINO via HF Transformers (no CUDA compilation needed)
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

import supervision as sv


class GroundedSAM2Wrapper:
    """Text-promptable 2D detection + segmentation pipeline.

    Uses HuggingFace transformers for Grounding DINO — zero compilation needed.
    SAM 2 is loaded from local checkpoint.

    Usage:
        det = GroundedSAM2Wrapper(config)
        results = det.detect("car. person. box.", image_bgr)
    """

    def __init__(self, config):
        self.cfg = config
        self._build()

    def _build(self):
        device = self.cfg.device
        # ---- SAM 2 ----
        sam2_model = build_sam2(
            self.cfg.sam2_model_config,
            self.cfg.sam2_checkpoint,
            device=device,
        )
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        # ---- Grounding DINO via HF Transformers ----
        model_id = "IDEA-Research/grounding-dino-base"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

        self.device = device
        self._setup_amp()

    def _setup_amp(self):
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_properties(0).major
            if cap >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    @torch.inference_mode()
    def detect(
        self,
        text_prompt: str,
        image_bgr: np.ndarray,
    ) -> list:
        """Run Grounding DINO (HF) + SAM 2 on a single RGB image.

        Args:
            text_prompt: e.g. "car. person. box."  (lowercased, dot-separated)
            image_bgr: HxWx3 BGR numpy array (OpenCV default)

        Returns:
            List of dicts:
                {
                    "class_name": str,
                    "confidence": float,
                    "bbox_xyxy": [x1, y1, x2, y2],  # pixel coords
                    "mask": np.ndarray (H, W, bool),
                    "centroid_2d": (u, v),
                }
        """
        # Convert BGR → RGB (Grounding DINO HF processor expects PIL/RGB tensor)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = image_rgb.shape

        # --- Step 1: Grounding DINO via HF ---
        # The HF processor wants a PIL image
        from PIL import Image
        pil_image = Image.fromarray(image_rgb)

        inputs = self.processor(images=pil_image, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        # Post-process: returns list of dicts (one per image in batch)
        # transformers v4.49+ changed 'threshold' → 'box_threshold'
        try:
            results_hf = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.cfg.box_threshold,
                text_threshold=self.cfg.text_threshold,
                target_sizes=[(h, w)],
            )[0]
        except TypeError:
            results_hf = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=self.cfg.box_threshold,
                text_threshold=self.cfg.text_threshold,
                target_sizes=[(h, w)],
            )[0]

        if len(results_hf["boxes"]) == 0:
            return []

        input_boxes = results_hf["boxes"].cpu().numpy()  # (n, 4) in xyxy pixel coords
        confidences = results_hf["scores"].cpu().numpy()
        labels = results_hf["labels"]

        # --- Step 2: SAM 2 ---
        self.sam2_predictor.set_image(image_rgb)

        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            masks, mask_scores, _ = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=self.cfg.multimask_output,
            )

        if masks.ndim == 4:
            masks = masks.squeeze(1)  # (n, H, W)

        # --- Build results ---
        results = []
        for i in range(len(input_boxes)):
            mask_bool = masks[i].astype(bool)
            ys, xs = np.where(mask_bool)
            if len(ys) == 0:
                centroid = (int(input_boxes[i][0]), int(input_boxes[i][1]))
            else:
                centroid = (int(xs.mean()), int(ys.mean()))

            results.append({
                "class_name": labels[i],
                "confidence": float(confidences[i]),
                "bbox_xyxy": input_boxes[i].tolist(),
                "mask": mask_bool,
                "centroid_2d": centroid,
            })

        return results

    def visualize(
        self,
        image_bgr: np.ndarray,
        detections: list,
        output_path: str,
    ):
        """Draw detection results on image and save."""
        if not detections:
            cv2.imwrite(output_path, image_bgr)
            return

        xyxy = np.array([d["bbox_xyxy"] for d in detections])
        masks = np.array([d["mask"] for d in detections])
        labels = [f"{d['class_name']} {d['confidence']:.2f}" for d in detections]
        class_ids = np.arange(len(detections))

        det = sv.Detections(
            xyxy=xyxy,
            mask=masks,
            class_id=class_ids,
        )

        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        mask_annotator = sv.MaskAnnotator()

        frame = box_annotator.annotate(scene=image_bgr.copy(), detections=det)
        frame = label_annotator.annotate(scene=frame, detections=det, labels=labels)
        frame = mask_annotator.annotate(scene=frame, detections=det)
        cv2.imwrite(output_path, frame)

    def dump_json(self, detections: list, img_path: str, output_path: str):
        """Save detection results as JSON."""
        image = cv2.imread(img_path)
        h, w = image.shape[:2]

        annotations = []
        for d in detections:
            mask_rle = mask_util.encode(
                np.array(d["mask"][:, :, None], order="F", dtype="uint8")
            )[0]
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            annotations.append({
                "class_name": d["class_name"],
                "bbox": d["bbox_xyxy"],
                "segmentation": mask_rle,
                "score": d["confidence"],
                "centroid_2d": d["centroid_2d"],
            })

        results = {
            "image_path": img_path,
            "annotations": annotations,
            "box_format": "xyxy",
            "img_width": w,
            "img_height": h,
        }
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
