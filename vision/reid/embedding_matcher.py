"""
embedding_matcher.py — CLIP-based visual re-identification for persistent object IDs.

Takes a cropped RGB image of a detected object and computes a 512-dim unit-normalized
embedding. The embedding is used to answer: "is this the same object I saw before?"
Uses cosine similarity with a position-based tiebreaker for ambiguous cases.
"""

import numpy as np
import torch
import open_clip


CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
MATCH_THRESHOLD = 0.82
AMBIGUITY_THRESHOLD = 0.65
POSITION_TIEBREAKER_M = 0.5


class EmbeddingMatcher:
    """CLIP-based visual object fingerprinting and re-identification."""

    def __init__(self, model: str = CLIP_MODEL,
                 pretrained: str = CLIP_PRETRAINED,
                 device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model, _, self._transform = open_clip.create_model_and_transforms(
            model, pretrained=pretrained)
        self.model = self.model.to(self.device).eval()
        self._tokenizer = open_clip.get_tokenizer(model)

    # ── Public API ──────────────────────────────────────────────────────

    def compute_embedding(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return 512-dim unit-normalized CLIP embedding for an RGB crop.

        Args:
            image_rgb: HxWx3 uint8 RGB image (bounding-box crop).

        Returns:
            float32 array of length 512, L2 norm = 1.0.
        """
        from PIL import Image

        pil_img = Image.fromarray(image_rgb)
        tensor = self._transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model.encode_image(tensor)

        vec = emb.cpu().numpy().flatten().astype(np.float32)
        norm = np.linalg.norm(vec)
        return (vec / norm) if norm > 1e-8 else vec

    def match(self, embedding: np.ndarray,
              candidates: list[tuple[str, np.ndarray, tuple]]) -> tuple:
        """Find the best-matching known object or return None.

        Args:
            embedding: Query embedding (512-dim unit vector).
            candidates: List of (object_id, stored_embedding, world_position_xyz).

        Returns:
            (matched_object_id or None, cosine_similarity, reason_string)
            reason: "match" | "ambiguous_position" | "new_object"
        """
        if not candidates:
            return None, 0.0, "no_candidates"

        best_id = None
        best_sim = -1.0
        best_pos = None

        for obj_id, stored_emb, pos in candidates:
            sim = float(np.dot(embedding, stored_emb))
            if sim > best_sim:
                best_sim = sim
                best_id = obj_id
                best_pos = pos

        if best_sim >= MATCH_THRESHOLD:
            return best_id, best_sim, "match"

        if best_sim >= AMBIGUITY_THRESHOLD:
            # Position tiebreaker: if within POSITION_TIEBREAKER_M → same object
            if best_pos is not None and len(best_pos) >= 2:
                dist = np.linalg.norm(np.array(embedding[:2]) - np.array(best_pos[:2]))
            else:
                dist = float('inf')
            if dist <= POSITION_TIEBREAKER_M:
                return best_id, best_sim, "match"
            return best_id, best_sim, "ambiguous_position"

        return None, best_sim, "new_object"
