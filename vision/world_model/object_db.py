"""
object_db.py — Persistent object database with CLIP-based re-identification.

Every detected object gets a unique human-readable ID (chair_1, chair_2, ...).
Re-identification uses CLIP embedding cosine similarity, with position tiebreaker
for visually identical objects (e.g. two identical chairs next to each other).

Objects persist for the entire run (or until explicitly forgotten). SQLite
persistence saves object records + naming counters across restarts.
"""

import json
import time
import sqlite3
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional


# ── Defaults ───────────────────────────────────────────────────────────

MATCH_THRESHOLD = 0.82
AMBIGUITY_THRESHOLD = 0.65
POSITION_TIEBREAKER_M = 0.5  # meters


# ── Object record ──────────────────────────────────────────────────────

@dataclass
class ObjectRecord:
    """A single tracked object in the world model."""
    object_id: str            # "chair_1", "cup_3", etc.
    class_name: str           # "chair", "cup", "table"
    embedding: list = field(default_factory=list)  # 512-dim float list
    description: str = ""     # lazy VLM description ("Red plastic chair")
    position_world: tuple = (0.0, 0.0, 0.0)  # (x, y, z) meters
    position_uncertainty: float = 0.0  # meters std
    confidence: float = 0.0
    first_seen: float = 0.0  # monotonic time
    last_seen: float = 0.0   # monotonic time
    observations: list = field(default_factory=list)
    obb_width: float = 0.0
    obb_depth: float = 0.0
    obb_height: float = 0.0
    obb_angle: float = 0.0
    mask_area: float = 0.0
    point_count: int = 0
    metadata: dict = field(default_factory=dict)

    # Convenience properties
    @property
    def age(self) -> float:
        return time.monotonic() - self.first_seen

    @property
    def age_last_seen(self) -> float:
        return time.monotonic() - self.last_seen

    @property
    def observation_count(self) -> int:
        return len(self.observations)


# ── Object database ────────────────────────────────────────────────────

class ObjectDB:
    """"Persistent object database with CLIP-based re-identification.

    Stores objects by class-indexed IDs (chair_1, chair_2...). Objects can only
    be removed via forget_object/forget_class/forget_all — no stale eviction.
    SQLite persistence is optional but recommended across restarts.
    """

    def __init__(
        self,
        embedder=None,              # EmbeddingMatcher instance
        db_path: str = ":memory:",
        persistence: bool = False,
        match_threshold: float = MATCH_THRESHOLD,
        ambiguity_threshold: float = AMBIGUITY_THRESHOLD,
        position_tiebreaker_m: float = POSITION_TIEBREAKER_M,
    ):
        self._objects: dict[str, ObjectRecord] = {}
        self._counters: dict[str, int] = {}  # class_name → next index
        self._embedder = embedder
        self._match_threshold = match_threshold
        self._ambiguity_threshold = ambiguity_threshold
        self._position_tiebreaker = position_tiebreaker_m

        if persistence and db_path != ":memory:":
            self._persistence = True
            self._db_path = db_path
            self._init_db()
            self._load_from_db()

    def _init_db(self):
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("""CREATE TABLE IF NOT EXISTS objects (
            id TEXT PRIMARY KEY,
            class_name TEXT NOT NULL,
            embedding TEXT,
            description TEXT,
            position TEXT,
            uncertainty REAL,
            confidence REAL,
            first_seen REAL,
            last_seen REAL,
            observations TEXT,
            obb_width REAL,
            obb_depth REAL,
            obb_height REAL,
            obb_angle REAL,
            mask_area REAL,
            point_count INTEGER,
            metadata TEXT
        )""")
        self._conn.execute("""CREATE TABLE IF NOT EXISTS counters (
            class_name TEXT PRIMARY KEY,
            next_index INTEGER NOT NULL
        )""")
        self._conn.commit()

    # ── Naming ──────────────────────────────────────────────────────────

    def _next_object_id(self, class_name: str) -> str:
        """Return the next unique ID for this class, e.g. 'chair_3'."""
        idx = self._counters.get(class_name, 0) + 1
        self._counters[class_name] = idx
        return f"{class_name}_{idx}"

    # ── Core operations ────────────────────────────────────────────────

    def add(self, rgb_crop: Optional[np.ndarray] = None,
            class_name: str = "unknown",
            position_world: tuple = (0.0, 0.0, 0.0),
            confidence: float = 0.0,
            timestamp: Optional[float] = None,
            observation: Optional[dict] = None,
            force_new: bool = False,
            ) -> tuple[ObjectRecord, str]:
        """Add a detection — either creates or updates an existing object.

        If embedder is available, matches against existing objects of the same
        class using CLIP visual embedding. If a match is found, updates the
        existing record (position smoothing, append observation). Otherwise
        creates a new named record.

        Args:
            rgb_crop: HxWx3 uint8 RGB image of the detection crop (for embedding).
            class_name: Detector class label.
            position_world: (x, y, z) in map frame.
            confidence: Detection confidence.
            timestamp: time.monotonic() value.
            observation: Dict with detection metadata (bbox, depth, camera, etc.).
            force_new: If True, skip matching and always create new record.

        Returns:
            (ObjectRecord, action) where action is "matched" | "new"
        """
        if timestamp is None:
            timestamp = time.monotonic()
        if observation is None:
            observation = {}

        # Compute embedding if possible
        embedding: Optional[np.ndarray] = None
        if self._embedder is not None and rgb_crop is not None:
            embedding = self._embedder.compute_embedding(rgb_crop)

        # Try to match against existing objects of the same class
        if not force_new and self._embedder is not None and embedding is not None:
            candidates = []
            for oid, obj in self._objects.items():
                if obj.class_name == class_name and len(obj.embedding) == 512:
                    stored_emb = np.array(obj.embedding, dtype=np.float32)
                    candidates.append((oid, stored_emb, obj.position_world))

            if candidates:
                matched_id, sim, reason = self._embedder.match(
                    embedding, candidates)

                # Apply our own thresholds to the match result
                if matched_id and sim >= self._match_threshold:
                    is_same = True
                elif matched_id and sim >= self._ambiguity_threshold:
                    # Position tiebreaker
                    obj = self._objects.get(matched_id)
                    if obj:
                        dist = np.linalg.norm(
                            np.array(position_world[:2]) -
                            np.array(obj.position_world[:2]))
                        is_same = dist <= self._position_tiebreaker
                    else:
                        is_same = False
                else:
                    is_same = False

                if is_same:
                    # Update existing record
                    self._update(matched_id, embedding, position_world,
                                 confidence, timestamp, observation)
                    return self._objects[matched_id], "matched"

        # Create new record
        oid = self._next_object_id(class_name)
        obj = ObjectRecord(
            object_id=oid,
            class_name=class_name,
            embedding=embedding.tolist() if embedding is not None else [],
            position_world=position_world,
            confidence=confidence,
            first_seen=timestamp,
            last_seen=timestamp,
            observations=[observation],
        )
        self._objects[oid] = obj
        self._persist(obj)
        return obj, "new"

    def _update(self, object_id: str, embedding: Optional[np.ndarray],
                position: tuple, confidence: float,
                timestamp: float, observation: dict):
        """Update an existing record — position smoothing + embed blending."""
        obj = self._objects.get(object_id)
        if obj is None:
            return

        # Position smoothing (EMA alpha=0.3)
        old_pos = np.array(obj.position_world, dtype=float)
        new_pos = np.array(position, dtype=float)
        if np.linalg.norm(new_pos - old_pos) < 5.0:  # sanity: don't blend if >5m jump
            alpha = 0.3
            smooth = alpha * new_pos + (1.0 - alpha) * old_pos
            obj.position_world = tuple(smooth.tolist())
        else:
            obj.position_world = position

        # Embedding blending (slow EMA — new observations refine slowly)
        if embedding is not None and len(obj.embedding) == 512:
            old_emb = np.array(obj.embedding, dtype=np.float32)
            alpha_emb = 0.15
            blended = alpha_emb * embedding + (1.0 - alpha_emb) * old_emb
            norm = np.linalg.norm(blended)
            obj.embedding = (blended / norm).tolist() if norm > 0 else blended.tolist()

        obj.confidence = confidence
        obj.last_seen = timestamp
        if observation:
            obj.observations.append(observation)
        self._persist(obj)

    # ── Retrieval ───────────────────────────────────────────────────────

    def get(self, object_id: str) -> Optional[ObjectRecord]:
        return self._objects.get(object_id)

    def get_by_name(self, name: str) -> Optional[ObjectRecord]:
        """Find object by its unique name (e.g. 'chair_1') or class prefix (e.g. 'chair')."""
        # Exact match first
        if name in self._objects:
            return self._objects[name]
        # Fuzzy by class name
        for obj in self._objects.values():
            if obj.object_id == name or obj.class_name == name:
                return obj
        return None

    def get_all(self) -> list[ObjectRecord]:
        return list(self._objects.values())

    def get_by_class(self, class_name: str) -> list[ObjectRecord]:
        """All objects matching a class (fuzzy substring)."""
        return [o for o in self._objects.values()
                if class_name.lower() in o.class_name.lower()]

    def has(self, object_id: str) -> bool:
        return object_id in self._objects

    def remove(self, object_id: str):
        self._objects.pop(object_id, None)
        if getattr(self, '_persistence', False):
            self._conn.execute("DELETE FROM objects WHERE id = ?", (object_id,))
            self._conn.commit()

    # ── Forgetting ──────────────────────────────────────────────────────

    def forget_object(self, object_id: str) -> Optional[str]:
        """Remove a single named object. Returns its name or None."""
        obj = self._objects.pop(object_id, None)
        if obj:
            if getattr(self, '_persistence', False):
                self._conn.execute("DELETE FROM objects WHERE id = ?", (object_id,))
                self._conn.commit()
            return object_id
        return None

    def forget_class(self, class_name: str) -> int:
        """Remove all objects of a class. Returns count removed."""
        to_remove = [oid for oid, obj in self._objects.items()
                     if class_name.lower() in obj.class_name.lower()]
        for oid in to_remove:
            del self._objects[oid]
        if getattr(self, '_persistence', False):
            placeholders = ",".join("?" for _ in to_remove)
            self._conn.execute(f"DELETE FROM objects WHERE id IN ({placeholders})", to_remove)
            self._conn.commit()
        return len(to_remove)

    def forget_all(self) -> int:
        """Remove all objects. Returns count removed."""
        count = len(self._objects)
        self._objects.clear()
        if getattr(self, '_persistence', False):
            self._conn.execute("DELETE FROM objects")
            self._conn.execute("DELETE FROM counters")
            self._conn.commit()
        return count

    # ── Spatial queries ─────────────────────────────────────────────────

    def query_near(self, point: tuple, radius: float,
                   class_filter: Optional[list[str]] = None) -> list[ObjectRecord]:
        pt = np.array(point[:2])  # 2D projection
        results = []
        for obj in self._objects.values():
            pos = np.array(obj.position_world[:2])
            if np.linalg.norm(pos - pt) <= radius:
                if class_filter and obj.class_name not in class_filter:
                    continue
                results.append(obj)
        return results

    def nearest_to(self, point: tuple, class_filter: str = None,
                   max_results: int = 1) -> list[ObjectRecord]:
        pt = np.array(point[:2])
        candidates = list(self._objects.values())
        if class_filter:
            candidates = [o for o in candidates
                          if class_filter.lower() in o.class_name.lower()]
        candidates.sort(key=lambda o: np.linalg.norm(
            np.array(o.position_world[:2]) - pt))
        return candidates[:max_results]

    # ── Temporal queries ────────────────────────────────────────────────

    def query_recent(self, seconds: float, class_filter: str = None) -> list[ObjectRecord]:
        cutoff = time.monotonic() - seconds
        results = [o for o in self._objects.values() if o.last_seen >= cutoff]
        if class_filter:
            results = [o for o in results
                       if class_filter.lower() in o.class_name.lower()]
        return results

    def query_history(self, class_name: str,
                      minutes_ago: float = None) -> list[dict]:
        now = time.monotonic()
        results = []
        for obj in self._objects.values():
            if class_name.lower() not in obj.class_name.lower():
                continue
            for obs in obj.observations:
                ts = obs.get("timestamp", 0)
                if minutes_ago is not None and (now - ts) < minutes_ago * 60:
                    continue
                results.append({
                    "object_id": obj.object_id,
                    "class_name": obj.class_name,
                    "timestamp": ts,
                    "position": list(obj.position_world),
                    "confidence": obs.get("confidence", 0),
                })
        return results

    def has_moved(self, object_id: str, threshold_m: float = 0.2) -> dict:
        """Check if a specific named object has moved since its first observation."""
        obj = self._objects.get(object_id)
        if not obj:
            return {"moved": False, "reason": f"Object '{object_id}' not found"}
        if len(obj.observations) < 2:
            return {"moved": False, "reason": "Only 1 observation, cannot determine movement",
                    "observation_count": 1}
        first_pos = np.array(obj.position_world)
        last_pos = np.array(obj.position_world)
        delta = float(np.linalg.norm(last_pos - first_pos))
        return {
            "object_id": obj.object_id,
            "class_name": obj.class_name,
            "moved": delta > threshold_m,
            "delta_m": round(delta, 3),
            "first_seen": obj.first_seen,
            "observation_count": len(obj.observations),
        }

    # ── Description generation ─────────────────────────────────────────

    def describe_object(self, object_id: str,
                        rgb_crop: np.ndarray = None) -> Optional[str]:
        """Generate a short text description for an object.

        Uses the LLM (via the robot's description pipeline) or falls back
        to class_name + position. Stores the result in the record.
        """
        obj = self._objects.get(object_id)
        if not obj:
            return None
        # Fallback: just class name and position
        desc = f"{obj.class_name} at {tuple(round(v, 2) for v in obj.position_world)}"
        obj.description = desc
        return desc

    def snapshot(self) -> dict:
        """JSON-serializable snapshot for the dashboard."""
        return {
            oid: {
                "object_id": o.object_id,
                "class_name": o.class_name,
                "description": o.description or o.class_name,
                "position_world": list(o.position_world),
                "confidence": o.confidence,
                "last_seen": o.last_seen,
                "first_seen": o.first_seen,
                "observation_count": len(o.observations),
                "uncertainty": o.position_uncertainty,
            }
            for oid, o in self._objects.items()
        }

    # ── Persistence ─────────────────────────────────────────────────────

    def _persist(self, obj: ObjectRecord):
        if not getattr(self, '_persistence', False):
            return
        pos = obj.position_world
        if isinstance(pos, np.ndarray):
            pos = pos.tolist()
        self._conn.execute("""INSERT OR REPLACE INTO objects
            (id, class_name, embedding, description, position, uncertainty,
             confidence, first_seen, last_seen, observations,
             obb_width, obb_depth, obb_height, obb_angle,
             mask_area, point_count, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
            obj.object_id,
            obj.class_name,
            json.dumps(obj.embedding),
            obj.description,
            json.dumps(pos),
            obj.position_uncertainty,
            obj.confidence,
            obj.first_seen,
            obj.last_seen,
            json.dumps(obj.observations),
            obj.obb_width, obj.obb_depth, obj.obb_height,
            obj.obb_angle, obj.mask_area, obj.point_count,
            json.dumps(obj.metadata),
        ))
        self._conn.commit()

        # Save counters
        for cls, idx in self._counters.items():
            self._conn.execute(
                "INSERT OR REPLACE INTO counters (class_name, next_index) VALUES (?, ?)",
                (cls, idx))
        self._conn.commit()

    def _load_from_db(self):
        """Restore objects and counters from SQLite."""
        rows = self._conn.execute("SELECT * FROM objects").fetchall()
        for row in rows:
            embed_list = json.loads(row[2]) if row[2] else []
            obj = ObjectRecord(
                object_id=row[0],
                class_name=row[1],
                embedding=embed_list,
                description=row[3] or "",
                position_world=tuple(json.loads(row[4])),
                position_uncertainty=row[5] or 0.0,
                confidence=row[6] or 0.0,
                first_seen=row[7] or 0.0,
                last_seen=row[8] or 0.0,
                observations=json.loads(row[9]) if row[9] else [],
                obb_width=row[10] or 0.0,
                obb_depth=row[11] or 0.0,
                obb_height=row[12] or 0.0,
                obb_angle=row[13] or 0.0,
                mask_area=row[14] or 0.0,
                point_count=row[15] or 0,
                metadata=json.loads(row[16]) if row[16] else {},
            )
            self._objects[obj.object_id] = obj

        # Load counters
        crows = self._conn.execute("SELECT class_name, next_index FROM counters").fetchall()
        for cls, idx in crows:
            self._counters[cls] = idx

    # ── Utils ───────────────────────────────────────────────────────────

    def __len__(self):
        return len(self._objects)

    def __repr__(self):
        return f"ObjectDB(objects={len(self._objects)}, classes={len(self._counters)})"
