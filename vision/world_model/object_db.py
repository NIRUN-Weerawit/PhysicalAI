"""
Object database (World Model): persistent storage of tracked objects.

Stores every detected/identified object with:
  - object_id (UUID, assigned by CrossCameraMatcher)
  - class_name (from detector)
  - 3D position (in world frame)
  - observation history (list of (timestamp, camera, confidence, ...))
  - metadata (color histogram, bounding boxes, etc.)

The orchestrator queries this DB to answer:
  "Where is the blue box right now?"
  "What objects are within 0.5m of the arm end-effector?"
  "Show me all cans detected in the last 30 seconds."

Storage: in-memory dict with optional SQLite persistence.
"""
import json
import time
import uuid
import sqlite3
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import defaultdict


@dataclass
class ObjectRecord:
    """A single tracked object in the world model."""
    object_id: str
    class_name: str
    position_world: tuple  # (x, y, z) in meters
    confidence: float
    timestamp: float  # last seen
    first_seen: float
    observations: list = field(default_factory=list)
        # [{camera, confidence, timestamp, centroid_2d, depth, ...}]
    color_histogram: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def age(self) -> float:
        """Seconds since first seen."""
        return time.monotonic() - self.first_seen

    @property
    def age_last_seen(self) -> float:
        """Seconds since last seen."""
        return time.monotonic() - self.timestamp

    def is_stale(self, timeout: float = 30.0) -> bool:
        return self.age_last_seen > timeout


class ObjectDB:
    """In-memory object database with optional SQLite persistence.

    Provides spatio-temporal querying of tracked objects.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        stale_timeout: float = 30.0,
        max_objects: int = 1000,
        persistence: bool = False,
    ):
        self._objects: dict[str, ObjectRecord] = {}
        self._stale_timeout = stale_timeout
        self._max_objects = max_objects

        if persistence and db_path != ":memory:":
            self._persistence = persistence
            self._db_path = db_path
            self._init_db()

    def _init_db(self):
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS objects (
                id          TEXT PRIMARY KEY,
                class_name  TEXT NOT NULL,
                position    TEXT NOT NULL,   -- JSON: [x, y, z]
                confidence  REAL,
                timestamp   REAL,
                first_seen  REAL,
                observations TEXT,            -- JSON list
                color_hist   TEXT,            -- JSON list
                metadata     TEXT             -- JSON dict
            )
        """)
        self._conn.commit()

    # ---- Core operations ----

    def add(self, obj: ObjectRecord) -> str:
        """Upsert an object. Returns the object_id."""
        self._objects[obj.object_id] = obj
        if getattr(self, '_persistence', False):
            self._persist(obj)
        if len(self._objects) > self._max_objects:
            self.evict_stale()
        return obj.object_id

    def update(self, object_id: str, position: tuple, timestamp: float,
               observation: dict = None, confidence: float = None):
        """Update position and timestamp of an existing object.

        Merges: if the object was already in, adds a new observation and
        updates position using an exponential moving average.
        """
        obj = self._objects.get(object_id)
        if obj is None:
            print(f"[ObjectDB] Object {object_id} not found, creating new.")
            new_obj = ObjectRecord(
                object_id=object_id,
                class_name=observation.get("class_name", "unknown") if observation else "unknown",
                position_world=position,
                confidence=confidence or 0.0,
                timestamp=timestamp,
                first_seen=timestamp,
            )
            if observation:
                new_obj.observations.append(observation)
            self.add(new_obj)
            return

        # Exponential moving average for position smoothing
        alpha = 0.3
        pos = np.array(position)
        old_pos = np.array(obj.position_world)
        smooth_pos = alpha * pos + (1 - alpha) * old_pos
        obj.position_world = tuple(smooth_pos.tolist())

        obj.timestamp = timestamp
        if confidence is not None:
            obj.confidence = confidence
        if observation:
            obj.observations.append(observation)

        if getattr(self, '_persistence', False):
            self._persist(obj)

    def get(self, object_id: str) -> Optional[ObjectRecord]:
        return self._objects.get(object_id)

    def get_all(self, include_stale: bool = False) -> list[ObjectRecord]:
        if include_stale:
            return list(self._objects.values())
        return [o for o in self._objects.values() if not o.is_stale(self._stale_timeout)]

    def has(self, object_id: str) -> bool:
        return object_id in self._objects

    def remove(self, object_id: str):
        self._objects.pop(object_id, None)

    def evict_stale(self, timeout: float = None):
        """Remove objects not seen in `timeout` seconds."""
        if timeout is None:
            timeout = self._stale_timeout
        to_remove = [oid for oid, obj in self._objects.items() if obj.age_last_seen > timeout]
        for oid in to_remove:
            del self._objects[oid]

    # ---- Spatial queries ----

    def query_near(self, point: tuple, radius: float,
                   class_filter: list = None) -> list[ObjectRecord]:
        """All objects within `radius` meters of `point` in world frame."""
        import numpy as np
        pt = np.array(point)
        results = []
        for obj in self._objects.values():
            pos = np.array(obj.position_world)
            if np.linalg.norm(pos - pt) <= radius:
                if class_filter and obj.class_name not in class_filter:
                    continue
                results.append(obj)
        return results

    def query_by_class(self, class_name: str) -> list[ObjectRecord]:
        """All objects with the given class_name (fuzzy match on substring)."""
        return [o for o in self._objects.values()
                if class_name.lower() in o.class_name.lower()]

    def query_by_id(self, object_id: str) -> Optional[ObjectRecord]:
        return self._objects.get(object_id)

    def nearest_to(self, point: tuple, class_filter: str = None,
                   max_results: int = 1) -> list[ObjectRecord]:
        """Find the N nearest objects to a point, optionally filtered.

        Useful for: "Which can should the arm pick up?" → find nearest can to TCP.
        """
        import numpy as np
        pt = np.array(point)
        candidates = list(self._objects.values())
        if class_filter:
            candidates = [o for o in candidates if class_filter.lower() in o.class_name.lower()]
        candidates.sort(key=lambda o: np.linalg.norm(np.array(o.position_world) - pt))
        return candidates[:max_results]

    # ---- Temporal queries ----

    def query_recent(self, seconds: float, class_filter: str = None) -> list[ObjectRecord]:
        """Objects seen within the last `seconds`."""
        cutoff = time.monotonic() - seconds
        results = [o for o in self._objects.values() if o.timestamp >= cutoff]
        if class_filter:
            results = [o for o in results if class_filter.lower() in o.class_name.lower()]
        return results

    # ---- Transform queries (needs TransformTree) ----

    def query_near_frame(self, target_frame: str, source_frame: str,
                         radius: float, tf_tree,
                         class_filter: list = None) -> list[ObjectRecord]:
        """Objects within `radius` of target_frame, expressed in source_frame.

        This is KEY for the orchestrator:
          "What objects are within 0.3m of the arm TCP?"
        """
        import numpy as np
        # Transform target_frame origin into world
        T = tf_tree.lookup("world", target_frame)
        target_in_world = tuple(T[:3, 3])

        return self.query_near(target_in_world, radius, class_filter)

    # ---- Persistence ----

    def _persist(self, obj: ObjectRecord):
        """Save object to SQLite."""
        position = obj.position_world
        if isinstance(position, np.ndarray):
            position = position.tolist()

        self._conn.execute("""
            INSERT OR REPLACE INTO objects
            (id, class_name, position, confidence, timestamp, first_seen,
             observations, color_hist, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            obj.object_id,
            obj.class_name,
            json.dumps(position),
            obj.confidence,
            obj.timestamp,
            obj.first_seen,
            json.dumps(obj.observations),
            json.dumps(obj.color_histogram if isinstance(obj.color_histogram, list) else obj.color_histogram.tolist()),
            json.dumps(obj.metadata),
        ))
        self._conn.commit()

    def load_from_db(self):
        """Restore objects from SQLite."""
        if not getattr(self, '_persistence', False):
            return
        rows = self._conn.execute("SELECT * FROM objects").fetchall()
        for row in rows:
            obj = ObjectRecord(
                object_id=row[0],
                class_name=row[1],
                position_world=json.loads(row[2]),
                confidence=row[3],
                timestamp=row[4],
                first_seen=row[5],
                observations=json.loads(row[6]),
                color_histogram=json.loads(row[7]) if row[7] else [],
                metadata=json.loads(row[8]) if row[8] else {},
            )
            self._objects[obj.object_id] = obj

    # ---- Diagnostics ----

    def snapshot(self) -> dict:
        """Return a full snapshot of the DB as a JSON-serializable dict."""
        return {
            oid: {
                "object_id": o.object_id,
                "class_name": o.class_name,
                "position_world": o.position_world,
                "confidence": o.confidence,
                "last_seen": o.timestamp,
                "age_seconds": o.age_last_seen,
                "observation_count": len(o.observations),
            }
            for oid, o in self._objects.items()
        }

    def __len__(self):
        return len(self._objects)

    def __repr__(self):
        return f"ObjectDB(objects={len(self._objects)})"


# Convenience: numpy is used inside spatial queries but imported lazily
import numpy as np
