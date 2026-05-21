"""
Transform tree (TF tree) for managing coordinate frames.

Provides a ROS-style /tf tree: every frame is a node in a directed
tree rooted at world.  Given any two frames (A, B) the system returns
the rigid transform from A to B.

Why this matters:
  The orchestrator needs to know "where is the blue box relative to
  the arm end-effector?"  The answer is:

  T_world→tcp ⊕ T_world→box = T_tcp→box

  where T comes from the TF tree.

Frame naming convention:
  <parent>/child   —  e.g. "world/camera_top", "world/arm_base"

Example:

  tree = TransformTree()
  # world is the root (implied)
  tree.set_static("camera_top", [0.0, 0.0, 1.5], [1, 0, 0, 0])  # camera above workspace
  tree.set_static("arm_base",   [0.5, 0.0, 0.5], [1, 0, 0, 0])  # arm on table
  tree.set_static("arm_tcp",    [0.3, 0.0, 0.0], frame_parent="arm_base")  # offset from arm_base
  tree.set_dynamic("camera_front", [0.0, 0.5, 1.2], [1, 0, 0, 0], timestamp=...)

  # "Where is point (0.1, 0.2, 0.3) in camera_top frame, expressed in arm_tcp frame?"
  T = tree.lookup("camera_top", "arm_tcp", time=None)
  point_in_tcp = T @ point_in_camera
"""
import time
import numpy as np
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation
from typing import Optional


@dataclass
class TransformStamped:
    """A single named transform at a point in time."""
    parent: str              # parent frame name
    child: str               # child frame name
    translation: np.ndarray  # [x, y, z] in meters
    rotation: np.ndarray     # rotation_matrix (3x3)
    timestamp: float         # seconds (monotonic or wall clock)


class TransformTree:
    """Maintains a tree of coordinate frames and answers transform queries."""

    def __init__(self, buffer_duration: float = 30.0):
        """buffer_duration: how long to keep historical poses (for lookups)."""
        self._transforms: dict[str, list[TransformStamped]] = {}
        self._children: dict[str, str] = {}       # child → parent (static mapping)
        self._buffer_dur = buffer_duration
        self._root = "world"                       # default root frame

    # ------------------------------------------------------
    # Setting transforms
    # ------------------------------------------------------
    def set_static(
        self,
        frame: str,
        translation: list,
        rotation: list = None,  # quaternion [w, x, y, z] or None = identity
        frame_parent: str = "world",
    ):
        """Register a frame that does NOT move over time."""
        self._children[frame] = frame_parent
        rot_mat = self._quaternion_to_matrix(rotation)
        ts = TransformStamped(
            parent=frame_parent,
            child=frame,
            translation=np.array(translation, dtype=np.float64),
            rotation=rot_mat,
            timestamp=-np.inf,  # applies at all times
        )
        self._transforms.setdefault(frame, []).append(ts)

    def set_dynamic(
        self,
        frame: str,
        translation: list,
        rotation: list = None,
        frame_parent: str = "world",
        timestamp: Optional[float] = None,
    ):
        """Update (or add) a frame that CAN move over time.

        Call this every time a camera/arm reports a new pose.
        """
        if frame not in self._children:
            self._children[frame] = frame_parent
        else:
            frame_parent = self._children[frame]

        rot_mat = self._quaternion_to_matrix(rotation)
        ts = TransformStamped(
            parent=frame_parent,
            child=frame,
            translation=np.array(translation, dtype=np.float64),
            rotation=rot_mat,
            timestamp=timestamp or time.monotonic(),
        )
        self._transforms.setdefault(frame, []).append(ts)
        self._purge(frame)

    # ------------------------------------------------------
    # Querying
    # ------------------------------------------------------
    def lookup(
        self,
        target_frame: str,
        source_frame: str,
        time: Optional[float] = None,
    ) -> np.ndarray:
        """Return 4x4 homogeneous transform from source_frame to target_frame.

        Usage:
            T = tree.lookup("arm_tcp", "camera_top", time=t)
            pt_in_tcp = T[:3, :3] @ pt_in_cam + T[:3, 3]
        """
        # Decompose into world-relative chains
        T_sw = self._chain_to_root(source_frame, time)
        T_tw = self._chain_to_root(target_frame, time)
        return np.linalg.inv(T_tw) @ T_sw

    def lookup_latest(self, target_frame: str, source_frame: str) -> np.ndarray:
        """Convenience: lookup with the most recent timestamp available."""
        return self.lookup(target_frame, source_frame, time=None)

    # ------------------------------------------------------
    # Helpers
    # ------------------------------------------------------
    def _chain_to_root(self, frame: str, time: Optional[float]) -> np.ndarray:
        """Build 4x4 transform from `frame` up to the root (world)."""
        if frame not in self._children:
            raise KeyError(f"Unknown frame: {frame}")

        T_total = np.eye(4)
        current = frame
        visited = set()

        while current != self._root:
            if current in visited:
                raise RuntimeError(f"Cycle detected in TF tree at frame {current}")
            visited.add(current)

            # Find the best transform for this frame at the given time
            ts = self._get_best_ts(current, time)
            T_local = self._to_homogeneous(ts)

            # Parent-relative transform prepends: T_world→frame = T_parent→frame @ T_world→parent
            T_total = T_local @ T_total
            current = ts.parent

        return T_total

    def _get_best_ts(self, frame: str, time: Optional[float]) -> TransformStamped:
        """Get the transform for `frame` closest to `time`."""
        candidates = self._transforms.get(frame, [])
        if not candidates:
            raise KeyError(f"No transform registered for frame: {frame}")

        # Prefer static (t=-inf), then closest-in-time
        return max(candidates, key=lambda t: (0 if t.timestamp == -np.inf else 0, t.timestamp))

    def _purge(self, frame: str):
        """Remove old dynamic transforms beyond the buffer window."""
        now = time.monotonic()
        candidates = self._transforms.get(frame, [])
        static_only = [t for t in candidates if t.timestamp == -np.inf]
        self._transforms[frame] = static_only + [
            t for t in candidates
            if t.timestamp != -np.inf and (now - t.timestamp) < self._buffer_dur
        ]

    def _quaternion_to_matrix(self, q: list | None) -> np.ndarray:
        if q is None:
            return np.eye(3)
        q = np.array(q, dtype=np.float64)
        return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

    @staticmethod
    def _to_homogeneous(ts: TransformStamped) -> np.ndarray:
        """4x4 homogeneous matrix from a TransformStamped."""
        H = np.eye(4)
        H[:3, :3] = ts.rotation
        H[:3, 3] = ts.translation[:3]
        return H

    # ------------------------------------------------------
    # Convenience
    # ------------------------------------------------------
    def to_frame(self, point: np.ndarray, source: str, target: str, time: float = None) -> np.ndarray:
        """Transform a single point from source frame to target frame."""
        T = self.lookup(target, source, time)
        return (T[:3, :3] @ point + T[:3, 3])

    def add_camera_chain(self, name: str,
                         translation: list,
                         rotation: list | None = None,
                         parent: str = "world"):
        """Register a camera frame in the TF tree (static setup)."""
        self.set_static(f"{self._root}/{name}", translation, rotation, parent)

    def add_arm_chain(self,
                      base_translation: list,
                      base_rotation: list | None = None,
                      tcp_offset: list = None,
                      tcp_rotation: list | None = None,
                      name: str = "arm"):
        """Register arm_base and arm/tcp in one call."""
        self.set_static(f"{self._root}/{name}_base", base_translation, base_rotation, self._root)
        tcp_frame = f"{self._root}/{name}_base"
        self.set_static(
            f"{tcp_frame}/tcp",
            tcp_offset or [0, 0, 0],
            tcp_rotation,
            frame_parent=tcp_frame,
        )

    def print_tree(self):
        """Print the current frame tree."""
        for child, parent in sorted(self._children.items()):
            print(f"  {child} ← {parent}")
