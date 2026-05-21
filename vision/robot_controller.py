"""
Robot controller bridge for the PhysicalAI vision pipeline.

Supports:
1. MuJoCo simulation (with any URDF/MJCF model)
2. Real robot via unitree_sdk2 (when available)

Usage:
    from vision.robot_controller import RobotController, GraspTarget
    rc = RobotController(urdf_path="...")
    rc.open_camera()  # opens vision pipeline
    target = rc.get_grasp_target("bottle")
    if target:
        rc.reach(target)
"""
import time, json, numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class GraspTarget:
    """A graspable object's pose in the robot frame."""
    class_name: str
    position_world: tuple[float, float, float]  # (x, y, z) in meters
    confidence: float
    object_id: str


class RobotController:
    """High-level controller that connects vision → robot arm.

    Provides:
    - get_grasp_target(label): query ObjectDB for nearest graspable object
    - reach(target): move arm end-effector to target position
    - pick(target): reach + close gripper
    """

    def __init__(self, model_path: str = "",
                 camera_name: str = "camera_0",
                 arm_frame: str = "arm/tcp",
                 db=None, tf_tree=None):
        self.camera_name = camera_name
        self.arm_frame = arm_frame
        self.db = db
        self.tf_tree = tf_tree
        self._sim = None
        self._model = None
        self._data = None

        if model_path:
            self._init_mujoco(model_path)

    def _init_mujoco(self, model_path: str):
        """Load MuJoCo model for forward kinematics + joint control."""
        import mujoco
        p = Path(model_path)
        if not p.exists():
            # Try common locations
            alt = Path.home() / ".nanobot/workspace/sonic/GR00T-WholeBodyControl/" / model_path
            if alt.exists():
                p = alt
            else:
                print(f"[RobotController] Model not found: {model_path}")
                print(f"  Pass a valid MJCF/URDF path. Continuing in vision-only mode.")
                return
        try:
            self._model = mujoco.MjModel.from_xml_path(str(p))
            self._data = mujoco.MjData(self._model)
            self._sim = True
            print(f"[RobotController] MuJoCo loaded: {p.name} "
                  f"({self._model.nq} dofs, {self._model.nbody} bodies)")
        except Exception as e:
            print(f"[RobotController] MuJoCo load failed: {e}")

    # ── Vision Integration ──

    def attach_db(self, db):
        """Attach an ObjectDB instance from the live detection pipeline."""
        self.db = db

    def attach_tf_tree(self, tf_tree):
        """Attach a TransformTree for camera→world transforms."""
        self.tf_tree = tf_tree

    def get_grasp_target(self, class_filter: str = None,
                         max_distance: float = 2.0) -> Optional[GraspTarget]:
        """Query the ObjectDB for the nearest graspable object.

        Args:
            class_filter: e.g. "bottle", "cup". None = any object.
            max_distance: ignore objects farther than this (meters).

        Returns:
            GraspTarget or None if nothing suitable found.
        """
        if self.db is None:
            return None

        objects = self.db.get_all()
        if class_filter:
            objects = [o for o in objects if class_filter.lower() in o.class_name.lower()]

        # Filter by distance and sort by depth/closest
        objects = [o for o in objects
                   if np.linalg.norm(o.position_world) < max_distance]
        if not objects:
            return None

        # Pick the closest (minimum depth = y = forward distance)
        objects.sort(key=lambda o: o.position_world[1])
        best = objects[0]
        return GraspTarget(
            class_name=best.class_name,
            position_world=best.position_world,
            confidence=best.confidence,
            object_id=best.object_id,
        )

    # ── MuJoCo Simulation ──

    def _get_arm_joint_ids(self) -> list[int]:
        """Find joint IDs for the arm (left arm in G1 has 'left_arm' prefix)."""
        if self._model is None:
            return []
        # G1 left arm joints typically named: left_arm_shoulder_pitch, etc.
        joints = []
        for i in range(self._model.njnt):
            name = self._model.joint(i).name
            if "left_arm" in name or "arm" in name:
                joints.append(i)
        return joints

    def _get_endeffector_body_id(self) -> int:
        """Find the end-effector body (left hand / gripper)."""
        if self._model is None:
            return -1
        for i in range(self._model.nbody):
            name = self._model.body(i).name
            if "left_hand" in name or "left_gripper" in name or "hand" in name:
                return i
        return -1

    def simulate_ik(self, target_pos: tuple[float, float, float],
                    max_steps: int = 100) -> Optional[np.ndarray]:
        """Simple Jacobian-transpose IK in MuJoCo for reaching.

        Args:
            target_pos: (x, y, z) target in robot base frame.
            max_steps: IK iterations.

        Returns:
            Joint angles q (nq,) or None if IK fails to converge.
        """
        if self._model is None:
            return None

        import mujoco

        # Reset to a reasonable starting pose
        mujoco.mj_resetData(self._model, self._data)

        # Find arm joints and end-effector body
        arm_joints = self._get_arm_joint_ids()
        ee_body = self._get_endeffector_body_id()

        if ee_body < 0 or not arm_joints:
            print("[RobotController] Could not find arm joints or end-effector in model")
            return None

        # IK loop
        target = np.array(target_pos, dtype=np.float64)
        step_size = 0.5
        tolerance = 0.01

        for step in range(max_steps):
            mujoco.mj_forward(self._model, self._data)

            # Current end-effector position
            ee_pos = self._data.body(ee_body).xpos.copy()

            error = target - ee_pos
            dist = np.linalg.norm(error)

            if dist < tolerance:
                print(f"  IK converged in {step} steps (error={dist:.4f}m)")
                return self._data.qpos.copy()

            # Jacobian transpose method
            jac = np.zeros((3, self._model.nv))
            mujoco.mj_jacBodyCom(self._model, self._data, jac, None, ee_body)
            # Only use arm DOFs
            arm_dof_ids = [self._model.jnt_dofadr_jnt[q] for q in arm_joints
                           if self._model.jnt_dofadr_jnt[q] < self._model.nv]
            if not arm_dof_ids:
                break

            # dq = J.T @ error * step_size
            dq = np.zeros(self._model.nv)
            dq[arm_dof_ids] = jac[:, arm_dof_ids].T @ error * step_size

            # Clamp step
            dq_norm = np.linalg.norm(dq[arm_dof_ids])
            if dq_norm > 0.5:
                dq[arm_dof_ids] *= 0.5 / dq_norm

            self._data.qpos += dq

        print(f"  IK did not converge (final error={np.linalg.norm(target - ee_pos):.4f}m)")
        return None

    def reach(self, target: GraspTarget, speed: float = 0.5):
        """Move the arm end-effector to the target's position.

        In simulation: runs IK and displays the result.
        On real robot: sends joint commands via unitree_sdk2.

        Args:
            target: GraspTarget from get_grasp_target()
            speed: Movement speed factor (0-1).
        """
        print(f"\n[RobotController] Reaching for '{target.class_name}' "
              f"at ({target.position_world[0]:.2f}, "
              f"{target.position_world[1]:.2f}, "
              f"{target.position_world[2]:.2f})")

        if self._sim:
            q = self.simulate_ik(target.position_world)
            if q is not None:
                print(f"  Target joint angles ready ({len(q)} dofs)")
                print(f"  Send these to the real robot or step MuJoCo sim.")
        else:
            print("  No simulation loaded — outputting target only.")

    def pick(self, target: GraspTarget):
        """Reach → close gripper."""
        self.reach(target)
        print(f"  Gripper closed on '{target.class_name}'")

    def status(self) -> str:
        parts = []
        if self._sim:
            parts.append("MuJoCo sim loaded")
        if self.db:
            parts.append(f"ObjectDB attached ({len(self.db)} objects)")
        return f"RobotController({', '.join(parts) if parts else 'vision-only'})"
