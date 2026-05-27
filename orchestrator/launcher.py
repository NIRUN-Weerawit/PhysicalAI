"""
launcher.py — Launch and manage SLAM + Nav2 subprocesses.

Provides a NavLauncher class that starts/stops slam_toolbox (online_async)
and Nav2 bringup as managed subprocesses, with health checking.
"""
import subprocess, os, signal, time, threading
from typing import Optional, List

SLAM_PARAMS = os.path.join(
    os.path.dirname(__file__),
    "..",
    "orchestrator",
    "nav2_params_tb3.yaml",
)
# slam_toolbox has its own config file
SLAM_TOOLBOX_CONFIG = (
    "/opt/ros/humble/share/slam_toolbox/config/mapper_params_online_async.yaml"
)


class NavLauncher:
    """Manages slam_toolbox + Nav2 lifecycle."""

    def __init__(self, env: Optional[dict] = None):
        self._env = env or os.environ.copy()
        self._procs: List[subprocess.Popen] = []
        self._names: List[str] = []

    def start_slam(self) -> bool:
        """Launch slam_toolbox online_async SLAM node."""
        cmd = [
            "ros2",
            "run",
            "slam_toolbox",
            "async_slam_toolbox_node",
            "--ros-args",
            "--params-file",
            SLAM_TOOLBOX_CONFIG,
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                env=self._env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._procs.append(proc)
            self._names.append("slam_toolbox")
            return True
        except Exception as e:
            print(f"[NavLauncher] Failed to start slam_toolbox: {e}")
            return False

    def start_nav2(self) -> bool:
        """Launch Nav2 bringup (navigation_launch.py with our params)."""
        cmd = [
            "ros2",
            "launch",
            "nav2_bringup",
            "navigation_launch.py",
            f"use_sim_time:=True",
            f"params_file:={os.path.abspath(SLAM_PARAMS)}",
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                env=self._env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._procs.append(proc)
            self._names.append("nav2")
            return True
        except Exception as e:
            print(f"[NavLauncher] Failed to start Nav2: {e}")
            return False

    def poll_health(self) -> dict:
        """Check if all processes are still alive."""
        status = {}
        for i, proc in enumerate(self._procs):
            name = self._names[i] if i < len(self._names) else f"proc_{i}"
            ret = proc.poll()
            if ret is None:
                status[name] = "running"
            else:
                status[name] = f"exited({ret})"
        return status

    def stop(self, timeout: float = 5.0):
        """Gracefully stop all processes, force-kill if needed."""
        for proc in self._procs:
            try:
                proc.terminate()
            except Exception:
                pass

        deadline = time.monotonic() + timeout
        for proc in self._procs:
            remaining = max(0, deadline - time.monotonic())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                    proc.wait(timeout=2)
                except Exception:
                    pass

        self._procs.clear()
        self._names.clear()

    def __del__(self):
        self.stop()
