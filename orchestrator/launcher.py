"""
launcher.py — Launch and manage SLAM + Nav2 subprocesses.

Provides a NavLauncher class that starts/stops slam_toolbox (online_async)
and Nav2 bringup as managed subprocesses, with health checking.
On stop, kills the entire process group to prevent zombie children.
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
    """Manages slam_toolbox + Nav2 lifecycle.
    
    Spawns processes in their own process group (setsid) so stop()
    can kill the full tree with SIGTERM, not just the parent shell.
    """

    def __init__(self, env: Optional[dict] = None):
        self._env = env or os.environ.copy()
        self._procs: List[subprocess.Popen] = []
        self._names: List[str] = []
        # NOT registering signal handlers here — they conflict with rclpy's
        # built-in SIGINT/SIGTERM handling. Cleanup is done in main() finally.

    def _handle_signal(self, signum, frame):
        """Handle termination signals by cleaning up child processes.
        
        For SIGINT (Ctrl+C), raise KeyboardInterrupt after cleanup so
        rclpy.spin() and the main try/except can exit properly.
        """
        signame = signal.Signals(signum).name
        print(f"\n[NavLauncher] Caught {signame}, cleaning up child processes...",
              flush=True)
        self._sigterm_caught = True
        self.stop()
        # Re-raise so keyboard interrupt propagates
        if signum == signal.SIGINT:
            raise KeyboardInterrupt

    def _spawn(self, cmd: list, name: str) -> bool:
        """Spawn a process in a new process group for reliable kill.

        Uses setpgrp (not setsid/session) so ros2 launch's internal
        process management still works. killpg() can target the group.
        """
        try:
            proc = subprocess.Popen(
                cmd,
                env=self._env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                # Place in a new process group (same session) so killpg works
                preexec_fn=os.setpgrp,
            )
            self._procs.append(proc)
            self._names.append(name)
            return True
        except Exception as e:
            print(f"[NavLauncher] Failed to start {name}: {e}")
            return False

    def start_slam(self) -> bool:
        """Launch slam_toolbox online_async SLAM node."""
        return self._spawn([
            "ros2", "run", "slam_toolbox", "async_slam_toolbox_node",
            "--ros-args", "--params-file", SLAM_TOOLBOX_CONFIG,
        ], "slam_toolbox")

    def start_nav2(self) -> bool:
        """Launch Nav2 bringup (navigation_launch.py with our params)."""
        return self._spawn([
            "ros2", "launch", "nav2_bringup", "navigation_launch.py",
            "use_sim_time:=True",
            f"params_file:={os.path.abspath(SLAM_PARAMS)}",
        ], "nav2")

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
        """Gracefully stop all processes, force-kill full process groups if needed."""
        for proc in self._procs:
            try:
                pgid = os.getpgid(proc.pid)
                # SIGTERM the entire process group (parents + children)
                os.killpg(pgid, signal.SIGTERM)
            except Exception:
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
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
                    proc.wait(timeout=2)
                except Exception:
                    pass

        self._procs.clear()
        self._names.clear()

    def __del__(self):
        self.stop()
