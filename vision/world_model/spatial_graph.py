"""
spatial_graph.py — Spatial knowledge graph for object relationships.

Builds a NetworkX graph from ObjectDB entries, dynamically computing
spatial relationships (near, adjacent, contains, left_of, right_of)
to answer relational queries like "what is next to the table?"
"""
import numpy as np
import networkx as nx
from typing import Any


class SpatialGraph:
    """Knowledge graph of spatial + temporal relationships between objects.

    Nodes: ObjectDB entries, robot pose, named places.
    Edges: Computed dynamically on query — always up to date.
    """

    def __init__(self, near_threshold: float = 3.0):
        self._graph = nx.Graph()
        self._near_threshold = near_threshold
        self._robot_pose = None
        self._places = {}

    def clear(self):
        self._graph.clear()
        self._robot_pose = None

    def add_object(self, obj) -> None:
        """Add or update an object node."""
        self._graph.add_node(obj.object_id,
                             class_name=obj.class_name,
                             position=obj.position_world[:2],
                             confidence=obj.confidence,
                             timestamp=obj.timestamp,
                             obb_width=obj.obb_width,
                             obb_depth=obj.obb_depth,
                             obb_angle=obj.obb_angle)

    def remove_object(self, object_id: str) -> None:
        if self._graph.has_node(object_id):
            self._graph.remove_node(object_id)

    def update_robot_pose(self, pose: tuple) -> None:
        self._robot_pose = pose

    def add_place(self, name: str, position: tuple) -> None:
        self._places[name] = position
        self._graph.add_node(f"place_{name}",
                             class_name="place",
                             position=position[:2],
                             is_place=True)

    def remove_place(self, name: str) -> None:
        node = f"place_{name}"
        if self._graph.has_node(node):
            self._graph.remove_node(node)
        self._places.pop(name, None)

    def _rebuild_edges(self):
        """Recompute all spatial edges from current positions."""
        self._graph.clear_edges()
        nodes = list(self._graph.nodes(data=True))

        for i, (nid_a, data_a) in enumerate(nodes):
            if data_a.get("is_place"):
                continue
            pos_a = np.array(data_a["position"])
            obb_a = (data_a.get("obb_width", 0), data_a.get("obb_depth", 0),
                     data_a.get("obb_angle", 0))

            for j in range(i + 1, len(nodes)):
                nid_b, data_b = nodes[j]
                pos_b = np.array(data_b["position"])

                dist = float(np.linalg.norm(pos_a - pos_b))

                # Near (distance threshold)
                if dist < self._near_threshold:
                    self._graph.add_edge(nid_a, nid_b,
                                         relation="spatial_near",
                                         distance=round(dist, 3))

                # Adjacent (OBB edges touch or overlap)
                if self._obbs_overlap(data_a, data_b, dist):
                    self._graph.add_edge(nid_a, nid_b,
                                         relation="spatial_adjacent",
                                         distance=0.0)

                # Containment (B inside A's OBB)
                if self._contains(data_a, data_b):
                    self._graph.add_edge(nid_a, nid_b,
                                         relation="spatial_contains",
                                         distance=0.0)

                # Directional (relative to robot heading)
                if self._robot_pose:
                    direction = self._direction_to_robot(pos_a, pos_b)
                    if direction:
                        self._graph.add_edge(nid_a, nid_b,
                                             relation=direction,
                                             distance=round(dist, 3))

    def _obbs_overlap(self, data_a: dict, data_b: dict, dist: float) -> bool:
        """Check if two OBBs touch or overlap."""
        if not data_a.get("obb_width") or not data_b.get("obb_width"):
            return dist < 0.5
        return dist < (data_a["obb_width"] + data_b["obb_width"]) / 2

    def _contains(self, outer: dict, inner: dict) -> bool:
        """Check if inner object is inside outer's OBB (e.g., bottle on table)."""
        if not outer.get("obb_width") or not outer.get("obb_depth"):
            return False
        pos_diff = np.array(inner["position"]) - np.array(outer["position"])
        return float(np.linalg.norm(pos_diff)) < min(outer["obb_width"], outer["obb_depth"]) / 2

    def _direction_to_robot(self, pos: np.ndarray, other: np.ndarray) -> str:
        """Classify direction of 'other' relative to robot's heading."""
        if self._robot_pose is None:
            return ""
        rx, ry, rtheta = self._robot_pose
        robot_pos = np.array([rx, ry])
        to_other = other - robot_pos
        to_obj = pos - robot_pos

        # Simplified: classify based on angle from robot
        angle = np.arctan2(to_other[1], to_other[0]) - rtheta
        angle = np.degrees(angle) % 360

        if angle < 45 or angle > 315:
            return "in_front_of"
        elif 135 < angle < 225:
            return "behind"
        elif 45 <= angle <= 135:
            return "left_of"
        else:
            return "right_of"

    def query(self, expression: str) -> list[dict]:
        """Answer a natural-language spatial query.

        Supported patterns:
          "nearest object to <target>"
          "objects within Nm of <target>"
          "what is near <target>"
          "objects on top of <target>"
          "what is to the left of the robot"
          "largest object"

        Args:
            expression: A description of what to find.

        Returns:
            List of dicts with node_id, class_name, relation, distance.
        """
        self._rebuild_edges()
        expr_lower = expression.lower()

        targets = []
        # Extract target name from query
        for node, data in self._graph.nodes(data=True):
            cn = data.get("class_name", "").lower()
            if cn and cn in expr_lower and not data.get("is_place"):
                targets.append((node, data))

        if expr_lower.startswith("nearest") and targets:
            target_node, target_data = targets[0]
            pos = np.array(target_data["position"])
            return self._nearest_to(pos, exclude=target_node)

        if "within" in expr_lower:
            import re
            m = re.search(r"within\s+([\d.]+)m", expr_lower)
            radius = float(m.group(1)) if m else self._near_threshold
            if targets:
                t_node, t_data = targets[0]
                pos = np.array(t_data["position"])
                return self._within_radius(pos, radius, exclude=t_node)

        if "near" in expr_lower and targets:
            t_node, t_data = targets[0]
            pos = np.array(t_data["position"])
            return self._nearest_to(pos, exclude=t_node, max_results=5)

        if ("on top" in expr_lower or "contain" in expr_lower) and targets:
            t_node, t_data = targets[0]
            results = []
            for nid, data in self._graph.nodes(data=True):
                if nid == t_node:
                    continue
                if self._contains(t_data, data):
                    results.append({
                        "node_id": nid,
                        "class_name": data.get("class_name", ""),
                        "relation": "spatial_contains",
                        "distance": 0.0,
                    })
            return results

        if "left of" in expr_lower or "right of" in expr_lower:
            direction = "left_of" if "left" in expr_lower else "right_of"
            return self._filter_by_relation(direction)

        if expr_lower.startswith("largest"):
            nodes = [(n, d) for n, d in self._graph.nodes(data=True)
                     if d.get("obb_width", 0) > 0]
            if not nodes:
                return [{"message": "No object dimensions available."}]
            nodes.sort(key=lambda x: x[1].get("obb_width", 0), reverse=True)
            nid, data = nodes[0]
            return [{"node_id": nid, "class_name": data.get("class_name", ""),
                     "width": data.get("obb_width", 0)}]

        return [{"message": f"Could not parse query: {expression}",
                 "hint": ("Try: 'nearest object to table', 'what is near the sofa', "
                         "'objects within 2m of door', 'what is on top of table'")}]

    def _nearest_to(self, pos: np.ndarray, exclude: str = None,
                    max_results: int = 3) -> list[dict]:
        results = []
        for nid, data in self._graph.nodes(data=True):
            if nid == exclude or data.get("is_place"):
                continue
            dist = float(np.linalg.norm(np.array(data["position"]) - pos))
            results.append({
                "node_id": nid,
                "class_name": data.get("class_name", ""),
                "distance": round(dist, 3),
                "confidence": data.get("confidence", 0),
            })
        results.sort(key=lambda x: x["distance"])
        return results[:max_results]

    def _within_radius(self, center: np.ndarray, radius: float,
                       exclude: str = None) -> list[dict]:
        results = []
        for nid, data in self._graph.nodes(data=True):
            if nid == exclude or data.get("is_place"):
                continue
            dist = float(np.linalg.norm(np.array(data["position"]) - center))
            if dist <= radius:
                results.append({
                    "node_id": nid,
                    "class_name": data.get("class_name", ""),
                    "distance": round(dist, 3),
                })
        return results

    def _filter_by_relation(self, relation: str) -> list[dict]:
        results = []
        for u, v, data in self._graph.edges(data=True):
            if data.get("relation") == relation:
                results.append({
                    "node_a": self._graph.nodes[u].get("class_name", u),
                    "node_b": self._graph.nodes[v].get("class_name", v),
                    "relation": relation,
                })
        return results

    def to_text(self) -> str:
        """Human-readable summary of the graph for LLM context."""
        self._rebuild_edges()
        nodes = [d.get("class_name", "?") for _, d in self._graph.nodes(data=True)
                 if not d.get("is_place")]
        edges_descriptions = []
        for u, v, data in self._graph.edges(data=True):
            cu = self._graph.nodes[u].get("class_name", u)
            cv = self._graph.nodes[v].get("class_name", v)
            rel = data.get("relation", "connected")
            dist = data.get("distance")
            if dist:
                edges_descriptions.append(f"{cu} {rel} {cv} ({dist}m)")
            else:
                edges_descriptions.append(f"{cu} {rel} {cv}")

        lines = [f"SpatialGraph: {len(nodes)} objects, {len(edges_descriptions)} relationships"]
        lines.extend(f"  {e}" for e in edges_descriptions[:20])
        return "\n".join(lines)
