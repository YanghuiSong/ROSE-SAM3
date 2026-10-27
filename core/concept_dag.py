"""Finite class-concept graph with cycle checks and shared nodes."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import re
from typing import Dict, Iterable, List


def normalize_phrase(text: str) -> str:
    text = text.casefold().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class ConceptNode:
    node_id: str
    class_id: int
    class_name: str
    phrase: str
    normalized_phrase: str
    relation: str
    depth: int
    parent_ids: List[str] = field(default_factory=list)
    status: str = "unseen"
    metrics: Dict = field(default_factory=dict)
    prune_reasons: List[str] = field(default_factory=list)


class ConceptDAG:
    def __init__(self) -> None:
        self.nodes: Dict[str, ConceptNode] = {}
        self.edges: Dict[str, List[str]] = {}
        self.aliases: Dict[str, str] = {}

    def _key(self, class_name: str, phrase: str) -> str:
        return f"{class_name}:{normalize_phrase(phrase)}"

    def _would_cycle(self, parent_id: str, child_id: str) -> bool:
        todo = [child_id]
        seen = set()
        while todo:
            cur = todo.pop()
            if cur == parent_id:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            todo.extend(self.edges.get(cur, []))
        return False

    def add_node(self, node: ConceptNode) -> str:
        existing = self._key(node.class_name, node.phrase)
        if existing in self.nodes:
            self.aliases[node.node_id] = existing
            self.nodes[existing].parent_ids = sorted(set(self.nodes[existing].parent_ids + node.parent_ids))
            return existing
        node.node_id = existing
        node.normalized_phrase = normalize_phrase(node.phrase)
        self.nodes[existing] = node
        self.edges.setdefault(existing, [])
        return existing

    def add_edge(self, parent_id: str, child_id: str) -> None:
        parent_id = self.aliases.get(parent_id, parent_id)
        child_id = self.aliases.get(child_id, child_id)
        if parent_id == child_id or self._would_cycle(parent_id, child_id):
            raise ValueError(f"edge would create a cycle: {parent_id} -> {child_id}")
        if child_id not in self.edges.setdefault(parent_id, []):
            self.edges[parent_id].append(child_id)
        if parent_id not in self.nodes[child_id].parent_ids:
            self.nodes[child_id].parent_ids.append(parent_id)
            self.nodes[child_id].parent_ids.sort()

    def descendants(self, node_id: str) -> List[str]:
        out, todo, seen = [], [node_id], set()
        while todo:
            cur = todo.pop()
            for nxt in self.edges.get(cur, []):
                if nxt not in seen:
                    seen.add(nxt); out.append(nxt); todo.append(nxt)
        return out


    def serialize(self) -> Dict:
        return {
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": {k: sorted(v) for k, v in self.edges.items()},
            "merged_aliases": dict(sorted(self.aliases.items())),
        }


def infer_relation(class_name: str, phrase: str, root: str) -> str:
    p = normalize_phrase(phrase)
    if p == normalize_phrase(root):
        return "native"
    if p in {"house", "bareland", "soil", "tree"}:
        return "synonym"
    if class_name == "water" and p in {"water body", "river", "pond", "lake", "continuous water surface", "smooth water region", "dark water area"}:
        return "narrower"
    if class_name == "forest" and p in {"woodland", "forest canopy", "tree covered region", "continuous tree cover", "dense forest", "natural woodland"}:
        return "narrower"
    if any(t in p for t in ("appearance", "red ", "dark ", "yellow ", "gray ", "smooth ", "dense ", "sparse ", "rectangular ")):
        return "appearance"
    if any(t in p for t in ("surface", "area", "region", "land", "field", "parcel", "patch", "corridor", "network", "structure", "roof", "canopy", "crown")):
        return "narrower"
    return "scene_context"


def build_finite_concept_dag(candidate_sets: Dict[str, List[str]], class_order: Iterable[str]) -> ConceptDAG:
    dag = ConceptDAG()
    class_order = list(class_order)
    for ci, cls in enumerate(class_order):
        phrases = list(dict.fromkeys(candidate_sets[cls]))
        root = phrases[0]
        for phrase in phrases:
            relation = infer_relation(cls, phrase, root)
            node = ConceptNode(
                node_id=f"{cls}:{normalize_phrase(phrase)}", class_id=ci,
                class_name=cls, phrase=phrase,
                normalized_phrase=normalize_phrase(phrase), relation=relation,
                depth=0 if phrase == root else 1,
                status="native" if phrase in phrases[:1] else "unseen")
            dag.add_node(node)
        root_id = dag._key(cls, root)
        for phrase in phrases[1:]:
            child = dag._key(cls, phrase)
            parent = root_id
            # Refinements may share multiple semantic parents.
            if cls == "forest" and phrase == "tree covered region":
                parent = dag._key(cls, "forest canopy")
                dag.add_edge(root_id, child)
                dag.add_edge(dag._key(cls, "tree"), child)
            elif cls == "water" and phrase in {"river", "pond", "lake"}:
                parent = dag._key(cls, "water body")
            elif cls == "forest" and phrase in {"woodland", "forest canopy", "dense forest", "natural woodland", "continuous tree cover"}:
                parent = root_id
            # Link to the root when an intermediate concept is absent.
            if parent not in dag.nodes:
                parent = root_id
            dag.add_edge(parent, child)
            dag.nodes[child].depth = min(dag.nodes[child].depth, dag.nodes[parent].depth + 1)
    return dag
