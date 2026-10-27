"""Generic Pareto archive and deterministic best-first DAG traversal."""
from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Dict, Iterable, List, Tuple

EPS = 1e-8

OBJECTIVES = (
    "pseudo_reference_fidelity", "domain_matchability",
    "visual_concept_agreement", "native_core_preservation",
    "min_competitor_protection", "partition_robustness",
)


def vector(metrics: Dict) -> Tuple[float, ...]:
    return tuple(float(metrics.get(k, 0.0)) for k in OBJECTIVES)


def dominates(a: Dict, b: Dict) -> bool:
    va, vb = vector(a), vector(b)
    return all(x >= y - EPS for x, y in zip(va, vb)) and any(x > y + EPS for x, y in zip(va, vb))




class ParetoArchive:
    def __init__(self):
        self.nodes: List = []

    def insert(self, node) -> bool:
        if any(dominates(old.metrics, node.metrics) for old in self.nodes):
            node.status = "dominated"; node.prune_reasons.append("pareto_dominated")
            return False
        kept = []
        for old in self.nodes:
            if dominates(node.metrics, old.metrics):
                old.status = "dominated"; old.prune_reasons.append("pareto_dominated")
            else:
                kept.append(old)
        node.status = "pareto"
        self.nodes = kept + [node]
        return True

    def as_list(self):
        return sorted(self.nodes, key=lambda n: (n.class_id, n.phrase.casefold()))


@dataclass
class SearchResult:
    archives: Dict[str, ParetoArchive]
    expanded: List[str]
    branch_pruned: List[str]
    bound_pruned: List[str]


def best_first_search(dag, metrics_by_node: Dict[str, Dict], safe_ids: set,
                      roots: Iterable[str]) -> SearchResult:
    """Traverse cached node metrics with optimistic descendant bounds."""
    archives = {cls: ParetoArchive() for cls in {dag.nodes[x].class_name for x in dag.nodes}}
    expanded, branch_pruned, bound_pruned = [], [], []
    roots = list(roots)
    def ub(nid):
        ids = [nid] + dag.descendants(nid)
        vals = [metrics_by_node[i] for i in ids if i in safe_ids and i in metrics_by_node]
        return tuple(max((vector(v)[j] for v in vals), default=0.0) for j in range(len(OBJECTIVES)))
    def can_enter(bound, archive):
        if not archive.nodes:
            return True
        return not all(any(bound[j] <= vector(x.metrics)[j] + EPS for x in archive.nodes) for j in range(len(OBJECTIVES)))
    queue = []
    seq = 0
    for rid in roots:
        heapq.heappush(queue, (-sum(ub(rid)), seq, rid)); seq += 1
    seen = set()
    while queue:
        _, _, nid = heapq.heappop(queue)
        if nid in seen:
            continue
        seen.add(nid)
        node = dag.nodes[nid]
        if nid not in safe_ids:
            node.status = "unsafe"; node.prune_reasons.append("competition_unsafe")
            branch_pruned.append(nid)
            continue
        archive = archives[node.class_name]
        if not can_enter(ub(nid), archive) and node.relation != "native":
            node.status = "branch_pruned"; node.prune_reasons.append("upper_bound_dominated")
            bound_pruned.append(nid)
            continue
        expanded.append(nid)
        archive.insert(node)
        for child in dag.edges.get(nid, []):
            if child not in seen:
                heapq.heappush(queue, (-sum(ub(child)), seq, child)); seq += 1
    # Native roots are conservative fallbacks even if another node dominates.
    for rid in roots:
        node = dag.nodes[rid]
        if not any(x.node_id == rid for x in archives[node.class_name].nodes):
            node.status = "native"
            archives[node.class_name].nodes.append(node)
    return SearchResult(archives, expanded, branch_pruned, bound_pruned)
