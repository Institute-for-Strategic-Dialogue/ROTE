"""Graph-based keyword discovery via community detection and centrality."""

from __future__ import annotations

import io
import logging
import re
from typing import Any

import networkx as nx
import numpy as np

from ..corpus import Corpus
from ..models import Candidate
from .base import DiscoveryModule

logger = logging.getLogger(__name__)

# Cap per-document term count to avoid O(n^2) edge explosion
_MAX_TERMS_PER_DOC = 40


def _get_evidence(term: str, raw_texts: list[str], doc_indices: list[int], max_n: int = 5) -> list[str]:
    pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
    excerpts: list[str] = []
    for idx in doc_indices:
        if idx < len(raw_texts):
            text = raw_texts[idx]
            if pattern.search(text):
                snippet = text[:300].strip()
                if len(text) > 300:
                    snippet += "..."
                excerpts.append(snippet)
                if len(excerpts) >= max_n:
                    break
    return excerpts


def build_cooccurrence_graph(
    corpus: Corpus,
    seed_keywords: list[str],
    min_df: int = 5,
    max_nodes: int = 1000,
    hops: int = 2,
) -> nx.Graph:
    """Build a co-occurrence graph in the neighbourhood of seed keywords.

    Nodes are terms, edges are weighted by co-occurrence count.
    The graph is limited to the ego-graph of seed keywords at *hops* distance,
    keeping at most *max_nodes* nodes.
    """
    seed_set = {s.lower() for s in seed_keywords}

    # Gather candidate terms: seeds + top-freq vocab (capped)
    candidate_terms: set[str] = set()
    for s in seed_set:
        if s in corpus.doc_freq:
            candidate_terms.add(s)

    # Add top-frequency terms up to max_nodes
    sorted_vocab = sorted(corpus.doc_freq.items(), key=lambda x: x[1], reverse=True)
    for term, freq in sorted_vocab:
        if freq < min_df:
            break
        candidate_terms.add(term)
        if len(candidate_terms) >= max_nodes:
            break

    if not candidate_terms:
        return nx.Graph()

    # Build inverted index: doc -> terms (only terms in our candidate set)
    # Cap terms per doc to avoid quadratic blowup
    doc_to_terms: dict[int, list[str]] = {}
    for term in candidate_terms:
        for doc_idx in corpus.term_doc_index.get(term, []):
            dt = doc_to_terms.get(doc_idx)
            if dt is None:
                doc_to_terms[doc_idx] = [term]
            elif len(dt) < _MAX_TERMS_PER_DOC:
                dt.append(term)

    # Build edges from co-occurrence within documents
    edge_counts: dict[tuple[str, str], int] = {}
    for doc_terms in doc_to_terms.values():
        n = len(doc_terms)
        if n < 2:
            continue
        for i in range(n):
            for j in range(i + 1, n):
                a, b = doc_terms[i], doc_terms[j]
                key = (a, b) if a < b else (b, a)
                edge_counts[key] = edge_counts.get(key, 0) + 1

    G = nx.Graph()
    G.add_nodes_from(candidate_terms)
    for (a, b), count in edge_counts.items():
        G.add_edge(a, b, weight=count)

    # Extract ego-graph around seed nodes
    seed_nodes = [s for s in seed_set if s in G]
    if not seed_nodes:
        return G

    ego_nodes: set[str] = set()
    for seed_node in seed_nodes:
        try:
            ego = nx.ego_graph(G, seed_node, radius=hops)
            ego_nodes.update(ego.nodes())
        except nx.NetworkXError:
            continue

    # Trim to max_nodes by keeping highest-degree nodes
    if len(ego_nodes) > max_nodes:
        non_seed = ego_nodes - seed_set
        degrees = {n: G.degree(n) for n in non_seed}
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[: max_nodes - len(seed_set & ego_nodes)]
        ego_nodes = (seed_set & ego_nodes) | set(top_nodes)

    subgraph = G.subgraph(ego_nodes).copy()
    logger.info("Network graph: %d nodes, %d edges", subgraph.number_of_nodes(), subgraph.number_of_edges())
    return subgraph


def export_graphml(G: nx.Graph) -> bytes:
    """Export *G* as GraphML bytes."""
    buf = io.BytesIO()
    nx.write_graphml(G, buf)
    return buf.getvalue()


class NetworkDiscovery(DiscoveryModule):
    """Discover terms via graph community detection and centrality measures."""

    def __init__(self, hops: int = 2, max_nodes: int = 1000):
        self.hops = hops
        self.max_nodes = max_nodes
        self.last_graph: nx.Graph | None = None

    def discover(
        self,
        seed_keywords: list[str],
        corpus: Corpus,
        top_n: int = 50,
        min_df: int = 5,
    ) -> list[Candidate]:
        """Build a co-occurrence graph and score terms by centrality."""
        G = build_cooccurrence_graph(
            corpus, seed_keywords, min_df=min_df,
            max_nodes=self.max_nodes, hops=self.hops,
        )
        self.last_graph = G

        if G.number_of_nodes() < 3:
            return []

        seed_set = {s.lower() for s in seed_keywords}

        # Community detection via Louvain
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G, random_state=42)
        except ImportError:
            logger.warning("python-louvain not installed; skipping community detection")
            partition = {n: 0 for n in G.nodes()}

        # Identify seed communities
        seed_communities: set[int] = set()
        for seed in seed_set:
            if seed in partition:
                seed_communities.add(partition[seed])

        # Assign community_id to all nodes
        for node in G.nodes():
            G.nodes[node]["community"] = partition.get(node, -1)

        # Personalized PageRank seeded from seed nodes
        personalization = {n: 0.0 for n in G.nodes()}
        for seed in seed_set:
            if seed in personalization:
                personalization[seed] = 1.0
        total = sum(personalization.values())
        if total > 0:
            personalization = {k: v / total for k, v in personalization.items()}
            try:
                pagerank = nx.pagerank(G, personalization=personalization, max_iter=100)
            except nx.PowerIterationFailedConvergence:
                pagerank = {n: 0.0 for n in G.nodes()}
        else:
            pagerank = {n: 0.0 for n in G.nodes()}

        # Betweenness centrality (use k-sampling for large graphs)
        n_nodes = G.number_of_nodes()
        if n_nodes > 500:
            betweenness = nx.betweenness_centrality(G, weight="weight", k=min(200, n_nodes))
        else:
            betweenness = nx.betweenness_centrality(G, weight="weight")

        # Score non-seed nodes
        scored: list[tuple[str, float, int, bool]] = []
        pr_vals = list(pagerank.values())
        bt_vals = list(betweenness.values())
        pr_max = max(pr_vals) if pr_vals else 1.0
        bt_max = max(bt_vals) if bt_vals else 1.0

        for node in G.nodes():
            if node in seed_set:
                continue
            comm = partition.get(node, -1)
            if comm not in seed_communities:
                neighbours_in_seed_comm = any(
                    partition.get(nbr, -1) in seed_communities for nbr in G.neighbors(node)
                )
                if not neighbours_in_seed_comm:
                    continue

            pr = pagerank.get(node, 0) / pr_max if pr_max > 0 else 0
            bt = betweenness.get(node, 0) / bt_max if bt_max > 0 else 0
            score = 0.5 * bt + 0.5 * pr

            neighbor_comms = {partition.get(nbr, -1) for nbr in G.neighbors(node)}
            is_bridge = len(neighbor_comms & seed_communities) > 1

            scored.append((node, score, comm, is_bridge))

        scored.sort(key=lambda x: x[1], reverse=True)

        candidates: list[Candidate] = []
        for term, score, comm, is_bridge in scored[:top_n]:
            if score <= 0:
                continue
            doc_indices = corpus.term_doc_index.get(term, [])
            evidence = _get_evidence(term, corpus.raw_texts, doc_indices)
            doc_count = corpus.doc_freq.get(term, 0)

            ctx = f"Graph centrality (community {comm})"
            if is_bridge:
                ctx += " - BRIDGE term connecting multiple communities"

            candidates.append(
                Candidate(
                    term=term,
                    score=round(min(1.0, score), 4),
                    source="network",
                    sources=["network"],
                    context=ctx,
                    evidence=evidence,
                    community_id=comm,
                    doc_count=doc_count,
                )
            )

        return candidates
