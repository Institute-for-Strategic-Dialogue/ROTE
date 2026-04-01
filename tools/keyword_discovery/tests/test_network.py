"""Tests for graph construction and community detection."""

import pytest
import networkx as nx

from tools.keyword_discovery.discovery.network import (
    build_cooccurrence_graph,
    export_graphml,
    NetworkDiscovery,
)


class TestGraphConstruction:
    def test_builds_graph(self, small_corpus):
        G = build_cooccurrence_graph(small_corpus, ["globalist"], min_df=1, max_nodes=500)
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0

    def test_seed_in_graph(self, small_corpus):
        G = build_cooccurrence_graph(small_corpus, ["globalist"], min_df=1, max_nodes=500)
        assert "globalist" in G.nodes()

    def test_edges_have_weight(self, small_corpus):
        G = build_cooccurrence_graph(small_corpus, ["globalist"], min_df=1, max_nodes=500)
        for u, v, data in G.edges(data=True):
            assert "weight" in data
            assert data["weight"] > 0

    def test_ego_graph_limits_size(self, small_corpus):
        G = build_cooccurrence_graph(
            small_corpus, ["globalist"], min_df=1, max_nodes=10, hops=1
        )
        assert G.number_of_nodes() <= 20  # Soft check, ego-graph may exceed slightly


class TestGraphML:
    def test_export_produces_bytes(self, small_corpus):
        G = build_cooccurrence_graph(small_corpus, ["globalist"], min_df=1, max_nodes=500)
        data = export_graphml(G)
        assert isinstance(data, bytes)
        assert b"graphml" in data.lower()


class TestNetworkDiscovery:
    def test_discovers_terms(self, small_corpus):
        disc = NetworkDiscovery(max_nodes=500)
        candidates = disc.discover(["globalist"], small_corpus, top_n=10, min_df=1)
        assert isinstance(candidates, list)
        assert all(c.source == "network" for c in candidates)

    def test_stores_graph(self, small_corpus):
        disc = NetworkDiscovery(max_nodes=500)
        disc.discover(["globalist"], small_corpus, top_n=10, min_df=1)
        assert disc.last_graph is not None
        assert isinstance(disc.last_graph, nx.Graph)

    def test_no_seeds_in_results(self, small_corpus):
        disc = NetworkDiscovery(max_nodes=500)
        candidates = disc.discover(["globalist", "cabal"], small_corpus, top_n=10, min_df=1)
        terms = {c.term for c in candidates}
        assert "globalist" not in terms
        assert "cabal" not in terms
