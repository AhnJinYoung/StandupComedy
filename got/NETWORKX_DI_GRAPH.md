# Quick Guide: `networkx.DiGraph` (as used here)

The GoT implementation uses `networkx.DiGraph` to store the reasoning graph. Below are the most relevant concepts and APIs for reading or extending the code.

- **Create a directed graph**
  - `import networkx as nx`
  - `g = nx.DiGraph()`

- **Add nodes (thoughts) with attributes**
  - `g.add_node(node_id, content="text", type="setup", score=0, metadata={})`
  - Nodes are keyed by arbitrary hashable IDs (strings/UUIDs in this repo).

- **Add directed edges (dependencies)**
  - `g.add_edge(parent_id, child_id)` creates an arrow from parent → child.
  - Raises `KeyError` if either node is missing.

- **Access a node and its attributes**
  - `attrs = g.nodes[node_id]` returns a dict-like mapping of stored fields.
  - Example: `attrs["content"]`, `attrs.get("score", 0)`.

- **Traverse**
  - `g.successors(node_id)` → iterator over children (outgoing edges).
  - `g.predecessors(node_id)` → iterator over parents (incoming edges).
  - `list(g.nodes)` / `list(g.edges)` to inspect the whole graph.
  - Leaf detection (no outgoing edges): `g.out_degree(n) == 0`.

- **Update attributes**
  - `g.nodes[node_id]["score"] = 8`
  - `g.nodes[node_id].update({"reasoning": "Concise critique"})`

- **Integrity checks**
  - Existence: `if node_id not in g: ...`
  - DiGraph accepts parallel edges? No—`DiGraph` is simple; for multi-edges use `MultiDiGraph` (not used here).

- **Serialization hints**
  - Common pattern: `nx.to_dict_of_dicts(g)` or `nx.node_link_data(g)` (not implemented here but useful for saving/loading thought graphs).

In this codebase, `GraphReasoningState` wraps these calls so most graph mutations go through `add_vertex` and reads go through `get_vertex` or traversal helpers. Extend those helpers rather than manipulating the raw graph in orchestration code to keep invariants consistent.
