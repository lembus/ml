# Graph Neural Networks — A Comprehensive Reference Guide

> A standalone, textbook-quality reference for someone with no prior exposure to
> graph neural networks (GNNs). The guide assumes basic familiarity with linear
> algebra, calculus, and standard supervised learning, but defines every
> graph-theoretic and GNN-specific concept from first principles.

**Table of Contents**

- Topic 1: Graph Neural Networks — The Big Picture
- Topic 2: The Message Passing Framework (Node-, Edge-, and Graph-Level Tasks)
- Topic 3: Spectral Methods — Graph Convolutional Networks via the Graph Laplacian
- Topic 4: Spatial Methods — GAT (Graph Attention Networks) and GraphSAGE
- Topic 5: Applications — Molecules, Social Networks, Recommendation Systems
- Interview Preparation Section

---

# Topic 1: Graph Neural Networks — The Big Picture

## 1.1 Motivation & Intuition

### Why we need a new kind of neural network at all

Classical deep learning works because the network architecture matches the
structure of the data. Convolutional Neural Networks (CNNs) work for images
because images live on a regular 2D grid: every pixel has the same number of
neighbors (up, down, left, right), every pixel has a canonical position
(row *i*, column *j*), and a small local pattern (an edge, a texture) means
the same thing wherever it occurs. Recurrent Neural Networks (RNNs) work for
text and audio because language is a chain: there is a clear "before" and
"after," and each token has exactly one predecessor and one successor.

Now ask: what does a *molecule* look like? A molecule is a collection of atoms
connected by bonds. The number of atoms varies between molecules. Each atom
has a different number of neighbors (a hydrogen has one bond, a carbon up to
four). There is no canonical ordering — the same molecule can be drawn with
its atoms labeled in any order, and every reasonable model should produce the
same prediction. None of the structure that CNNs and RNNs depend on is there.

The same is true for *social networks* (people connected by friendships),
*citation networks* (papers connected by citations), *knowledge graphs* (entities
connected by typed relations), *transaction networks* (accounts connected by
money flows), *protein interaction networks*, *road networks*, and many others.
These data are fundamentally **graph-structured**: a set of items with a set of
pairwise relationships among them.

A Graph Neural Network is a neural architecture designed so that:

1. Its inputs can be graphs of arbitrary size and shape.
2. Its outputs do not depend on how we happened to label the nodes
   (permutation invariance / equivariance).
3. Its hidden representations of each node are learned by combining
   information from that node's neighbors.

### A concrete first example: predicting whether a molecule is toxic

Imagine you have a dataset of 10,000 small molecules, and for each one you
know whether it is toxic to the liver. You want to train a model that, given
a new molecule, predicts toxicity.

How would you turn a molecule into something a feed-forward neural network can
ingest? You could try to come up with hand-crafted features — molecular weight,
number of nitrogen atoms, presence of certain substructures (so-called
"molecular fingerprints"). This works to some extent and was the dominant
approach in cheminformatics for decades, but it forces you to anticipate which
features matter.

A GNN takes a different route. Treat each atom as a *node*, with its element
type (C, N, O, …), formal charge, hybridization, etc., as its initial
*node features*. Treat each bond as an *edge*, with its type (single, double,
aromatic) as its *edge feature*. Now repeatedly let each atom "look at" its
neighbors and update its own representation based on what it sees. After a few
rounds, each atom's vector encodes information about its local chemical
environment — which is exactly what determines reactivity, polarity, and
ultimately biological activity. Finally, pool all atom vectors together into
one molecule-level vector and feed that into a standard classifier.

The model never had to be told what a "carbonyl group" or an "aromatic ring"
is. If those substructures matter for toxicity, the network learns to
recognize them.

### A second concrete example: who should you be friends with?

Social media platforms recommend new connections. The signal "friend of a
friend" is intuitively useful, but in reality the signal is much richer:
*which* friends matter? Friends who have many friends in common with you?
Friends who share interests? A GNN can learn an embedding for each user such
that users who *should* be connected end up close in the embedding space,
using both user attributes (age, location, interests) and the structure of
the existing friendship graph. The score for "should A and B be friends?"
becomes simply the similarity (often dot product) of their learned embeddings.

### Why this is hard, mathematically

The reason this is harder than a CNN or RNN is that there is no canonical
ordering of nodes. If I label a graph's nodes 1, 2, …, *N* in one order and
you label the same graph's nodes in a different order, our adjacency matrices
will be different (they will be permutations of each other). A model that
takes the adjacency matrix and node features as input could, in principle,
produce different outputs for what is the same graph. We need the model to be
constructed so that it provably gives the same answer either way. Fancy term:
**permutation invariance** for graph-level outputs, **permutation
equivariance** for node-level outputs (relabel the input nodes → outputs get
relabeled the same way, but the values attached to each node don't change).

This single requirement — be invariant or equivariant to node permutations —
turns out to drive almost the entire design of GNNs.

## 1.2 Conceptual Foundations

### Definitions you must know cold

- **Graph** *G* = (*V*, *E*): a set of **vertices** (or **nodes**) *V* and a
  set of **edges** *E* ⊆ *V* × *V*. We write *N* = |*V*| for the number of
  nodes and *M* = |*E*| for the number of edges.
- **Directed vs. undirected**: in an undirected graph (*u*, *v*) and (*v*, *u*)
  refer to the same edge. In a directed graph they are different.
- **Self-loop**: an edge from a node to itself, (*v*, *v*).
- **Neighborhood** N(*v*): the set of nodes adjacent to *v*. In a directed
  graph one distinguishes in-neighbors and out-neighbors.
- **Degree** deg(*v*) = |N(*v*)| (in-degree and out-degree in the directed
  case).
- **Adjacency matrix** *A* ∈ ℝ^(*N*×*N*): *A*ᵢⱼ = 1 if there is an edge from
  *i* to *j*, else 0. For weighted graphs, *A*ᵢⱼ holds the edge weight. For
  undirected graphs *A* is symmetric.
- **Degree matrix** *D*: a diagonal matrix with *D*ᵢᵢ = deg(*i*).
- **Node features** *X* ∈ ℝ^(*N*×*d*): row *i* is the feature vector of node
  *i*. For atoms this might be a one-hot encoding of element + chemistry
  features; for users this might be demographics + interests.
- **Edge features** *E* ∈ ℝ^(*M*×*dₑ*) (notation overloaded with edge set —
  context disambiguates): vectors attached to each edge.
- **Graph features** *u* ∈ ℝ^(*dᵤ*): global attributes of the whole graph
  (e.g., temperature in a molecular dynamics simulation).

### The three task levels

GNN applications fall into one of three buckets, and the architecture differs
mainly at the output:

- **Node-level**: output one prediction per node. Examples: classify each
  user's risk of fraud; classify each paper into a topic; predict each
  protein residue's secondary structure.
- **Edge-level (link prediction)**: output one prediction per pair of nodes.
  Examples: will user A friend user B? Does a drug interact with a target
  protein? Is the missing entry in a knowledge graph (head, relation, tail)
  true?
- **Graph-level**: output one prediction per entire graph. Examples: is this
  molecule toxic? What is the energy of this protein conformation? What is
  the difficulty rating of this circuit?

### Inductive vs. transductive learning

A subtle but enormously important distinction.

- **Transductive**: training, validation, and test nodes all live in the same
  fixed graph. You see the entire graph structure during training, but only
  some node labels. The classic example is the Cora citation network: 2,708
  papers, ~5% labeled, predict topics of the rest.
- **Inductive**: at test time, the model must generalize to brand-new graphs
  (whole new molecules, whole new social subgraphs) or new nodes added to a
  growing graph (a new user joins).

Some architectures (vanilla spectral GCNs) are inherently transductive
because they depend on properties of a *specific* graph's Laplacian.
Others (GraphSAGE, GAT, message passing networks) are naturally inductive.

### Permutation invariance and equivariance — precisely

Let *P* be an *N* × *N* permutation matrix (a square binary matrix with
exactly one 1 in each row and column). Permuting the nodes of a graph
amounts to:

- Replacing the feature matrix *X* with *PX* (rows reordered).
- Replacing the adjacency matrix *A* with *PAP*ᵀ (rows *and* columns reordered
  by the same permutation).

A graph-level function *f*(*X*, *A*) is **permutation-invariant** if
*f*(*PX*, *PAP*ᵀ) = *f*(*X*, *A*) for every permutation *P*.

A node-level function *f*(*X*, *A*) ∈ ℝ^(*N*×*d′*) is **permutation-equivariant**
if *f*(*PX*, *PAP*ᵀ) = *Pf*(*X*, *A*) for every *P*.

Almost every architectural decision in GNNs is justified by ensuring one of
these two properties holds.

### Underlying assumptions and what breaks when they fail

GNNs typically assume:

1. **The graph structure is meaningful**: nearby nodes (in graph distance)
   should be more related than distant ones. This is sometimes called the
   **homophily** assumption when discussing node classification (connected
   nodes tend to share labels). When violated — in **heterophilic** graphs —
   off-the-shelf GNNs often perform worse than a graph-blind MLP.
2. **Local computation suffices**: useful information about a node can be
   gathered from a few hops of its neighborhood. When tasks require
   long-range reasoning across many hops, vanilla message passing suffers
   from **over-squashing** (information from many distant nodes must be
   compressed through a bottleneck) and **over-smoothing** (after too many
   layers, all nodes' representations collapse).
3. **The graph is reasonably static during inference**: streaming graphs
   require dynamic GNNs.
4. **The graph is correctly observed**: missing edges or noisy edges can
   propagate errors quickly.

We will return to each of these.

## 1.3 Mathematical Formulation

### Notation we will use throughout

- *G* = (*V*, *E*), *N* = |*V*|, *M* = |*E*|.
- *A* ∈ {0, 1}^(*N*×*N*) (or ℝ^(*N*×*N*) for weighted graphs).
- *D* the diagonal degree matrix.
- *X* ∈ ℝ^(*N*×*d*) the node feature matrix; row *i* is **x**ᵢ.
- *H*^(*l*) ∈ ℝ^(*N*×*dₗ*) the hidden representations after layer *l*; row *i*
  is **h**ᵢ^(*l*). By convention *H*^(0) = *X*.
- N(*v*) the (open) neighborhood of *v*; N̄(*v*) = N(*v*) ∪ {*v*} the closed
  neighborhood.
- *W*^(*l*), *b*^(*l*) trainable parameters at layer *l*.
- σ(·) a nonlinearity (ReLU, GELU, …).

### The general GNN layer

A single GNN layer updates every node's hidden vector based on its current
vector and the vectors of its neighbors. In its most general form:

  **h**ᵥ^(*l*+1) = UPDATE^(*l*)( **h**ᵥ^(*l*), AGGREGATE^(*l*)({{ **h**ᵤ^(*l*) : *u* ∈ N(*v*) }}))

Three things to notice:

1. The double braces {{·}} denote a **multiset** (a set that can contain
   duplicates), because two distinct neighbors may currently have identical
   hidden vectors and we need to count both.
2. The aggregator must be a **permutation-invariant function on a multiset**
   (sum, mean, max, attention-weighted sum, …). This is what guarantees the
   GNN does not depend on how we ordered *v*'s neighbors.
3. UPDATE typically combines the node's previous state with the aggregated
   neighbor information, often by an MLP, a GRU cell, or a simple linear
   transformation followed by a nonlinearity.

The architecture is defined by *which* AGGREGATE and UPDATE you choose. As we
will see, GCN picks a degree-normalized sum, GraphSAGE picks (mean | LSTM |
max), GAT picks attention-weighted sum, GIN picks plain sum + an MLP, and so
on.

### Stacking layers gives multi-hop receptive fields

After one layer, **h**ᵥ depends on *v* and its 1-hop neighbors. After two
layers, **h**ᵥ depends on *v*'s 1-hop neighbors *and on their 1-hop neighbors*,
i.e., *v*'s 2-hop neighborhood. After *L* layers, **h**ᵥ depends on the
*L*-hop neighborhood. This is exactly analogous to stacking convolutional
layers in a CNN to grow the receptive field, but on irregular graphs the
receptive-field size grows much faster (potentially exponentially).

### Permutation equivariance — proof sketch

Suppose AGGREGATE is a permutation-invariant function on multisets and UPDATE
operates pointwise on each node. Then permuting the nodes of the input graph
permutes the rows of *H*^(*l*) the same way: *H*^(*l*+1) is
permutation-equivariant. By induction, any stack of such layers is
permutation-equivariant, and a permutation-invariant readout (e.g.,
sum/mean/max over rows) on top yields a permutation-invariant graph-level
output.

### A useful matrix form

When AGGREGATE is a (possibly normalized) sum of neighbors and UPDATE is a
linear map followed by a nonlinearity, the entire layer can be written as a
single matrix expression:

  *H*^(*l*+1) = σ( *Â* *H*^(*l*) *W*^(*l*) )

for some *Â* derived from *A*. Different choices of *Â* give different GNNs:

- *Â* = *A* — sum-aggregation (no self).
- *Â* = *A* + *I* — sum-aggregation including self.
- *Â* = *D*⁻¹*A* — mean-aggregation.
- *Â* = *D̃*⁻¹ᐟ²(*A* + *I*)*D̃*⁻¹ᐟ² — symmetric normalization (GCN's choice,
  derived in Topic 3).

This matrix form is convenient for implementation on dense graphs but for
sparse graphs is implemented via scatter/gather operations.

## 1.4 Worked Example

We will run one round of message passing through a tiny graph by hand, using
sum aggregation.

### The graph

Four nodes, four edges (undirected):

```
    1 ─── 2
    │     │
    3 ─── 4
```

Edges: {1–2, 1–3, 2–4, 3–4}. Each node has a 2-dimensional feature vector:

  **x**₁ = (1, 0),  **x**₂ = (0, 1),  **x**₃ = (1, 1),  **x**₄ = (0, 0).

Adjacency and degree matrices:

  *A* = [[0,1,1,0],
        [1,0,0,1],
        [1,0,0,1],
        [0,1,1,0]],     *D* = diag(2, 2, 2, 2).

### Aggregating once with a plain sum

  **a**₁ = **x**₂ + **x**₃ = (0,1) + (1,1) = (1, 2).
  **a**₂ = **x**₁ + **x**₄ = (1,0) + (0,0) = (1, 0).
  **a**₃ = **x**₁ + **x**₄ = (1,0) + (0,0) = (1, 0).
  **a**₄ = **x**₂ + **x**₃ = (0,1) + (1,1) = (1, 2).

Notice **a**₁ = **a**₄ and **a**₂ = **a**₃. This is because nodes 1 & 4 have
the same neighborhood multiset of features, and similarly for nodes 2 & 3.
After many layers a structurally symmetric pair will keep producing identical
hidden vectors — this is fundamental, not a bug.

### Combining with the node's own state via a learned linear map

Pick *W*^(0) = [[1, 0, 1], [0, 1, 1]] (a 2×3 matrix), so the layer maps each
node's combined "self + aggregated neighbors" through *W*^(0). Concretely we
form **z**ᵥ = (**x**ᵥ + **a**ᵥ) *W*^(0), then ReLU.

  **x**₁ + **a**₁ = (2, 2).  **z**₁ = (2, 2)·*W*^(0) = (2, 2, 4).
  **x**₂ + **a**₂ = (1, 1).  **z**₂ = (1, 1, 2).
  **x**₃ + **a**₃ = (2, 1).  **z**₃ = (2, 1, 3).
  **x**₄ + **a**₄ = (1, 2).  **z**₄ = (1, 2, 3).

ReLU does nothing here (all values non-negative). These are the new hidden
vectors **h**ᵥ^(1).

Notice that nodes 2 and 3 now have *different* representations even though
they are structurally similar — because their initial features differed.
Nodes 1 and 4 were structurally symmetric *and* had different initial
features (1, 0) vs (0, 0), so they also separated. Initial features and
structure both contribute.

### What if we wanted a graph-level prediction?

We would apply a permutation-invariant readout — say, a sum:

  **h**_*G* = **h**₁^(1) + **h**₂^(1) + **h**₃^(1) + **h**₄^(1) = (6, 6, 12).

Then feed **h**_*G* into a standard MLP for whatever the graph-level target is.

## 1.5 Relevance to Machine Learning Practice

### Where GNNs are deployed today

- **Drug discovery**: Atomwise, Insilico Medicine, Recursion, and major pharma
  use GNNs for molecular property prediction (toxicity, solubility, binding
  affinity to a target).
- **Structural biology**: AlphaFold-2 uses an attention mechanism over residue
  graphs (its Evoformer can be viewed as a kind of GNN over residue pairs);
  AlphaFold-3 extends this to general molecular complexes.
- **Recommendation systems**: Pinterest's PinSage uses GraphSAGE-style GNNs at
  industrial scale (billions of nodes, tens of billions of edges) to learn pin
  embeddings used in recommendation.
- **Maps & ETA prediction**: DeepMind and Google Maps use GNNs over road
  networks to improve estimated arrival times.
- **Fraud and anti-money-laundering**: banks model accounts and transactions
  as graphs; GNNs catch fraud rings that tabular models miss.
- **Code analysis**: program graphs (AST + data/control flow) drive bug
  detection and vulnerability prediction.
- **Physics simulators**: GNS (Graph Network Simulators) from DeepMind learn
  rigid body and fluid dynamics by treating particles as nodes and contacts as
  edges.

### When to use a GNN — and when not to

Use a GNN when:

- The relational structure is informative for the task (homophily or known
  causal links).
- The structure varies across instances (so feature engineering doesn't
  scale).
- Permutation invariance is required by the problem definition.

Avoid GNNs when:

- Your "graph" is just a fully connected network of features — a Transformer
  is essentially a GNN on a complete graph and is often easier to train.
- The graph signal is very weak relative to node features (a well-tuned MLP
  on node features may match or beat a GNN; always include this baseline).
- The graph is huge and dynamic and your latency budget is tight — sometimes
  precomputed graph features (PageRank, node2vec embeddings) inside an MLP
  are simpler and good enough.
- The task is heterophilic and you haven't chosen an architecture designed for
  that case (FAGCN, H2GCN, GPR-GNN, etc.).

### Trade-offs

- **Bias–variance**: GNNs encode a strong inductive bias (locality and
  permutation symmetry). When that bias matches the data, they need fewer
  examples than an MLP. When the bias is wrong, they can systematically
  underperform.
- **Interpretability**: GNN predictions can be explained by attribution to
  subgraphs (GNNExplainer, PGExplainer, SubgraphX). Better than dense MLPs
  for relational data, but worse than e.g. logistic regression on hand-crafted
  features.
- **Robustness**: small adversarial edge perturbations can change predictions
  dramatically (Nettack, Metattack). This is an active research area.
- **Compute**: inference cost depends on neighborhood size, not just the model
  size. Hub nodes in social networks (millions of neighbors) cause memory
  blow-up unless sampling is used.

## 1.6 Common Pitfalls & Misconceptions

- **"More layers must be better."** No — beyond ~3–4 layers, vanilla GNNs
  often *degrade* due to **over-smoothing** (all hidden vectors converge) and
  **over-squashing** (information from many distant nodes is squeezed through
  a few intermediate ones). Architectures use residual connections, jumping
  knowledge, and identity mapping (GCNII) to push depth further.
- **"GNNs use the graph, so they automatically beat MLPs."** Not always.
  Always include the MLP-on-features baseline; on heterophilic graphs, an MLP
  can win.
- **"The adjacency matrix is the only structural information I need."**
  Edge features (bond type in chemistry, relation type in knowledge graphs,
  amount in transactions) are often crucial and are easy to forget.
- **"My GNN should generalize from one graph to another."** Only if you chose
  an inductive architecture (GraphSAGE, GAT, GIN with proper readout) and
  trained on enough graphs. A single-graph GCN can be transductive.
- **"Mean aggregation is fine."** Mean aggregation cannot distinguish a node
  with two identical neighbors from a node with one neighbor of that type —
  loses count information. Sum aggregation preserves it (this is the central
  insight of GIN).
- **"Self-loops aren't important."** Without self-loops, a node's own current
  representation is dropped from the aggregation. Many architectures
  (GCN included) add self-loops explicitly via *Ã* = *A* + *I*.
- **"I'll just use a Transformer instead."** A vanilla Transformer treats the
  input as a fully connected graph. On large graphs this is *N*² complexity
  and loses the structural inductive bias. For small graphs (molecules with a
  few dozen atoms) it can work; for graphs with millions of nodes it does
  not.

---

# Topic 2: The Message Passing Framework (Node-, Edge-, Graph-Level Tasks)

## 2.1 Motivation & Intuition

### Why we want a single framework

By 2017 there were dozens of GNN proposals — spectral, spatial, attention-based,
recurrent — each with its own derivation and notation. Gilmer et al. (2017),
in the paper "Neural Message Passing for Quantum Chemistry," noticed that
nearly all of them could be expressed in the same three-step recipe:

1. Each edge produces a **message** that is a function of the connected
   nodes (and possibly the edge's features).
2. Each node **aggregates** the messages from its incoming edges.
3. Each node **updates** its hidden state based on its previous state and
   the aggregated message.

Calling this the **Message Passing Neural Network (MPNN)** framework, they
showed that GCN, GG-NN (gated graph NN), interaction networks, and several
others were special cases. The framework also clarifies what is *missing* in
any new proposal: did the authors specify all three of message, aggregate,
update?

The intuition is closely analogous to belief propagation on graphical models
(the algorithm Pearl developed for exact inference on trees), and to
distributed computation in general: nodes don't see the whole graph; they
only see what their neighbors send them, and they have to combine those
incoming signals into a useful summary.

### A concrete worked-up intuition

Imagine you're at a dinner party where 30 people are seated at small tables.
You can only talk to your immediate tablemates. After ten minutes of
conversation, what do you know? You know about yourself plus your tablemates
(layer 1). Now everyone moves to a new table, with one person from each
original table going to each new table. After another ten minutes, you've
learned things that originated from people two tables away through the
intermediate guests (layer 2). After enough rounds, gossip from anywhere in
the room can reach you. This is exactly how multi-layer message passing
spreads information.

## 2.2 Conceptual Foundations

### The three components

For each layer *l*:

- **MESSAGE**: a function ψ that, for each directed edge *u* → *v*, produces a
  message **m**_{u→v}^(*l*) from the current states of *u* and *v* (and
  optionally the edge feature **e**_{uv}).
- **AGGREGATE**: a function ⊕ that combines the multiset of incoming
  messages at node *v* into a single vector **a**_v^(*l*).
- **UPDATE**: a function φ that combines node *v*'s current state with
  **a**_v^(*l*) to produce **h**_v^(*l*+1).

For graph-level tasks we additionally need:

- **READOUT**: a permutation-invariant function ρ that takes the multiset of
  final node embeddings and produces a single graph embedding **h**_*G*.

### Why each step exists

- The **MESSAGE** step lets the model reason about *pairs* of nodes: what does
  a relation between two specific nodes look like? It is also where edge
  features can enter.
- The **AGGREGATE** step is the source of permutation invariance: the
  aggregator treats messages as a multiset, with no ordering.
- The **UPDATE** step is where the model can choose to forget some of its
  previous state, blend it with new information, and apply nonlinearities.
  Without UPDATE, a layer's output is fully determined by the neighborhood —
  which is too forgetful when, e.g., the node's own identity matters.
- The **READOUT** step is the analogue of global average pooling in CNNs.

### Choices and what they imply

- **Sum vs mean vs max aggregation**:
  - *Sum* preserves multiset cardinality and value information; the most
    expressive in a precise theoretical sense (see WL test below).
  - *Mean* loses cardinality (a node with three neighbors of feature *x*
    looks the same as one with one neighbor of feature *x*).
  - *Max* captures extreme values but loses the rest.
- **Attention-weighted aggregation** (GAT): learns the relative weight of
  each neighbor.
- **Residual / skip connections** in UPDATE: combat over-smoothing.
- **READOUT**: sum, mean, max, attention-pooling, or hierarchical pooling
  (DiffPool, SAGPool).

### Connection to the Weisfeiler-Leman test

The 1-dimensional **Weisfeiler-Leman (1-WL) graph isomorphism test** is a
classical algorithm for distinguishing non-isomorphic graphs. It works by
iteratively assigning each node a "color" that hashes the multiset of its
neighbors' colors, refining the colors until they stabilize. Two graphs that
end with different color histograms are guaranteed to be non-isomorphic
(though the converse fails).

This is precisely a discrete version of message passing! Xu et al. (2019)
proved:

> A message-passing GNN can distinguish two graphs *only if* the 1-WL test
> can distinguish them.

So 1-WL is a hard ceiling on the expressive power of standard MPNNs. Their
**GIN** (Graph Isomorphism Network) achieves this ceiling by using a *sum*
aggregator and a *learnable injective* update via an MLP. Mean and max
aggregators are strictly weaker; they cannot match 1-WL.

This is one of the most important theoretical results about GNNs and informs
many practical decisions (e.g., why sum aggregation is the default in
expressive models).

### Underlying assumptions and their failures

- Messages are computed *only along existing edges*. If the graph is missing
  important edges (false negatives), information cannot flow. **Graph
  rewiring** techniques (DropEdge, GDC, expander graphs) help.
- Aggregation is *symmetric* across neighbors. If you really need an
  asymmetric notion (e.g., directional sequence-like reasoning), you need
  to encode position with edge types or embeddings.
- The model has no global view in any single layer. Tasks requiring global
  reasoning either need many layers (over-smoothing risk) or a global
  readout / virtual node.

## 2.3 Mathematical Formulation

### General message-passing update

For each layer *l* and each node *v*:

  **m**_{u→v}^(*l*+1) = ψ^(*l*)( **h**_u^(*l*), **h**_v^(*l*), **e**_{uv} )      for *u* ∈ N(*v*)

  **a**_v^(*l*+1) = ⊕_{u ∈ N(v)} **m**_{u→v}^(*l*+1)

  **h**_v^(*l*+1) = φ^(*l*)( **h**_v^(*l*), **a**_v^(*l*+1) )

After *L* layers, for graph-level tasks:

  **h**_*G* = ρ( {{ **h**_v^(*L*) : *v* ∈ *V* }} )

Here ψ, φ are learned (often MLPs); ⊕ and ρ are permutation-invariant
multiset functions.

### Reductions to specific architectures

- **GCN** (Kipf & Welling): ψ(**h**_u, **h**_v) = (1 / √(deg(*u*) deg(*v*))) **h**_u,
  ⊕ = sum, φ(**h**_v, **a**_v) = σ(*W* **a**_v).
- **GraphSAGE-mean**: ψ(**h**_u, **h**_v) = **h**_u, ⊕ = mean,
  φ(**h**_v, **a**_v) = σ(*W* · CONCAT(**h**_v, **a**_v)).
- **GAT**: ψ(**h**_u, **h**_v) = α_{vu} *W* **h**_u with attention coefficient
  α, ⊕ = sum, φ = σ.
- **GIN**: ψ(**h**_u, **h**_v) = **h**_u, ⊕ = sum,
  φ(**h**_v, **a**_v) = MLP((1 + ε) **h**_v + **a**_v).
- **MPNN with edge features**: ψ(**h**_u, **h**_v, **e**_{uv}) =
  *A*(**e**_{uv}) **h**_u where *A*(·) is a small network that turns edge
  features into a matrix (used in the original Gilmer et al. paper).

### Expressive power of sum aggregation: a sketch

Suppose two nodes *v*₁ and *v*₂ have neighborhood multisets
**X**₁ = {{ a, a, b }} and **X**₂ = {{ a, b, b }}.

- *Sum* of features: 2*a* + *b* vs *a* + 2*b* — distinguishable (different
  vectors in general).
- *Mean*: (2*a* + *b*)/3 vs (*a* + 2*b*)/3 — also distinguishable.

But now suppose *v*₃ has neighborhood multiset {{ a, b }} and *v*₄ has
{{ a, a, b, b }}.

- *Mean*: (a+b)/2 vs (2a+2b)/4 = (a+b)/2 — **identical**.
- *Sum*: a+b vs 2a+2b — distinguishable.

Mean aggregation literally loses count information; this is the core
weakness it has compared to sum. GIN's theorem makes this rigorous: the
class of message-passing GNNs that can match 1-WL's discriminative power
must use injective multiset functions, and sum + MLP is a constructive
example.

### Node, edge, and graph heads

Once we have **h**_v^(*L*) for every node, the head differs by task type:

- **Node classification**: ŷ_v = softmax(*W*_out **h**_v^(*L*)).
- **Link prediction**: ŷ_{uv} = σ( **h**_u^(*L*) · **h**_v^(*L*) )
  (dot-product), or = σ(MLP([**h**_u^(*L*) ‖ **h**_v^(*L*)])).
- **Graph classification/regression**: ŷ = MLP( ρ({{**h**_v^(*L*)}}) ).

For **edge-level tasks with rich edge features** (e.g., reaction
prediction in chemistry), people often maintain *edge representations*
that are also updated each layer (as in the original MPNN paper).

## 2.4 Worked Example

Let us run two layers of message passing — including all three node, edge,
and graph predictions — on a small example.

### Setup

Same square graph as before:

```
    1 ─── 2
    │     │
    3 ─── 4
```

Initial features (1-D for simplicity):

  *h*₁^(0) = 1,  *h*₂^(0) = 2,  *h*₃^(0) = 3,  *h*₄^(0) = 4.

Let the layer be:

- MESSAGE: ψ(*h*_u, *h*_v) = *h*_u (just the source's value).
- AGGREGATE: sum.
- UPDATE: φ(*h*_v, *a*_v) = *h*_v + *a*_v (no nonlinearity to keep arithmetic
  visible).

### Layer 1

  *a*₁^(1) = *h*₂^(0) + *h*₃^(0) = 2 + 3 = 5.    *h*₁^(1) = 1 + 5 = 6.
  *a*₂^(1) = *h*₁^(0) + *h*₄^(0) = 1 + 4 = 5.    *h*₂^(1) = 2 + 5 = 7.
  *a*₃^(1) = *h*₁^(0) + *h*₄^(0) = 1 + 4 = 5.    *h*₃^(1) = 3 + 5 = 8.
  *a*₄^(1) = *h*₂^(0) + *h*₃^(0) = 2 + 3 = 5.    *h*₄^(1) = 4 + 5 = 9.

### Layer 2

  *a*₁^(2) = *h*₂^(1) + *h*₃^(1) = 7 + 8 = 15.   *h*₁^(2) = 6 + 15 = 21.
  *a*₂^(2) = *h*₁^(1) + *h*₄^(1) = 6 + 9 = 15.   *h*₂^(2) = 7 + 15 = 22.
  *a*₃^(2) = *h*₁^(1) + *h*₄^(1) = 6 + 9 = 15.   *h*₃^(2) = 8 + 15 = 23.
  *a*₄^(2) = *h*₂^(1) + *h*₃^(1) = 7 + 8 = 15.   *h*₄^(2) = 9 + 15 = 24.

### Three different heads

- **Node-level prediction** (e.g., a class score for each node): apply
  *W*_out to each *h*_v^(2). E.g., *W*_out = 1: predicted scores
  (21, 22, 23, 24).
- **Edge-level prediction** for the edge (1, 2): combine via a head, e.g.,
  *h*₁^(2) · *h*₂^(2) = 21 · 22 = 462 (interpret as an unnormalized
  link-prediction logit).
- **Graph-level prediction** with sum readout:
  *h*_*G* = 21 + 22 + 23 + 24 = 90; pass through MLP for final output.

### Lesson

You can see how quickly representations grow: starting from values 1–4, two
layers later we are at 21–24. With a real model the linear maps and
nonlinearities keep magnitudes controlled; in practice one normalizes (layer
norm, batch norm, GraphNorm) to prevent explosion, especially in deep
networks.

You can also see the symmetry of the graph reflected in the math: nodes 1
and 4 have neighborhoods that are mirror images, but here their initial
features differed (1 vs 4) so they remain distinguishable. If you reset
*h*₁^(0) = *h*₄^(0) = 1 and *h*₂^(0) = *h*₃^(0) = 2, then after both layers
*h*₁ = *h*₄ and *h*₂ = *h*₃ for ever. Plain message passing **cannot
distinguish automorphism-equivalent nodes**.

## 2.5 Relevance to Machine Learning Practice

The MPNN framework is the *lingua franca* of modern GNNs. Practical takeaways:

- All major GNN libraries (PyTorch Geometric, DGL, Jraph, Spektral) provide
  generic `MessagePassing` base classes. To implement a new GNN, you specify
  `message`, `aggregate` (often by name: `'add'`, `'mean'`, `'max'`), and
  `update`. This is the abstraction layer everyone codes against.
- The choice of aggregator affects expressiveness (sum > mean ≈ max in the
  WL sense) but also gradient flow and numerical stability. Mean is often
  preferred when node degrees vary wildly (sum can blow up at high-degree
  nodes); a normalized sum (degree-corrected) is a common compromise.
- For large graphs, the framework also tells you how to parallelize:
  messages and aggregations are local, so they fit naturally onto
  scatter/gather GPU primitives.
- For edge-conditional models (reactions, social interactions with type), you
  almost always want to update *edge* representations alongside node
  representations.

### When to use which aggregator

- **GIN-style sum + MLP**: when expressiveness matters, e.g., distinguishing
  graph isomorphism in chemistry benchmarks.
- **Mean (GraphSAGE/GCN-style)**: when node degree varies and you want
  scale-invariance in messages.
- **Max (GraphSAGE-pool)**: when "presence of any neighbor with property X"
  matters (e.g., toxicity from a substructure).
- **Attention (GAT)**: when neighbors should not be treated equally.
- **Sum + multiple parallel aggregators (PNA)**: when you don't want to
  commit to one — Principal Neighborhood Aggregation concatenates several.

## 2.6 Common Pitfalls & Misconceptions

- **Forgetting that message passing has a fixed receptive field per layer.**
  If your task requires reasoning about something five hops away, one or two
  layers will never see it. But naively stacking ten layers brings
  over-smoothing.
- **Using mean aggregation with GIN-style ambitions.** GIN's theoretical
  guarantees rely on *injective* aggregators; mean is not injective on
  multisets.
- **Conflating node ordering with anything causal.** The ordering of
  neighbors during aggregation is arbitrary; if your problem really has a
  natural ordering (e.g., a sequence with a graph overlay), you must encode
  position explicitly.
- **Choosing the wrong readout for graph-level prediction.** Sum-readout has
  the same expressiveness benefits as sum-aggregation; mean-readout loses
  graph-size information, max-readout loses everything except the most
  extreme features.
- **Assuming the model can use edge features without you wiring them up.**
  Most basic GCN/GraphSAGE/GAT implementations ignore edge features unless
  you explicitly include them in the message function.
- **Treating GNNs as a Black-Box "graph encoder."** They have a definite
  inductive bias (locality, permutation symmetry, and at most 1-WL
  expressive power); knowing where this bias fails saves a lot of confusion
  later.

---

# Topic 3: Spectral Methods — Graph Convolutional Networks via the Graph Laplacian

## 3.1 Motivation & Intuition

### The fundamental question

What does it mean to do "convolution" on a graph? On an image, convolution
is straightforward: slide a small filter over every spatial location and
take a weighted sum of pixel values. The reason this is well-defined is
that image data lives on a regular grid: every pixel has the same number of
neighbors arranged in the same geometric pattern, so a fixed filter
"means the same thing" at every location.

A graph has neither uniform connectivity nor any geometric notion of
"same direction." We cannot translate a filter across a graph in any
straightforward sense. But there *is* a generalization, and it comes from
signal processing.

### The signal-processing intuition

In ordinary signal processing, the convolution theorem says:

> Convolution in the time domain = multiplication in the frequency domain.

So if convolution is hard to define directly on a graph, maybe we can:

1. Define a "Fourier transform" for graphs.
2. Multiply in the frequency domain.
3. Inverse-transform back.

It turns out the eigendecomposition of the **graph Laplacian** plays
exactly the role of the Fourier transform for graph signals. Its
eigenvectors are the analogues of complex exponentials; its eigenvalues are
the analogues of frequencies. This is the foundational insight of
**spectral graph theory** and the basis of spectral GNNs.

### From idea to GCN

The early spectral GNNs (Bruna et al. 2013) literally computed the Laplacian
eigendecomposition and learned filters in the spectral domain. This was
expensive (O(*N*³) eigendecomposition), poorly localized in space, and
non-transferable across graphs. Defferrard, Bresson, and Vandergheynst
(2016) introduced **ChebNet**, which approximates spectral filters with a
truncated Chebyshev polynomial expansion, giving local filters and avoiding
explicit eigendecomposition. Kipf and Welling (2017) made one further
simplification — first-order Chebyshev with a renormalization trick —
producing the now-iconic **Graph Convolutional Network (GCN)**:

  *H*^(*l*+1) = σ( *D̃*⁻¹ᐟ² *Ã* *D̃*⁻¹ᐟ² *H*^(*l*) *W*^(*l*) ),  where *Ã* = *A* + *I*, *D̃*ᵢᵢ = Σⱼ *Ã*ᵢⱼ.

This is the most-cited GNN equation in the literature, and despite the
spectral motivation it has a clean spatial interpretation: every node
takes a degree-normalized average of its neighbors (including itself) and
applies a linear transformation.

## 3.2 Conceptual Foundations

### The graph Laplacian

The (combinatorial / unnormalized) Laplacian of an undirected graph is

  *L* = *D* − *A*,

where *D* is the diagonal degree matrix and *A* is the adjacency matrix.

Why "Laplacian"? On a regular grid, the discrete Laplacian operator is the
second difference (*f*(*x* + 1) − 2*f*(*x*) + *f*(*x* − 1)), which measures
how much a function deviates from being linear. On a graph, (*L***x**)ᵢ =
deg(*i*) *x*ᵢ − Σⱼ∈N(i) *x*ⱼ measures how much **x** at node *i* differs
from the average of its neighbors. So *L* is the graph analog of the
Laplacian operator from continuous calculus.

Properties:

- *L* is symmetric (for undirected graphs) and positive semi-definite.
- Its eigenvalues 0 = λ₀ ≤ λ₁ ≤ … ≤ λ_{*N*−1} are non-negative.
- The multiplicity of the eigenvalue 0 equals the number of connected
  components of the graph.
- The eigenvector for λ₀ = 0 in a connected graph is constant (the
  all-ones vector, normalized).

### Normalized Laplacians

Two commonly used normalized variants:

- **Symmetric normalized Laplacian**:
  *L*_sym = *D*⁻¹ᐟ² *L* *D*⁻¹ᐟ² = *I* − *D*⁻¹ᐟ² *A* *D*⁻¹ᐟ².
- **Random-walk normalized Laplacian**:
  *L*_rw = *D*⁻¹ *L* = *I* − *D*⁻¹ *A*.

Eigenvalues of *L*_sym lie in [0, 2]. *L*_sym is symmetric (and so admits an
orthonormal eigenbasis); *L*_rw is generally not symmetric but has the same
eigenvalues as *L*_sym.

### The graph Fourier transform

Diagonalize the symmetric Laplacian:

  *L* = *U* Λ *U*ᵀ,

where *U* is an *N* × *N* orthogonal matrix whose columns are eigenvectors,
and Λ is a diagonal matrix of eigenvalues.

For any graph signal **x** ∈ ℝ^*N* (one scalar per node), define the
**graph Fourier transform**:

  **x̂** = *U*ᵀ **x**,        and the inverse:    **x** = *U* **x̂**.

The *k*-th component **x̂**_k is the projection of **x** onto the *k*-th
eigenvector. Eigenvectors with small eigenvalues correspond to "smooth"
signals (vary slowly across the graph), eigenvectors with large eigenvalues
to "rough" signals (vary quickly between adjacent nodes). This is *exactly*
analogous to low- vs high-frequency components in classical Fourier
analysis.

### Spectral convolution

Define convolution of **x** with a filter parameterized by *g*_θ via the
spectral domain:

  *g*_θ ⋆ **x** = *U* *g*_θ(Λ) *U*ᵀ **x**,

where *g*_θ(Λ) is a diagonal matrix obtained by applying the scalar
function *g*_θ to each eigenvalue. The filter is fully specified by the
*N* values *g*_θ(λ_k).

Problems with this raw formulation:

1. **Computational cost**: eigendecomposition is O(*N*³); applying *U*, *U*ᵀ
   is O(*N*²) per vector.
2. **Locality**: a generic *g*_θ produces filters that are not localized in
   the spatial domain (a single filter affects every node).
3. **Transferability**: the Laplacian and its spectrum depend on the
   specific graph; a learned spectral filter does not transfer to a
   different graph.

ChebNet and GCN solve all three by restricting *g*_θ to a special form.

### From spectral filter to GCN: the chain of approximations

Step 1 (ChebNet): approximate *g*_θ(Λ) by a truncated Chebyshev polynomial:

  *g*_θ(Λ) ≈ Σ_{k=0}^{K} θ_k *T*_k(Λ̃),     where Λ̃ = (2 / λ_max) Λ − *I*.

The *T*_k are Chebyshev polynomials of the first kind, defined by the
recurrence *T*_0(x) = 1, *T*_1(x) = x, *T*_{k+1}(x) = 2x *T*_k(x) − *T*_{k−1}(x).
Rescaling Λ to Λ̃ ensures the input to *T*_k lies in [-1, 1] where
Chebyshev polynomials behave nicely.

The crucial fact: applying a polynomial of *L* avoids eigendecomposition,
because

  *U* *T*_k(Λ̃) *U*ᵀ = *T*_k(*L̃*),     where *L̃* = (2 / λ_max) *L* − *I*.

So spectral convolution becomes a sum of polynomial-of-Laplacian terms
applied to the signal — and each *L̃**x* and higher powers can be computed
recursively without ever forming *U* or Λ.

Furthermore, *T*_k(*L̃*) only mixes nodes within graph distance *k* (because
*L̃* itself is a 1-hop operator), so the resulting filter is exactly
*K*-localized in the spatial domain.

Step 2 (GCN): take *K* = 1, and assume λ_max ≈ 2 (a good approximation for
the symmetric normalized Laplacian, since its largest eigenvalue is at
most 2). Then *L̃* = *L*_sym − *I* = −*D*⁻¹ᐟ² *A* *D*⁻¹ᐟ², and

  *g*_θ ⋆ **x** ≈ θ_0 **x** + θ_1 (−*D*⁻¹ᐟ² *A* *D*⁻¹ᐟ²) **x**.

Setting θ = θ_0 = −θ_1 (to reduce parameters and stabilize),

  *g*_θ ⋆ **x** ≈ θ ( *I* + *D*⁻¹ᐟ² *A* *D*⁻¹ᐟ² ) **x**.

The matrix *I* + *D*⁻¹ᐟ² *A* *D*⁻¹ᐟ² has spectrum in [0, 2]; repeated
multiplication can cause numerical instability.

Step 3 (the renormalization trick): replace *I* + *D*⁻¹ᐟ² *A* *D*⁻¹ᐟ² with
*D̃*⁻¹ᐟ² *Ã* *D̃*⁻¹ᐟ², where *Ã* = *A* + *I* and *D̃*ᵢᵢ = Σⱼ *Ã*ᵢⱼ. The
intuition: instead of separately adding self-loops and normalizing, build
self-loops directly into the adjacency before normalizing. The new operator
has eigenvalues in [0, 1] (more stable), and self-loops are still included.

Generalize to multiple input/output channels:

  *H*^(*l*+1) = σ( *D̃*⁻¹ᐟ² *Ã* *D̃*⁻¹ᐟ² *H*^(*l*) *W*^(*l*) ).

That is the GCN.

### Underlying assumptions

- The graph is **undirected** (otherwise *L* is not symmetric and
  eigendecomposition is not orthonormal).
- The graph is **fixed** during training and inference — the operator
  *D̃*⁻¹ᐟ² *Ã* *D̃*⁻¹ᐟ² is precomputed from a specific graph. This is why
  vanilla GCN is **transductive**.
- λ_max ≈ 2 — a fine approximation for the symmetric normalized Laplacian
  but breaks for the unnormalized one.
- Self-loops are useful (the renormalization trick assumes you want them).

When violated:

- Directed graphs: need asymmetric extensions (e.g., MagNet using complex
  Hermitian Laplacian).
- Graph changes: must recompute *D̃*⁻¹ᐟ² *Ã* *D̃*⁻¹ᐟ²; doesn't scale to
  streaming.
- Heterophilic graphs: the implicit "smoothing toward neighbors" assumption
  hurts performance.

## 3.3 Mathematical Formulation

### Setup

Undirected graph *G* = (*V*, *E*) with |*V*| = *N* nodes. Adjacency *A*,
degree *D*. Define *Ã* = *A* + *I* and *D̃* = diag(deg in Ã) = *D* + *I*.

### Forward pass

Input feature matrix *X* ∈ ℝ^(*N*×*d*). Set *H*^(0) = *X*. For each layer
*l* = 0, 1, …, *L* − 1:

  *H*^(*l*+1) = σ( *Â* *H*^(*l*) *W*^(*l*) ),  where *Â* = *D̃*⁻¹ᐟ² *Ã* *D̃*⁻¹ᐟ², *W*^(*l*) ∈ ℝ^(*d_l* × *d*_{l+1}).

Final output: a softmax classifier over *H*^(*L*) for node classification,
or a sum/mean readout for graph classification.

### Per-node interpretation

Writing the matrix form out for a single node *v*:

  **h**_v^(*l*+1) = σ( Σ_{u ∈ N̄(*v*)} (1 / √( *d̃*_v *d̃*_u )) *W*^(*l*) **h**_u^(*l*) ),

where *d̃*_v = deg(*v*) + 1 (because of the added self-loop). Each
neighbor's contribution is scaled by the square root of the product of the
two endpoints' degrees — this gives high-degree neighbors *less* influence,
preventing them from dominating.

### Connection back to the Laplacian

Note that for the symmetric normalized Laplacian *L*_sym, we have

  *I* − *L*_sym = *D*⁻¹ᐟ² *A* *D*⁻¹ᐟ².

So *D̃*⁻¹ᐟ² *Ã* *D̃*⁻¹ᐟ² is essentially *I* minus a renormalized symmetric
Laplacian on the graph with self-loops. The GCN layer is thus
"propagate one step of a (renormalized) random walk, then linearly
transform." The choice corresponds to a low-pass filter in the spectral
domain (small λ → preserved; large λ → suppressed).

### Why low-pass smoothing is the implicit bias

In the graph Fourier basis, the operator *D̃*⁻¹ᐟ² *Ã* *D̃*⁻¹ᐟ* multiplies
the *k*-th frequency component by approximately 1 − λ_k (where λ_k is the
*k*-th eigenvalue of the renormalized Laplacian). Small λ_k (smooth signals
across the graph) get multiplied by something close to 1; large λ_k
(rough signals) get multiplied by something small. So GCN systematically
*smooths* node features over the graph. This is good when neighboring nodes
share labels (homophily) and bad when they don't (heterophily).

### Over-smoothing — the spectral view

After *L* layers without nonlinearity, the propagation operator becomes
*Â*^*L*. As *L* → ∞, *Â*^*L* converges to a rank-1 matrix proportional to
the dominant eigenvector — i.e., all rows of *Â*^*L* *X* converge to the
same vector. Even with nonlinearities and weight matrices, this tendency
toward feature collapse persists empirically. For most node-classification
benchmarks, GCN performs best with 2–3 layers; performance degrades with 5+.

## 3.4 Worked Example

### A small graph, end-to-end

Same square graph: nodes {1, 2, 3, 4}, edges {1–2, 1–3, 2–4, 3–4}.

**Adjacency**:

```
A =  0 1 1 0
     1 0 0 1
     1 0 0 1
     0 1 1 0
```

**Add self-loops** (*Ã* = *A* + *I*):

```
Ã =  1 1 1 0
     1 1 0 1
     1 0 1 1
     0 1 1 1
```

**Degrees in Ã**: *d̃*₁ = 3, *d̃*₂ = 3, *d̃*₃ = 3, *d̃*₄ = 3 (each row sums
to 3).

**Symmetric normalization**: every entry *Ã*ᵢⱼ becomes
*Ã*ᵢⱼ / √(*d̃*ᵢ *d̃*ⱼ) = *Ã*ᵢⱼ / 3 (since all degrees are 3 here).

So *Â* = (1/3) *Ã*:

```
Â =  1/3 1/3 1/3 0
     1/3 1/3 0   1/3
     1/3 0   1/3 1/3
     0   1/3 1/3 1/3
```

**Initial features** (1-D for simplicity):
*X* = [1, 2, 3, 4]ᵀ.

**Apply *Â* *X*** (one layer of propagation, no weights or nonlinearity yet):

  (*Â* *X*)₁ = (1/3)(1 + 2 + 3) = 2.
  (*Â* *X*)₂ = (1/3)(1 + 2 + 4) = 7/3 ≈ 2.33.
  (*Â* *X*)₃ = (1/3)(1 + 3 + 4) = 8/3 ≈ 2.67.
  (*Â* *X*)₄ = (1/3)(2 + 3 + 4) = 3.

So features get smoothed: the spread (1, 2, 3, 4) became roughly (2, 2.33,
2.67, 3). Already the differences across nodes are smaller. Repeating this
operation many times will pull all nodes toward the mean.

**Apply a 1×2 weight matrix** *W*^(0) = [1, −1]:

  *H*^(1) = ReLU( *Â* *X* *W*^(0) ).

Each *Â* *X* row is a scalar; multiplied by [1, −1] gives a 2-vector;
ReLU keeps the positive part.

  Row 1: (2)(1, −1) = (2, −2) → ReLU → (2, 0).
  Row 2: (7/3, −7/3) → (7/3, 0).
  Row 3: (8/3, −8/3) → (8/3, 0).
  Row 4: (3, −3) → (3, 0).

After one GCN layer the second feature is 0 for all nodes — the second
column of *W*^(0) is a useless filter. In a real model, *W* is learned, so
both filters would carry signal.

### Comparison: what would change without the renormalization trick?

Using *D*⁻¹ᐟ² *A* *D*⁻¹ᐟ² (no self-loop) on the same graph: each row would
be 1/2 (1/2 in the off-diagonal positions where there was an edge, since
deg = 2 for all nodes). Multiplied by *X*:

  Row 1: (1/2)(2 + 3) = 2.5 (the node's *own* feature 1 is dropped).
  Row 2: (1/2)(1 + 4) = 2.5.
  Row 3: (1/2)(1 + 4) = 2.5.
  Row 4: (1/2)(2 + 3) = 2.5.

All nodes collapse to the same value 2.5 in one step — much more aggressive
smoothing. Self-loops keep each node's own information from being immediately
washed out, justifying the renormalization trick.

## 3.5 Relevance to Machine Learning Practice

### Where GCN is used

GCN became the standard baseline for **semi-supervised node classification**
on citation networks (Cora, Citeseer, PubMed) and is still the first thing
to try on a single fixed graph with sparse labels. The original 2017 paper
showed massive improvements over previous methods on these benchmarks with a
two-layer model and tens of thousands of parameters.

### When to use GCN

- Single fixed graph with node features and sparse labels.
- Homophilic graph (connected nodes share labels).
- Reasonably sized graph (up to a few hundred thousand nodes; for larger,
  consider sampling-based methods like GraphSAGE).
- Computational simplicity matters and you want a strong baseline.

### When not to

- **Inductive setting** (new graphs at test time, or new nodes added):
  GCN's operator depends on a specific Laplacian; use GraphSAGE or GAT.
- **Heterophilic graphs**: GCN's smoothing makes labels of dissimilar
  neighbors *more* similar, hurting accuracy. Use FAGCN, H2GCN, GPR-GNN,
  etc.
- **Very deep networks**: vanilla GCN over-smooths. Use GCNII (with initial
  residual + identity mapping) or include residual connections.
- **Edge-feature-rich graphs**: GCN ignores edge features. Use MPNN-style
  models.
- **Directed graphs**: vanilla GCN assumes symmetric *A*. Use DGCN, MagNet,
  or convert by symmetrization.

### Trade-offs

- **Bias–variance**: GCN has very strong inductive bias (low-pass filter on
  graph signals), giving low variance / high bias. Few labels suffice; on
  the wrong graph it underperforms an MLP.
- **Interpretability**: each layer is a closed-form linear operator on
  features. Explanations via masking edges or features are interpretable
  with tools like GNNExplainer.
- **Computational cost**: O(|*E*| * *d*) per layer for sparse implementation
  — extremely fast.
- **Memory**: must hold the entire propagation matrix (sparse) and all node
  features in memory; for very large graphs, this is the bottleneck.

## 3.6 Common Pitfalls & Misconceptions

- **"GCN's spectral derivation means it's a deep, principled method."** It
  is principled, but every approximation along the way (truncation, λ_max ≈
  2, single parameter, renormalization) is a *simplification*. The final
  GCN operator can equally be motivated as "degree-normalized neighbor
  averaging with a self-loop" — and this view often guides better
  intuition.
- **"Adding more layers means a bigger receptive field, so it must help."**
  Each layer adds smoothing. Past 3–4 layers, all node embeddings converge,
  and accuracy collapses. Use GCNII, JK-Net, residual connections, or
  PPR-based propagation to go deeper.
- **"GCN is permutation-equivariant, so I don't need to worry about node
  ordering."** True, but only as long as you keep using a permutation-
  equivariant operator. If you concatenate node embeddings into a fixed-
  size vector somewhere, you've lost equivariance.
- **"GCN handles directed graphs."** Out of the box it doesn't. People
  symmetrize the adjacency, which throws away directional information. If
  direction matters, use a directed variant.
- **"Use the unnormalized Laplacian *L* = *D* − *A*."** Don't — its
  eigenvalues can be very large (up to twice the maximum degree) and the
  approximation λ_max ≈ 2 is only valid for *L*_sym.
- **"Skip the self-loop because the node already passes itself information
  via deeper layers."** This is the pre-renormalization formulation and is
  numerically less stable. Always add self-loops.
- **"GCN can learn whatever pattern I throw at it."** Its expressive power
  is bounded by 1-WL, like all MPNNs. Tasks that require distinguishing
  graphs that 1-WL can't (e.g., counting cycles of fixed length, or
  identifying the "anchor node") are out of reach for vanilla GCN.

---

# Topic 4: Spatial Methods — GAT (Graph Attention Networks) and GraphSAGE

## 4.1 Motivation & Intuition

### Why we want spatial methods

Spectral methods (Topic 3) require the entire graph's Laplacian. This is
fine for a single, modest, fixed graph (the citation networks GCN was built
for) but breaks down when:

- The graph is enormous (millions of nodes — Pinterest has billions of
  pins).
- The graph is dynamic (a new user signs up; an edge is added every
  millisecond).
- We want to train on one graph and predict on a totally different one
  (drug discovery: train on known molecules, predict on novel ones).

Spatial methods avoid the spectral framework altogether. They operate
directly on the local neighborhood of each node, looking only at messages
from neighbors and producing an updated representation. Two highly
influential examples:

- **GraphSAGE** (Hamilton, Ying, Leskovec 2017): "SAmple and aggreGatE."
  For each node, sample a fixed-size subset of its neighbors, aggregate
  their features, and update. Inductive by construction; scales to billion-
  node graphs.
- **GAT** (Veličković et al. 2018): treat every neighbor's contribution as
  a weighted message, with weights given by learned attention scores. This
  lets the model decide which neighbors matter for each prediction, rather
  than treating all neighbors equally (as GCN does after degree
  normalization).

### Why "attention" makes sense for graphs

Consider a paper-citation graph. A paper cited by hundreds of others has
many neighbors, but most citations are routine and uninformative; a few
are cited *because* they share the same topic. A model that gives every
incoming citation equal weight wastes capacity. Attention lets the model
learn "this citation is more relevant to my topic than that one." On
social graphs, similar reasoning: out of your 500 friends, 5 share your
political leanings strongly, and the rest are noise for that particular
prediction.

### Why "sampling" makes sense for very large graphs

The high-degree problem: a Pinterest user might have 10,000 boards; an
influencer on Twitter has millions of followers. If you must aggregate
across *every* neighbor, mini-batch training is infeasible. GraphSAGE
samples a small, fixed-size random neighborhood (e.g., 25 neighbors per
node, 10 from each of those, for a 2-layer model — a manageable few
hundred per minibatch root node). The cost is variance: each forward pass
sees a different subgraph. The benefit: scalability and inductivity.

## 4.2 Conceptual Foundations

### GraphSAGE: definitions and components

For each layer *l*:

1. **Sample**: for each "target" node *v*, draw a fixed-size sample
   *S*(*v*) ⊆ N(*v*) of its neighbors, of size *s*. (If |N(*v*)| ≤ *s*,
   take all neighbors.)
2. **Aggregate**: combine the multiset {{ **h**_u^(*l*) : *u* ∈ *S*(*v*) }}
   into a single vector **h**_{N(*v*)}^(*l*+1) using one of three
   aggregators:
   - **Mean aggregator**: simple average of neighbor features (combined with
     the self vector inside the aggregator).
   - **LSTM aggregator**: feed the neighbors' features as a sequence into an
     LSTM (with *random permutation*, to symmetrize over orderings during
     training).
   - **Pool aggregator (max-pool)**: pass each neighbor's features through a
     small MLP, then take element-wise max.
3. **Update**: combine the node's own previous state with the aggregated
   neighbor message:

   **h**_v^(*l*+1) = σ( *W*^(*l*) · CONCAT( **h**_v^(*l*), **h**_{N(*v*)}^(*l*+1) ) ).

4. **Normalize** (optional but standard): **h**_v^(*l*+1) ←
   **h**_v^(*l*+1) / ‖**h**_v^(*l*+1)‖₂.

For *L* layers, this defines the embedding **z**_v = **h**_v^(*L*).

### GAT: definitions and components

For each layer *l*:

1. **Project**: linearly transform each node's features:
   *W***h**_u (where *W* ∈ ℝ^(*d′* × *d*)).
2. **Compute attention scores** for each edge (*v*, *u*) with *u* ∈ N(*v*):
   *e*_vu = LeakyReLU( **a**ᵀ [ *W***h**_v ‖ *W***h**_u ] ),
   where **a** ∈ ℝ^(2*d′*) is a learnable parameter and ‖ denotes
   concatenation.
3. **Normalize**: α_vu = softmax over *u* ∈ N̄(*v*) of *e*_vu, i.e.,
   α_vu = exp(*e*_vu) / Σ_{k ∈ N̄(*v*)} exp(*e*_vk).
4. **Aggregate**: **h**_v^(*l*+1) = σ( Σ_{u ∈ N̄(*v*)} α_vu *W* **h**_u ).
5. **Multi-head attention**: do steps 1–4 with *K* independent attention
   mechanisms (different *W*, **a**) and either concatenate or average the
   results.

### Inductive learning, more concretely

**Inductive** = the same trained model can handle nodes / graphs unseen at
training time. Both GraphSAGE and GAT are inductive because their
parameters (*W*, *a*, MLPs) are functions of *features*, not of node
identity. Nothing in the model "knows" node 17 specifically — it just knows
how to aggregate neighborhood feature patterns.

This is in contrast to a transductive method that would learn, e.g., a
specific embedding for each node ID, which is meaningless for a brand-new
node.

### Underlying assumptions

- **GraphSAGE**:
  - The graph is sparse enough that, with neighborhood sampling, the
    training signal is rich enough.
  - Neighbor features are informative when sampled — i.e., neighbors don't
    differ wildly in importance (otherwise sampling has high variance).
  - The aggregator chosen is appropriate for the data (mean for general
    use, max-pool for "any neighbor with property X" detection).

- **GAT**:
  - Different neighbors carry different importance for the task.
  - Attention scores can be learned reliably from features alone — this can
    fail when neighborhood sizes are huge and the attention distribution
    becomes too peaked or too flat.
  - The graph structure provides candidate neighbors; attention reweights,
    not removes.

When violated:

- For GraphSAGE: very high-degree nodes with small samples → high
  variance, unstable training. Mitigation: importance sampling (FastGCN,
  AS-GCN), full-neighbor evaluation at test time.
- For GAT: in graphs where structure is a noisy signal, attention
  collapses to "uniform" or "one-hot," providing no benefit. Multiple
  heads and edge dropout help.

## 4.3 Mathematical Formulation

### GraphSAGE in equations

For each layer *l* = 1, …, *L* and each node *v* in the current minibatch:

  *S*(*v*) ← uniform random sample of size *s_l* from N(*v*)

  **h**_{N(*v*)}^(*l*) = AGGREGATE_*l*( {{ **h**_u^(*l*−1) : *u* ∈ *S*(*v*) }} )

  **h**_v^(*l*) = σ( *W*^(*l*) · CONCAT( **h**_v^(*l*−1), **h**_{N(*v*)}^(*l*) ) )

  **h**_v^(*l*) ← **h**_v^(*l*) / ‖ **h**_v^(*l*) ‖₂

The three aggregators in the original paper:

- **Mean**:
  AGGREGATE^(mean)({{ **h**_u }}) = (1 / |*S*|) Σ_{u ∈ *S*} **h**_u.

  In the **GCN-style mean variant** the self vector is included inside the
  aggregator instead of via concatenation:
  **h**_v^(*l*) = σ( *W* · MEAN( {{ **h**_v^(*l*−1) }} ∪ {{ **h**_u^(*l*−1) : *u* ∈ *S*(*v*) }} ) ).

- **LSTM**: feed the (randomly permuted) sequence of **h**_u into an LSTM,
  take the final hidden state. Not permutation-invariant by construction;
  permutation invariance is approximated by training on random orderings.

- **Pool (max-pool)**:
  AGGREGATE^(pool)({{ **h**_u }}) = max( {{ σ(*W*_pool **h**_u + *b*) : *u* ∈ *S* }} ),
  where max is element-wise.

### GraphSAGE training

Two main settings:

- **Supervised**: standard cross-entropy or MSE on labels.
- **Unsupervised** (the original "PinSage" setting): a contrastive loss
  encouraging connected pairs to have similar embeddings and random pairs
  to have dissimilar ones:

  J_*G*(**z**_v) = − log σ( **z**_v · **z**_u ) − Q · 𝔼_{**v**_n ~ P_n(*v*)} [ log σ( −**z**_v · **z**_{v_n} ) ]

  for *u* a neighbor of *v* sampled by a random walk, and *Q* "negative"
  samples drawn from a noise distribution *P_n*.

### GAT in equations

For each layer (showing single-head; multi-head adds an outer loop):

  **z**_v = *W* **h**_v,    for every *v*

  *e*_vu = LeakyReLU( **a**ᵀ [ **z**_v ‖ **z**_u ] ),    for every edge (*v*, *u*)

  α_vu = exp( *e*_vu ) / Σ_{k ∈ N̄(*v*)} exp( *e*_vk )

  **h**_v^(new) = σ( Σ_{u ∈ N̄(*v*)} α_vu **z**_u ).

Here LeakyReLU has a small negative slope (typically 0.2). The closed form
for a 2-element concatenation makes the attention computation cheap: it's a
single dot product per edge.

### Multi-head attention

With *K* heads, each producing **h**_v^{(k), new} ∈ ℝ^*d′*:

- **Concatenation** (intermediate layers):

  **h**_v^(new) = ‖_{k=1}^{K} σ( Σ_u α^{(k)}_vu *W*^{(k)} **h**_u ),
  output dimension *K* · *d′*.

- **Averaging** (final layer):

  **h**_v^(new) = σ( (1/*K*) Σ_{k=1}^{K} Σ_u α^{(k)}_vu *W*^{(k)} **h**_u ),
  output dimension *d′*.

### Computational cost

- **GraphSAGE forward**, *L* layers, sample size *s* per layer, *B* root
  nodes per minibatch:
  - Each root node touches *s*^*L* nodes through the sampled tree.
  - Time per minibatch: O(*B* · *s*^*L* · *d* · *d′*).
  - This is *independent* of the total graph size — that's the whole
    point.

- **GAT forward**, full graph, *K* heads, *L* layers:
  - Time: O(*L* · *K* · (|*V*| *d* + |*E*| *d′*)).
  - For sparse graphs with average degree *d̄*, this is essentially
    O(*L* *K* |*V*| *d̄* *d′*) — linear in the number of edges.

### Permutation equivariance

- **GraphSAGE-mean / pool**: AGGREGATE is symmetric → equivariant.
- **GraphSAGE-LSTM**: not symmetric in principle; trained over random
  orderings. The model is *approximately* equivariant.
- **GAT**: softmax over neighbors and weighted sum are symmetric →
  equivariant.

## 4.4 Worked Examples

### GAT attention computation by hand

Take a tiny graph and compute one GAT layer step by step.

Node *v* = 1 with three neighbors *u*₁ = 2, *u*₂ = 3, *u*₃ = 4. Self-loops
included, so N̄(1) = {1, 2, 3, 4}.

Initial features (2-D):

  **h**₁ = (1, 0),  **h**₂ = (0, 1),  **h**₃ = (1, 1),  **h**₄ = (0, 0).

Linear projection *W* (2 × 2):

  *W* = [[1, 1],
         [1, −1]].

Project:

  **z**₁ = (1, 0) *W*ᵀ ... actually let's use **z** = *W* **h** with column convention.

To avoid confusion, treat **h** as a column vector and *W* as 2×2:

  **z**₁ = *W* **h**₁ = [[1·1 + 1·0], [1·1 − 1·0]] = (1, 1).
  **z**₂ = *W* **h**₂ = (0 + 1, 0 − 1) = (1, −1).
  **z**₃ = *W* **h**₃ = (1 + 1, 1 − 1) = (2, 0).
  **z**₄ = *W* **h**₄ = (0, 0).

Attention vector **a** ∈ ℝ⁴ (since concat of two 2-vectors).
Pick **a** = (1, 1, 1, 1).

Raw attention scores (before LeakyReLU):

  *e*_{1,1} = **a**ᵀ ( **z**₁ ‖ **z**₁ ) = (1+1+1+1) = 4.
  *e*_{1,2} = **a**ᵀ ( **z**₁ ‖ **z**₂ ) = (1 + 1 + 1 + (−1)) = 2.
  *e*_{1,3} = **a**ᵀ ( **z**₁ ‖ **z**₃ ) = (1 + 1 + 2 + 0) = 4.
  *e*_{1,4} = **a**ᵀ ( **z**₁ ‖ **z**₄ ) = (1 + 1 + 0 + 0) = 2.

LeakyReLU is identity for positive values; all values positive here.

Softmax over {4, 2, 4, 2}:

  Numerators: e⁴ ≈ 54.60, e² ≈ 7.39, e⁴ ≈ 54.60, e² ≈ 7.39.
  Sum: 54.60 + 7.39 + 54.60 + 7.39 = 123.98.

  α_{1,1} = 54.60 / 123.98 ≈ 0.4404.
  α_{1,2} = 7.39 / 123.98 ≈ 0.0596.
  α_{1,3} = 54.60 / 123.98 ≈ 0.4404.
  α_{1,4} = 7.39 / 123.98 ≈ 0.0596.

The attention concentrates on node 1 (itself) and node 3 (the neighbor with
the largest projected feature). Aggregate:

  **h**₁^(new) (pre-σ) =
    α_{1,1} · **z**₁ + α_{1,2} · **z**₂ + α_{1,3} · **z**₃ + α_{1,4} · **z**₄
    = 0.4404 · (1, 1) + 0.0596 · (1, −1) + 0.4404 · (2, 0) + 0.0596 · (0, 0)
    = (0.4404 + 0.0596 + 0.8808 + 0, 0.4404 − 0.0596 + 0 + 0)
    = (1.3808, 0.3808).

Apply σ (say, ELU): ELU is identity for positive inputs, so
**h**₁^(new) ≈ (1.38, 0.38).

This single number-pushing step illustrates: attention let the model emphasize
nodes 1 and 3 (each contributing about 44%) over 2 and 4 (each 6%) — a
nontrivial reweighting from the uniform 25% that mean aggregation would
give.

### GraphSAGE on a hub node

Suppose node *v* is a celebrity Twitter account with N(*v*) = 1,000,000
followers. We choose *s* = 25.

- Sample *S*(*v*): 25 followers chosen uniformly at random from the million.
- Aggregate (mean): average their feature vectors.
- Update: concatenate with **h**_v, apply *W*, ReLU, normalize.

If we did this without sampling we would have to load and compute over a
million 128-dim vectors per minibatch step for *v* alone. With *s* = 25
this is 25 vectors. Across two layers and a minibatch of 256 root nodes:

  Total vectors loaded per step ≈ 256 · (25 + 25 · 25) ≈ 256 · 650 = 166,400
  vectors. Easily fits in GPU memory.

The catch: each minibatch sees a different random sample of *v*'s neighbors,
producing variance in *v*'s embedding from step to step. At inference time
one typically uses a larger sample or all neighbors (if affordable).

## 4.5 Relevance to Machine Learning Practice

### GraphSAGE in industry

- **Pinterest's PinSage** (Ying et al., 2018): GraphSAGE on a 3-billion-node,
  18-billion-edge bipartite graph of pins and boards. Used for "Related
  Pins" recommendations. Innovations beyond basic GraphSAGE: importance-
  weighted (random-walk PPR) sampling instead of uniform; producer-consumer
  curriculum during training; MapReduce inference.
- **Uber Eats** uses GNNs with sampling for restaurant recommendations.
- **LinkedIn** uses sampling-based GNNs for connection suggestions and job
  matching.

### GAT in practice

- Strong performance on citation network benchmarks, especially when graph
  has variable-importance neighbors.
- Foundation of many later models: Heterogeneous Graph Transformer (HGT),
  Graph Transformer Networks, etc.
- Less common at extreme scale than GraphSAGE because attention adds compute
  per edge.

### When to use which

- **GraphSAGE**: large graphs, inductive setting, scaling to billions of
  nodes, OK with mean/max-pool aggregators.
- **GAT**: graphs where neighbor importance varies, moderate scale, when
  interpretability of which neighbors matter is valued (attention scores
  give a kind of explanation).
- **GCN**: small/medium fixed graph, transductive, for a strong simple
  baseline.
- **GIN**: when graph-isomorphism-discriminating expressiveness matters,
  e.g., molecular benchmarks like ZINC, OGB-MolHIV.

### Trade-offs

- **GraphSAGE pros**: scalability, inductive, simple, robust.
- **GraphSAGE cons**: sampling variance; max-pool not invariant to multiset
  cardinality (loses some info); LSTM aggregator only approximately
  permutation-invariant.
- **GAT pros**: principled neighbor weighting, multi-head capacity,
  inductive.
- **GAT cons**: more compute per edge; attention can degenerate (collapse
  to uniform or near-one-hot); sensitive to softmax temperature implicitly.

### Computational considerations at scale

- Always implement scatter/gather sparsely; never densify the adjacency.
- For GraphSAGE: precompute and cache k-hop neighborhoods or use efficient
  samplers (PyG's `NeighborLoader`, DGL's `MultiLayerNeighborSampler`).
- For GAT: edge-softmax must be implemented efficiently across each node's
  incoming edges; PyG and DGL provide kernels.
- For both: distributed training across machines requires careful graph
  partitioning (METIS, RandomNodePartitioner).

## 4.6 Common Pitfalls & Misconceptions

- **"GAT's attention is interpretable as feature importance."** It tells
  you which *neighbors* the model attended to for a given node, but
  attention weights are not necessarily faithful explanations — multiple
  works show that attention can be uncorrelated with downstream gradients.
  Use dedicated explanation tools (GNNExplainer, PGExplainer) for serious
  interpretability.
- **"GraphSAGE-LSTM is permutation-equivariant because we randomize the
  order during training."** It is *empirically* close to invariant after
  training, but not provably so. If exact equivariance matters (e.g.,
  scientific benchmarks where reproducibility across runs is required),
  use mean or max-pool.
- **"Sampling is just for training; at test time we use the full
  neighborhood."** Often true and is the recommended practice — but it
  causes a *train-test mismatch* in the distribution of aggregated
  features. Use larger samples or full-neighbor inference and check
  carefully whether your model has been calibrated for this.
- **"More heads always helps GAT."** Diminishing returns and additional
  compute. Typical: 4 or 8 heads in intermediate layers, 1 head with
  averaging in the final layer.
- **"GAT and GraphSAGE need self-loops."** GAT typically includes the node
  itself in the softmax (so attention can attend to self); GraphSAGE
  achieves a similar effect via concatenation with **h**_v in the update.
  Both work without explicit self-loops in the graph, but you must
  include the self contribution somewhere.
- **"All neighborhood-sampling strategies are equivalent."** Uniform
  sampling has high variance for important neighbors; importance sampling
  (FastGCN, AS-GCN, PPR-based as in PinSage) significantly improves
  quality.
- **"GraphSAGE is just GCN with sampling."** No: GraphSAGE *concatenates*
  the self vector with the aggregated neighbor vector before the linear
  transform, while GCN folds the self into the aggregation via self-loops
  and degree normalization. The concatenation gives more capacity to
  distinguish "what I am" from "what my neighbors are."

---

# Topic 5: Applications — Molecules, Social Networks, Recommendation Systems

## 5.1 Motivation & Intuition

Three of the most impactful application domains for GNNs are molecular
property prediction, social-network analysis, and recommendation. Each
exhibits a different combination of graph properties (size, density,
homogeneity, dynamics), different task structure (graph-, node-, edge-level),
and different stakes — drug discovery has multi-year, multi-billion-dollar
implications; recommendation runs in real time on hundreds of millions of
users.

We'll discuss each in enough detail to understand why GNNs are well-suited,
how the problem is formalized as a graph learning task, what architectural
choices industry practitioners typically make, and what the failure modes
look like.

## 5.2 Conceptual Foundations

### Molecular property prediction

- **Inputs**: a molecule, represented as a graph where nodes are atoms and
  edges are bonds. Node features include element, formal charge,
  hybridization, aromaticity, hydrogen count, chirality, in-ring flag.
  Edge features include bond order (single, double, triple, aromatic),
  conjugation, in-ring flag, stereo configuration.
- **Tasks**:
  - **Graph regression**: predict a continuous quantity for the whole
    molecule — solubility (log *S*), boiling point, free energy, binding
    affinity to a target protein (pIC50).
  - **Graph classification**: toxicity (yes/no), passes blood-brain
    barrier, mutagenic.
  - **Edge-level**: bond polarity, partial charges.
  - **Reaction prediction**: input two molecules → output a third (this is
    a graph-to-graph problem).
- **Data**: typically thousands to millions of molecules. Public benchmarks:
  QM9 (134k small organic molecules), MoleculeNet (suite), OGB-MolHIV (HIV
  inhibition, ~41k molecules), OGB-MolPCBA (multi-task, 437k molecules).

### Social networks

- **Inputs**: a single huge graph (Twitter, Facebook), with users as nodes,
  friendships/follows as edges. Node features include profile attributes,
  posting history embeddings, demographics. Edge features: relation type
  (friend, follower, blocked), interaction frequency, time since last
  interaction.
- **Tasks**:
  - **Node classification**: predict a user attribute (interests,
    political leaning), often for ad targeting.
  - **Link prediction**: friend recommendation.
  - **Community detection**: classify users into communities (related to
    node clustering).
  - **Anomaly / fraud detection**: detect bots, spam accounts, coordinated
    inauthentic behavior — often by detecting unusual subgraph patterns.

### Recommendation systems

- **Inputs**: a bipartite graph between users and items (movies, products,
  pins). Node features: user demographics + history, item content
  features (text, images). Edge features: implicit (click, view) or
  explicit (rating, like) feedback.
- **Tasks**:
  - **Link prediction**: will user *u* engage with item *i*?
  - **Node embedding**: learn user/item embeddings used downstream by ANN
    (approximate nearest neighbor) retrieval.
  - **Sequential recommendation**: user-item bipartite graph augmented
    with temporal structure.
- **Data scale**: industrial systems have hundreds of millions of users
  and millions of items.

### What's in common, and what differs

| Property | Molecules | Social | Recsys |
|---|---|---|---|
| Graph count | Many small | One huge | One huge bipartite |
| Avg. nodes | 10–100 | 10⁸ | 10⁸–10⁹ |
| Avg. degree | 2–4 | 10–10³ | 10–10² |
| Edge features | Critical (bond type) | Sometimes | Sometimes (ratings) |
| Setting | Inductive | Often transductive, increasingly inductive | Streaming |
| Common task | Graph regression/classification | Node classification, link prediction | Link prediction |
| Typical model | MPNN with edge features (DimeNet, Schnet, MPNN) | GCN, GAT, GraphSAGE, hetero variants | GraphSAGE-style sampling (PinSage) |
| Key challenge | Capturing 3D geometry; long-range interactions | Heterophily; adversarial attacks | Cold start; latency; scale |

## 5.3 Mathematical Formulation (Application-Specific Heads)

### Molecular property prediction (graph-level regression)

After *L* GNN layers, each atom *v* has embedding **h**_v^(*L*) ∈ ℝ^*d*.
Pool to a graph embedding:

  **h**_*G* = Σ_v **h**_v^(*L*)   (sum-pool — most expressive),

or alternatively a learned set-pool (Set2Set, DeepSet), or hierarchical
pool (DiffPool).

Final prediction:

  ŷ = MLP( **h**_*G* ).

Loss: L2 (MSE) for regression of continuous properties; binary cross-entropy
for classification.

For state-of-the-art molecular GNNs (DimeNet, GemNet, SphereNet, Equiformer)
the message function is enriched with **3D geometric** information —
pairwise distances and angles — and the architecture is built to be
invariant or equivariant to 3D rotations and translations (E(3)-equivariance).
This is more than vanilla MPNN but follows the same structural recipe.

### Link prediction (recommendation, drug-target interaction, knowledge graph
completion)

After GNN training, each node *v* has embedding **z**_v. The score for an
edge (*u*, *v*):

- **Dot product**: *s*(*u*, *v*) = **z**_u · **z**_v. Cheap; suitable for
  ANN retrieval.
- **Cosine similarity**: *s*(*u*, *v*) = (**z**_u · **z**_v) / (‖**z**_u‖ ‖**z**_v‖).
- **Bilinear**: *s*(*u*, *v*) = **z**_uᵀ *M* **z**_v, with *M* learned.
- **MLP head**: *s*(*u*, *v*) = MLP( [**z**_u ‖ **z**_v] ) — most
  expressive but no longer dot-product-friendly for retrieval.
- **DistMult / ComplEx / RotatE** (knowledge graphs with relations
  *r*): *s*(*u*, *r*, *v*) = ⟨**z**_u, **z**_r, **z**_v⟩ (trilinear) etc.

Loss for link prediction is typically a **negative-sampling** binary
cross-entropy or a **margin-based** ranking loss (BPR — Bayesian
Personalized Ranking, popular in recommendation):

  L = − Σ_{(u, v) ∈ E_train} log σ( *s*(*u*, *v*) − *s*(*u*, *v⁻*) ),

where *v⁻* is a randomly sampled non-neighbor.

### Node classification (social, fraud)

After GNN, node *v* has logits ℓ_v = *W*_out **h**_v^(*L*); apply softmax;
cross-entropy loss.

Often the loss is restricted to a small labeled subset — semi-supervised
learning. Augmentations: pseudo-labeling, label propagation as a
post-processing step.

## 5.4 Worked Examples

### Example 1 — Predicting log *P* (lipophilicity) of a small molecule

**Molecule**: ethanol (CH₃CH₂OH). Two carbons, one oxygen, six hydrogens.
For simplicity, consider the heavy-atom skeleton: C₁ — C₂ — O.

- Node features (4-dim, one-hot for {C, O, H, other}, with H pruned):
  C₁ = (1, 0, 0, 0), C₂ = (1, 0, 0, 0), O = (0, 1, 0, 0).
- Edge features (3-dim, one-hot {single, double, triple}):
  C₁–C₂ single, C₂–O single.

Apply a 2-layer MPNN with edge-conditioned messages. Schematically, layer
*l*'s message for an edge (*u*, *v*) with edge feature **e**_{uv}:

  **m**_{u→v} = MLP_msg( CONCAT( **h**_u, **h**_v, **e**_{uv} ) ).

After two layers, each node has an embedding that encodes its 2-hop
chemical environment. Sum-pool gives **h**_*G*; an MLP outputs a scalar
log *P* prediction.

Training: hundreds of thousands of (molecule, log *P*) pairs from databases
like ChEMBL. Loss: MSE. After training, the model generalizes to unseen
small molecules.

### Example 2 — Pinterest PinSage architecture (sketch)

- Graph: bipartite, ~3 billion pins + ~1 billion boards + ~18 billion
  edges (a pin pinned to a board). Modeled as a unipartite pin-pin graph
  via folding.
- Sampling: random-walk based importance sampling: for each target pin,
  perform short random walks; nodes with high visit count are "important"
  neighbors.
- Aggregator: GraphSAGE-style mean of importance-weighted neighbor
  features, concatenated with the self vector, projected via a learned
  matrix and ReLU.
- 2-layer architecture; embedding dim 256.
- Training: maximum-margin loss with hard negative samples (engaged users'
  pins far in embedding space are mined as informative negatives).
- Output: 256-d pin embedding for each pin in the catalogue.
- Serving: ANN (nearest neighbor) lookup retrieves recommended pins for any
  source pin in milliseconds.

End result (per the PinSage paper): substantially better recommendation
quality on offline metrics than the prior production system, deployed at
scale.

### Example 3 — Friend recommendation as link prediction

- Graph: existing user–user friendship graph; node features include
  embeddings of recent activity, demographics.
- Train a 2-layer GraphSAGE to produce user embeddings.
- Loss: contrastive — connected pairs (existing friends) have high
  similarity; random pairs have low similarity.
- At serving time: for each user *u*, retrieve top-*k* candidate friends
  by ANN over the embedding space, then rerank with a richer model
  (e.g., a GBM with hand-crafted features).
- Cold start: for a brand-new user with no edges, the GNN cannot run any
  message passing. Use the user's profile features through an MLP to
  produce an initial embedding; once they make friends, switch to GNN.

## 5.5 Relevance to Machine Learning Practice

### Practical considerations across applications

- **Featurization is half the battle.** A well-featurized node (rich,
  informative initial vector) lets the GNN spend its capacity on
  *combination*, not on inferring features from structure alone.
  Chemistry: RDKit feature dictionaries. Recsys: rich content embeddings
  (CLIP-style image embeddings, BERT text embeddings).
- **Negative sampling matters enormously** for link-prediction tasks. Easy
  negatives (random pairs) train the model to a useless local optimum
  quickly; hard negatives (close in embedding space but not actually
  linked) drive most of the learning.
- **Evaluation metrics differ wildly by task.**
  - Molecules: ROC-AUC, RMSE, R².
  - Recsys: Recall@k, NDCG@k, hit rate, MAP@k.
  - Social link prediction: AUC, average precision, Hits@k.
  - Always benchmark against simple baselines: random, popularity, MLP on
    features only, classical graph features (PageRank, Jaccard, Adamic–Adar).
- **Scale**:
  - Molecules: small per-graph but many graphs → batch graphs together,
    pad or use batching by edge index concatenation.
  - Social/recsys: huge single graph → sampling, distributed training,
    careful memory management.
- **Latency at serving**:
  - Molecules: forward pass of a small GNN on a small graph is millisecond-scale.
  - Recsys: precompute embeddings offline; serve via ANN; recompute
    incrementally for changed users/items.

### When *not* to use a GNN

- **Recsys with very weak collaborative signal** (very sparse user-item
  matrix, brand-new platform): a content-based model on item attributes is
  often stronger.
- **Static node-classification problems where graph structure is
  uninformative**: just use an MLP.
- **Drug discovery problems requiring 3D conformational reasoning**: a
  vanilla 2D molecular graph GNN misses information; use 3D-aware models
  (DimeNet, Equiformer), or hybrid with QM/MM.
- **Knowledge graph completion with few entities and many relations**:
  TransE / DistMult / RotatE (non-GNN, tensor factorization style) are
  classical strong baselines; GNN variants (R-GCN, CompGCN) help when node
  features matter.

### Trade-offs in deployment

- **Online learning**: GNNs are harder to update incrementally than
  factorization models. Solutions: streaming GNNs (TGN, JODIE), or
  periodic batch retraining.
- **Cold start**: serious issue for both items and users. Use content
  features as fallback; warm up with a few interactions.
- **Cost vs. quality**: a deep GNN with attention is heavier than a
  simple matrix factorization baseline; ROI must justify the engineering
  investment. Many companies deploy GNNs only after exhausting simpler
  models.

## 5.6 Common Pitfalls & Misconceptions

- **"My recommender beats baselines on offline metrics, so it will win in
  production."** Online metrics (CTR, watch time, revenue) can move
  differently. Always run an A/B test.
- **"More signal in the graph is always better."** Adding noisy edges
  (e.g., low-confidence relationships) often *hurts* performance and can
  open adversarial attack surfaces.
- **"My molecule model with 90% accuracy is ready for the lab."** Distribution
  shift between training (drug-like molecules in databases) and the
  generative chemistry workspace is severe; a 90% in-distribution accuracy
  can be 50% on a novel chemotype. Use careful out-of-distribution splits
  (scaffold splits, time splits) for evaluation.
- **"3D geometry doesn't matter for GNNs on molecules."** For some tasks
  (formal charge, log *P*) 2D is fine. For others (binding affinity,
  conformational energy) 3D-aware GNNs are dramatically better.
- **"PinSage-style sampling is overkill for my graph."** Maybe — but check
  the size and degree distribution. If the max degree is 100k+, full-
  neighborhood aggregation will OOM your GPU; sampling is mandatory.
- **"Heterogeneous information (multi-typed nodes/edges) can be ignored."**
  Often the most informative signal is *which type* of relation links two
  nodes (rated 5 stars vs. clicked vs. dismissed). Use heterogeneous GNNs
  (R-GCN, HGT) when possible.
- **"My fraud detection GNN has high recall, so we caught all the bad
  guys."** Adversaries adapt: they may add edges to legitimate accounts to
  blend in, or coordinate small clusters to mimic normal users. GNN
  models for adversarial domains need adversarial training and frequent
  retraining.
- **"Link prediction = connecting any two nodes the model scores high."**
  Without negative sampling carefully crafted to match the deployment
  distribution, the model's "high score" can be meaningless out-of-sample.

---

# Interview Preparation Section

This section provides a comprehensive set of interview questions on Graph
Neural Networks, with answers at the depth expected for a machine-learning
engineer or applied-scientist interview. Questions are organized by
difficulty tier and category.

---

## Tier 1 — Foundational Questions

### Q1.1 What is a Graph Neural Network and why can't a regular feed-forward
network or CNN do the same job?

**Answer.** A Graph Neural Network is a neural architecture that operates on
graph-structured data — sets of nodes connected by edges, with per-node
features (and optionally per-edge or graph-level features). It produces
representations that respect two structural properties of graphs:

1. **Permutation invariance / equivariance**: relabeling the nodes does not
   change the prediction (for graph-level tasks) or only relabels the
   per-node outputs the same way (for node-level tasks).
2. **Variable size and shape**: the number of nodes and edges varies
   between inputs.

A regular feed-forward network requires fixed-size inputs and is sensitive
to feature ordering — feeding the same molecule with atoms in different
orders would give different predictions. A CNN assumes a regular grid
(every "pixel" has the same number of neighbors arranged identically),
which does not hold on graphs. RNNs assume sequential structure, also
absent.

GNNs achieve permutation invariance by computing node updates as
**permutation-invariant functions of the node's neighborhood multiset** —
e.g., sum, mean, max, attention-weighted sum.

### Q1.2 Explain the message-passing framework in three sentences.

**Answer.** Each layer of a message-passing GNN does three things: (1) for
each edge, compute a *message* as a function of the two endpoints'
representations (and possibly the edge's features); (2) at each node,
*aggregate* the incoming messages with a permutation-invariant operator
(sum / mean / max / attention); (3) at each node, *update* the
representation by combining the previous representation with the
aggregated message. After *L* layers, each node's representation depends
on its *L*-hop neighborhood; for graph-level tasks, a final permutation-
invariant *readout* pools all node representations into a single vector.

### Q1.3 What's the difference between transductive and inductive learning on
graphs?

**Answer.** In **transductive** learning, all nodes (training, validation,
and test) are part of a single fixed graph that is fully observed. The
model has access to the structure of the entire graph during training.
Test-time predictions are made for nodes whose features and graph
neighborhood the model has already "seen." Vanilla GCN is naturally
transductive because its propagation operator depends on the specific
graph's Laplacian.

In **inductive** learning, the model must generalize to nodes — or whole
graphs — unseen at training time. It cannot rely on any node-specific
parameters, only on functions of node features and neighborhood structure.
GraphSAGE, GAT, and GIN are inductive by design: their parameters are
shared across all nodes and graphs.

For molecules, inductive is natural — train on known molecules, predict on
novel ones. For citation networks, transductive can suffice for closed-
universe tasks (propagate labels in a fixed corpus). Most modern industrial
applications require inductive setups.

### Q1.4 Why is permutation invariance such an important constraint?

**Answer.** A graph is the same mathematical object regardless of how its
nodes are labeled. If our model produces different predictions for the
same graph under different labelings, we have to either (a) define a
canonical labeling — but no canonical labeling is computable in
polynomial time in general; or (b) train the model on every possible
labeling — exponential cost. The cleaner solution is to design the model
so that permutation invariance is *guaranteed* by construction, by using
permutation-invariant aggregation (sum, mean, max, attention over a
multiset) at every layer.

### Q1.5 What is over-smoothing in GNNs?

**Answer.** Over-smoothing is the empirical and theoretical observation that
as you stack more GNN layers, the hidden representations of all nodes
become increasingly similar to each other, eventually losing
discriminative power. The mechanism: each layer is essentially a smoothing
operator on the graph. Composing many such operators is analogous to
applying a low-pass filter many times, which drives all signals toward
the dominant eigenvector — typically a near-uniform vector. As a result,
deep vanilla GCNs (5+ layers) often perform *worse* than 2-layer ones on
node classification.

Mitigations: residual connections (skip past layers), Jumping Knowledge
networks (concatenate features from all layers), GCNII (initial residual +
identity mapping), normalization (PairNorm), or different architectures
designed for depth (GPRGNN, APPNP).

### Q1.6 What are the differences between sum, mean, and max aggregation, and
what does each lose?

**Answer.**

- **Sum** aggregation preserves both the *cardinality* (number of
  neighbors) and the *sum* of feature values. Combined with an MLP, it
  can implement any injective function on multisets (Xu et al. 2019), so
  it is the most expressive. It also can blow up in magnitude on high-
  degree nodes, so sometimes it requires normalization.
- **Mean** loses cardinality: a node with two neighbors having feature *x*
  looks the same as a node with one neighbor having feature *x*. For
  homogeneous neighborhoods this is fine; for distinguishing graph
  structure it is strictly weaker than sum.
- **Max** captures the most extreme feature values along each dimension
  but discards everything else — useful for "is there a neighbor with
  property X?" detection but very lossy.

In the Weisfeiler-Leman framework, only sum (with an injective
postprocessor) reaches the discriminative ceiling of 1-WL.

---

## Tier 2 — Mathematical & Derivation Questions

### Q2.1 Derive the GCN update rule from spectral filtering.

**Answer.** Start from spectral convolution of a graph signal **x** ∈ ℝ^*N*
with a filter *g*_θ:

  *g*_θ ⋆ **x** = *U* *g*_θ(Λ) *U*ᵀ **x**,

where *L*_sym = *I* − *D*⁻¹ᐟ² *A* *D*⁻¹ᐟ² = *U* Λ *U*ᵀ.

Step 1 — Polynomial parametrization (ChebNet). Approximate *g*_θ(Λ) by a
truncated Chebyshev expansion:

  *g*_θ(Λ) ≈ Σ_{k=0}^{K} θ_k *T*_k( Λ̃ ),    Λ̃ = (2 / λ_max) Λ − *I*.

Crucially, this avoids eigendecomposition because

  *U* *T*_k( Λ̃ ) *U*ᵀ = *T*_k( *L̃* ),    *L̃* = (2 / λ_max) *L*_sym − *I*.

Step 2 — First-order approximation: *K* = 1, λ_max ≈ 2.

  *L̃* = *L*_sym − *I* = − *D*⁻¹ᐟ² *A* *D*⁻¹ᐟ².

So *g*_θ ⋆ **x** ≈ θ_0 **x** + θ_1 (− *D*⁻¹ᐟ² *A* *D*⁻¹ᐟ²) **x**.

Step 3 — Single-parameter constraint: set θ = θ_0 = − θ_1 to reduce
overfitting and stabilize:

  *g*_θ ⋆ **x** ≈ θ ( *I* + *D*⁻¹ᐟ² *A* *D*⁻¹ᐟ² ) **x**.

Step 4 — Renormalization trick. Substitute *I* + *D*⁻¹ᐟ² *A* *D*⁻¹ᐟ² with
*D̃*⁻¹ᐟ² *Ã* *D̃*⁻¹ᐟ² where *Ã* = *A* + *I* and *D̃*ᵢᵢ = Σⱼ *Ã*ᵢⱼ. The
new operator has eigenvalues in [0, 1], so repeated applications are
stable.

Step 5 — Generalize to multi-channel signals (matrix *X* ∈ ℝ^(*N*×*d*) and
multiple output channels):

  *H*^(*l*+1) = σ( *D̃*⁻¹ᐟ² *Ã* *D̃*⁻¹ᐟ² *H*^(*l*) *W*^(*l*) ).

This is the GCN.

### Q2.2 Show that the symmetric normalized Laplacian's eigenvalues lie in
[0, 2].

**Answer.** Let *L*_sym = *I* − *D*⁻¹ᐟ² *A* *D*⁻¹ᐟ². Let
*M* = *D*⁻¹ᐟ² *A* *D*⁻¹ᐟ². Eigenvalues of *L*_sym = *I* − *M* are 1 − μ_k
where μ_k are the eigenvalues of *M*. We need 1 − μ_k ∈ [0, 2], i.e.,
μ_k ∈ [−1, 1].

The matrix *M* is the symmetric normalized adjacency. For undirected
non-bipartite graphs, μ_k ∈ [−1, 1]; the bound μ_k ≤ 1 follows from the
fact that *D*⁻¹ *A* (the random-walk transition matrix, similar to *M*) is
stochastic, and the Perron–Frobenius theorem gives spectral radius 1 for
connected graphs. The bound μ_k ≥ −1 comes from Cauchy–Schwarz applied to
the bilinear form **x**ᵀ *M* **x**, which equals 2 Σ_{(i,j) ∈ E} *x*_i *x*_j /
√(*d*_i *d*_j); this is bounded below by −Σ_i *x*_i² = −‖**x**‖² when
**x** is normalized. (Full proof uses the Cheeger inequality framework.)

So eigenvalues of *L*_sym lie in [0, 2], with 0 attained on connected
components and 2 attained iff the graph contains a bipartite connected
component.

### Q2.3 Prove the equivariance of the standard GNN layer.

**Answer.** Let **h**ᵥ^(*l*+1) = φ( **h**_v^(*l*), ⊕_{u ∈ N(*v*)} ψ(**h**_u^(*l*), **h**_v^(*l*)) ),
with ⊕ permutation-invariant on multisets and φ, ψ pointwise functions.

Let *P* be a permutation matrix; let σ be the corresponding permutation of
nodes (so σ(*i*) is the new label of old node *i*). After permutation, the
new feature matrix is *X*' = *PX* and the new adjacency is *A*' = *PAP*ᵀ.
We have N'(σ(*v*)) = σ(N(*v*)).

Then for any node *v*:

  **h**'_{σ(v)}^(*l*+1)
    = φ( **h**'_{σ(v)}^(*l*), ⊕_{u' ∈ N'(σ(v))} ψ(**h**'_{u'}^(*l*), **h**'_{σ(v)}^(*l*)) )
    = φ( **h**_v^(*l*), ⊕_{u' ∈ σ(N(v))} ψ(**h**_{σ⁻¹(u')}^(*l*), **h**_v^(*l*)) )    (by inductive hypothesis)
    = φ( **h**_v^(*l*), ⊕_{u ∈ N(v)} ψ(**h**_u^(*l*), **h**_v^(*l*)) )    (relabeling the multiset indices)
    = **h**_v^(*l*+1).

So the new representation at the permuted index σ(*v*) equals the old
representation at *v* — i.e., *H*'^(*l*+1) = *P H*^(*l*+1), which is
permutation equivariance. The induction base is *H*^(0) = *X* and *X*' =
*PX*.

### Q2.4 Write down the GAT attention computation and identify which step
ensures permutation equivariance.

**Answer.**

  **z**_v = *W* **h**_v.
  *e*_vu = LeakyReLU( **a**ᵀ [ **z**_v ‖ **z**_u ] ),    *u* ∈ N̄(*v*).
  α_vu = exp(*e*_vu) / Σ_{k ∈ N̄(*v*)} exp(*e*_vk).
  **h**_v^(new) = σ( Σ_{u ∈ N̄(*v*)} α_vu **z**_u ).

Permutation equivariance is ensured by the **softmax-and-weighted-sum**
combination: both the softmax denominator (a sum over the neighborhood
multiset) and the weighted sum are symmetric in the multiset of
neighbors. The linear projection *W* and the per-edge attention scoring
ψ are pointwise per-node and per-edge respectively, so they don't
introduce any ordering dependence.

If you replaced the softmax with, say, an LSTM applied to the ordered
sequence of neighbors, you would lose equivariance.

### Q2.5 Why does mean aggregation fail to be injective on multisets, with
a concrete counterexample?

**Answer.** Consider node features as scalars. Multisets {{1, 1}} and
{{1, 1, 1, 1}} both have mean 1. So a mean-aggregator produces identical
outputs for these two multisets — they are indistinguishable to the layer.

The same issue persists in higher dimensions: any two multisets that are
positive scalar multiples of each other (interpreted as a count vector)
have the same mean. This is exactly why mean aggregation lies strictly
below 1-WL in expressive power.

Sum aggregation gives 2 vs 4 — distinguishable.

### Q2.6 Why does the renormalization trick (using *D̃*⁻¹ᐟ² *Ã* *D̃*⁻¹ᐟ²
instead of *I* + *D*⁻¹ᐟ² *A* *D*⁻¹ᐟ²) help?

**Answer.** The pre-renormalization operator *I* + *D*⁻¹ᐟ² *A* *D*⁻¹ᐟ²
has eigenvalues in [0, 2]. Repeated multiplication by such a matrix can
amplify components with eigenvalue > 1, leading to numerical instability
and exploding activations — especially in deep stacks.

The renormalized operator *D̃*⁻¹ᐟ² *Ã* *D̃*⁻¹ᐟ² (where *Ã* = *A* + *I*
and *D̃* counts the self-loop) has eigenvalues in [0, 1]. This makes
deeper stacks better-behaved numerically. Conceptually, it corresponds to
*first* adding self-loops, *then* normalizing as if they were ordinary
edges — a uniform treatment that is also closer to a proper normalized
random-walk matrix on the augmented graph.

### Q2.7 Bound the receptive field of a *L*-layer GNN.

**Answer.** A node *v*'s representation after layer 1 depends on *v* and
its 1-hop neighborhood. By induction, after layer *L* it depends on *v*
and the union of all *k*-hop neighborhoods for *k* ≤ *L*. So the
receptive field of *v* after *L* layers is *v*'s ball of radius *L* in
the graph, B_*L*(*v*).

The size of this ball is bounded above by *d̄*^*L* where *d̄* is the
maximum degree, but in practice for sparse graphs is much smaller (and
in social networks often hits "small-world" saturation: only 4–6 hops
suffice to reach most of the graph).

### Q2.8 In what sense are message-passing GNNs bounded by the 1-WL test?

**Answer.** The 1-Weisfeiler-Leman test refines a coloring of nodes: the
color at iteration *t* + 1 is a hash of the multiset of colors of the
node and its neighbors at iteration *t*. Two graphs with identical color
histograms after refinement are "1-WL-equivalent."

A message-passing GNN does the exact same kind of computation, but with
continuous messages and a learned hash. Xu et al. (2019) proved:

> If the 1-WL test cannot distinguish two graphs, neither can any
> standard MPNN, no matter how many layers or how wide.

Conversely, with sum aggregation + an MLP (their GIN architecture), an
MPNN can match 1-WL discriminative power. This bound implies, e.g., that
MPNNs cannot count cycles of length ≥ 3 in general, cannot distinguish
certain regular graph pairs, and need higher-order extensions (k-WL,
k-FWL, GNNs with subgraph counts) for those tasks.

---

## Tier 3 — Applied / System-Design Questions

### Q3.1 You're tasked with predicting whether two LinkedIn users will
connect within the next 30 days. Walk me through your approach.

**Answer.**

**Problem framing.** This is link prediction on a temporal social graph.
Treat the problem as a binary classifier on candidate user pairs.

**Data and features.**

- Snapshot the user-user graph at time *t*.
- Labels: 1 if a new edge appears between *u* and *v* in [*t*, *t* + 30
  days], 0 if no edge appears in the same window. Sample positives from
  observed new edges; sample negatives from non-edges (with appropriate
  class imbalance handling — e.g., 1:5 to 1:50 ratio with hard negative
  mining).
- Node features: profile (industry, location, headline embedding from a
  text model), history (recent profile views, search queries),
  demographics.
- Edge features: existing connection count between *u* and *v*'s neighbors
  (mutual friends), interaction frequency, time since last interaction.

**Architecture.**

- 2-layer GraphSAGE with neighborhood sampling (e.g., 25 / 10 neighbors
  per layer); inductive so the model can serve users whose neighborhoods
  change.
- Concatenate node embeddings with classical pairwise graph features
  (Adamic–Adar, common neighbors count) for the head.
- Head: an MLP on [**z**_u ‖ **z**_v ‖ pairwise-features].

**Training.**

- Binary cross-entropy with hard negative mining: at each step, generate
  some random negatives but also some "hard" negatives — pairs that look
  similar in embedding space but didn't actually connect.
- Use temporal splits: train on edges added in [*t* − 90, *t* − 30 days],
  validate on [*t* − 30, *t*], test on [*t*, *t* + 30].

**Evaluation.**

- AUC and Precision@k (do users actually click on the top-k recommended
  connections?). Offline AUC is a sanity check; the real evaluation is an
  online A/B test on connection click-through and acceptance rate.

**Serving.**

- Precompute user embeddings nightly via batch inference (the embedding
  for an active user is recomputed daily, for an inactive user weekly).
- For each user, run ANN over the embedding space to retrieve a few
  hundred candidates; rerank with the full MLP head + features.

**Cold start.**

- New users have no edges. Use a separate "warmup" model based purely on
  profile features until they accumulate a minimum number of connections.

### Q3.2 Why might a GNN underperform a plain MLP on a node classification
benchmark, and how would you diagnose this?

**Answer.** Common reasons:

1. **Heterophily**: the graph's connected nodes do *not* share labels.
   GNNs implicitly assume homophily through smoothing. On heterophilic
   graphs (e.g., dating networks where men and women are often connected
   but have opposite labels for the gender attribute) smoothing actively
   hurts.
2. **Strong node features, weak graph signal**: if features alone are
   very predictive, the additional structural information may be noise.
3. **Over-smoothing**: too many layers; node embeddings have collapsed.
4. **Bug**: silently dropping self-loops, wrong normalization, ignoring
   edge features.
5. **Distribution shift**: train and test nodes are from different
   regions of the graph; the model overfits to local structure.

**Diagnosis steps**:

- Compare a 2-layer MLP-on-features against the GNN on the same labels.
- Compute the **edge homophily ratio**: fraction of edges where both
  endpoints share the same label. Below ~0.4, suspect heterophily; switch
  to a heterophily-aware architecture.
- Try shallower (1 layer) and deeper (4-6 layers) versions; if shallower
  is best, suspect over-smoothing or weak structure.
- Inspect attention or aggregation weights on hard examples; do they make
  sense?
- Use a probing classifier on per-layer embeddings to localize where
  performance drops.

### Q3.3 Pinterest has billions of pins. How do you train a GNN at this
scale?

**Answer.** The PinSage approach (Ying et al. 2018):

1. **Bipartite folding**: convert the pin–board bipartite graph to a
   pin–pin graph (two pins are adjacent if they share a board).
2. **Random-walk based sampling**: for each target pin, perform short
   random walks; nodes visited frequently are "important" neighbors.
   Sample a fixed number of these (e.g., 50). This is importance
   sampling — far better than uniform.
3. **Producer–consumer pipeline**: GPU does forward passes on minibatches;
   CPU samples and prepares the next minibatch concurrently.
4. **Hard negative mining**: random negatives are too easy; mine pins
   that are close in current embedding space but not actually clicked.
5. **MapReduce inference**: at inference time, distribute the graph and
   compute embeddings in parallel across many machines, with each node
   producing its embedding once.
6. **Margin loss**: train with max-margin loss on positive vs negative
   pairs.
7. **Architecture**: 2-layer GraphSAGE with 256-dim embeddings.

For other ultra-large-scale settings, modern alternatives include:

- **Cluster-GCN**: partition the graph into clusters via METIS, train on
  one cluster (or a few) at a time. Cheap, but loses inter-cluster
  edges.
- **GraphSAINT**: subgraph sampling with bias correction.
- **GBP / SIGN / SGC**: precompute multi-hop propagated features once,
  then train a simple MLP — eliminates the GNN at inference time entirely
  (very fast serving).

### Q3.4 What are the main differences between GraphSAGE and GAT, and when
would you pick one over the other?

**Answer.**

| Aspect | GraphSAGE | GAT |
|---|---|---|
| Aggregation | Mean / max-pool / LSTM | Attention-weighted sum |
| Neighbor weighting | Uniform | Learned per edge |
| Sampling | Built-in fixed-size sampling | Full neighborhood by default |
| Scaling | Excellent for huge graphs | Moderate; full attention is more expensive |
| Inductive | Yes | Yes |
| Interpretability | Mostly black-box | Attention provides per-edge weights |

**Pick GraphSAGE** when:
- The graph is huge (tens of millions of nodes or more).
- You need inductive predictions on streaming graphs.
- All neighbors carry similar importance, so uniform aggregation is fine.

**Pick GAT** when:
- Neighbors vary substantially in relevance and you want the model to
  learn which to weight.
- The graph is small enough that full-neighborhood attention is feasible.
- You want attention scores as a debugging or interpretability aid.

In practice, hybrid models exist: GraphSAGE-style sampling + attention
within sampled neighborhoods (e.g., AGNN, FastGAT).

### Q3.5 You train a GNN on a benchmark and get 92% accuracy. How do you
sanity-check the result?

**Answer.**

1. **Compare against MLP baseline**: same features, no graph. If the gap
   is small, your "GNN advantage" may be an artifact of better features
   or hyperparameters, not the graph structure.
2. **Compare against a graph-blind label-propagation baseline**: pure
   structural propagation. If LP is competitive, your model isn't using
   the features much.
3. **Shuffle labels** of training nodes within each class: model should
   crash to random if the labels are necessary. (Sanity check that
   training is in fact label-driven.)
4. **Randomize the graph**: replace the adjacency with a random graph
   with same degree distribution. If accuracy stays high, your model is
   ignoring structure; if it drops, structure was important.
5. **Cross-validate split**: re-evaluate with different train/val/test
   splits (random and, where appropriate, scaffold or temporal splits)
   to rule out lucky splits.
6. **Inspect calibration**: a model with high accuracy can still be
   poorly calibrated. Check ECE / reliability diagrams, especially if
   probabilities are used downstream.
7. **Look for label leakage**: in transductive settings, labeled nodes'
   features can leak through message passing; ensure the model's
   propagation isn't trivially memorizing the training labels.

### Q3.6 How would you build a fraud-detection GNN for a bank?

**Answer.**

**Problem.** Detect fraudulent transactions/accounts in a heterogeneous
graph of customers, merchants, devices, IPs, and transactions.

**Graph construction.**

- Node types: customers, merchants, devices, IPs, accounts.
- Edge types: customer–transaction, transaction–merchant, customer–device,
  customer–IP, etc.
- Edge features: amount, timestamp, MCC code, channel.

**Architecture.**

- Heterogeneous GNN: R-GCN or HGT, where each edge type has its own
  weight matrix; supports the multi-relational nature.
- 2–3 layers, with residual connections to combat over-smoothing.
- Concatenate hand-crafted features (transaction velocity, geolocation
  mismatch, time-since-account-creation).

**Labels.** Confirmed fraud cases (typically very rare — < 1%). Use a
chargeback signal as a label. This is severe class imbalance; use focal
loss or weighted cross-entropy.

**Training.**

- Mini-batch with heterogeneous graph sampling (DGL or PyG support).
- Online metrics: precision-at-budget (how many flagged transactions can
  the analysts review per day?).

**Adversarial concerns.**

- Fraud rings will adapt. Retrain frequently (daily or weekly).
- Add adversarial training (Nettack-style perturbations) to harden the
  model.
- Monitor for shifts: if training graph distribution drifts from
  serving, retrain.

**Explainability.** Regulatory pressure often requires explanations. Use
GNNExplainer or PGExplainer to identify the subgraph (e.g., the
fraud-ring substructure) that drove the prediction.

---

## Tier 4 — Debugging & Failure-Mode Questions

### Q4.1 Your 6-layer GCN trains to 99% training accuracy but only 60% test
accuracy. What's happening?

**Answer.** Likely a combination of:

1. **Overfitting from too many parameters** with too few labels —
   semi-supervised settings have very few labels, and 6 layers stack
   many *W* matrices.
2. **Over-smoothing**: by layer 6, all node embeddings have collapsed
   toward a near-uniform vector, so training accuracy is high only
   because the model is memorizing the labeled set via residual capacity
   somewhere (or the small labeled subset is actually distinguishable
   even after smoothing). Test accuracy collapses because everything
   else looks identical.
3. **No regularization**: missing dropout, no weight decay, no label
   smoothing.
4. **Distribution mismatch** between train and test nodes (e.g., a graph
   community split where train nodes come from one community).

Fixes (in order):

- Reduce to 2 layers (the original GCN paper sweet spot).
- Add residual / jumping-knowledge connections if depth is needed.
- Add dropout (on features and on the propagation matrix — DropEdge).
- Add weight decay.
- Inspect embedding similarity across all node pairs; if high, confirm
  over-smoothing.
- Re-split the data more carefully.

### Q4.2 Your GAT model's attention weights are nearly uniform across
neighbors. What does that mean?

**Answer.** Possible causes:

1. **Initialization / scale**: the attention vector **a** and projection
   *W* may be initialized so that pre-softmax scores have very small
   magnitude, making softmax outputs near-uniform. Fix: check init
   scheme (Glorot for **a** is standard); also consider scaling attention
   logits by a temperature.
2. **Features are uninformative for attention**: if all neighbors look
   alike to the model in the projected space, nothing distinguishes
   them. Try richer projections (multi-head, larger hidden dim) or add
   edge features into the attention scoring.
3. **Symmetry**: in a regular structure (e.g., ring graph with identical
   features), all neighbors really *are* equivalent.
4. **Training collapsed**: optimizer may have driven attention into a
   degenerate local minimum where uniform attention is optimal w.r.t.
   the training loss (because, e.g., mean is "good enough").

Diagnosis: compare against GCN (which uses uniform-degree-normalized
weights) — if GAT and GCN perform identically, attention is providing no
real signal; you might be paying the compute cost for nothing.

### Q4.3 Your GraphSAGE model performs worse with neighborhood sampling
than with full neighborhoods at test time. Why?

**Answer.** This is the **train-test mismatch** problem. During training
you used a sample size *s*, so the aggregator was trained to combine *s*
features. At test time with full neighborhoods (much larger than *s*),
the input distribution to the aggregator changes — values may have very
different magnitudes (especially for sum), means may have lower variance,
etc. The model is being asked to operate on a distribution it never saw.

Fixes:

- Use the same sampling at test time.
- Use mean aggregation (less sensitive to neighborhood size than sum).
- Train with a *range* of sample sizes (curriculum sampling) so the
  model is robust.
- Use normalized aggregators (e.g., L2-normalize after each layer).

### Q4.4 You add edge features to your model but performance drops. Why?

**Answer.** Possibilities:

1. **Ill-conditioned encoding**: edge features may be on different scales
   from node features. Normalize or use a per-feature embedding.
2. **Increased parameter count → overfitting**: more weights, same
   amount of data. Add regularization, reduce hidden dim, or share
   weights across edge types.
3. **Bug**: edge features may be wrongly aligned (e.g., reversed for
   undirected edges) and adding them is injecting noise.
4. **Edge features dominate but are noisy**: if the model is using edge
   features but they are themselves unreliable, performance degrades.
5. **Architecture choice**: simply concatenating edge features into the
   message function may not be the best way to use them; consider
   edge-conditioned filters, gated mechanisms, or multi-relational GNNs
   (R-GCN).

### Q4.5 At inference time, a GAT node has 100,000 neighbors and the
forward pass OOMs. What do you do?

**Answer.**

- **Sampling**: switch to neighborhood sampling at inference (matching
  the training setup if you trained with sampling).
- **Sparse attention kernels**: ensure you're using PyTorch Geometric's
  or DGL's sparse attention implementation, not a dense one.
- **Top-k attention**: restrict each node to its top-*k* most important
  neighbors based on an inexpensive prior (e.g., recent interactions).
- **Hierarchical aggregation**: cluster neighbors and aggregate at the
  cluster level first, then attend to clusters.
- **Architecture change**: replace GAT with a more scalable alternative
  for hub nodes (GraphSAGE-pool, SIGN, GBP).
- **Multi-step inference**: run the model in chunks, partitioning the
  neighborhood and combining results.

### Q4.6 You see a sudden drop in offline AUC for your recommendation GNN
after a graph update. What's the diagnostic process?

**Answer.**

1. **Check graph health**: did the update introduce many missing nodes,
   wrong edges, schema changes? Run a graph-statistics dashboard
   (degree distribution, average shortest path, number of components)
   and compare before/after.
2. **Compute embedding drift**: embed a fixed set of users with the old
   and new graph; if drift is large for unchanged users, the propagation
   has been disrupted.
3. **Check feature pipelines**: feature drift is often confused with
   model drift. Are user features computed identically?
4. **A/B retest**: roll back the graph (or use a snapshot) to confirm
   the cause.
5. **Inspect samples**: compare predictions for known-good queries before
   and after.
6. **Check sampling**: did the sampler change? A new max-degree cutoff
   or different importance weighting can shift performance.

---

## Tier 5 — Follow-up & Probing Questions

### Q5.1 In what specific cases is the 1-WL bound tight, and where does it
hurt in practice?

**Answer.** The 1-WL bound is tight in the sense that *some* MPNNs (GIN
with sum + MLP) match it. In practice, the bound *hurts* whenever a task
requires distinguishing graphs that 1-WL conflates. Examples:

- **Counting fixed-length cycles** (girth, presence of triangles): 1-WL
  cannot detect cycle counts in general regular graphs.
- **Distinguishing regular graphs**: any two *r*-regular graphs are
  1-WL-equivalent and so indistinguishable to MPNNs.
- **Graph property prediction in chemistry** sometimes requires counting
  specific substructures (rings of size 6 vs 5, e.g., in naphthalene
  isomers); MPNNs alone can be insufficient.

Solutions include higher-order GNNs (k-WL: k-GNNs, FWL-GNNs), subgraph
GNNs (ID-GNN, NGNN), or explicit substructure features (Graph
Substructure Networks).

### Q5.2 What is over-squashing? How does it differ from over-smoothing?

**Answer.**

- **Over-smoothing**: as depth grows, all node embeddings become similar
  (a *node-similarity* problem; the model loses discriminative power).
- **Over-squashing** (Alon & Yahav, 2021): even at moderate depth, when
  a target node needs information from many far-away nodes, all that
  information must pass through a small number of intermediate nodes —
  forming a "bottleneck" — and is lost or compressed beyond use.

The two phenomena coexist but are different. Over-smoothing is about the
*output spectrum* (everything converges); over-squashing is about
*information bottlenecks* on the path.

Mitigations for over-squashing: graph rewiring (DropEdge, GDC,
Cayley graphs from "Understanding Over-squashing and Bottlenecks
on Graphs via Curvature"), virtual nodes (a single node connected to
all), Transformer-on-graph hybrids.

### Q5.3 What's the difference between a graph Transformer and a GNN with
attention?

**Answer.** A **graph Transformer** in its purest form treats the graph
as a fully connected set: every node attends to every other node, with
the graph structure entering only as a positional / structural encoding
(e.g., shortest-path distance between nodes added to the attention
score). This gives global receptive field but loses locality bias, and
is *N*² in nodes — only feasible for small graphs.

A **GNN with attention** (e.g., GAT) restricts attention to actual
graph neighbors and so is *O*(|*E*|). It keeps the locality bias and
scales to large graphs but cannot reach distant nodes in one layer.

Hybrids (Graphormer, GraphGPS, SAN) combine the two: local
message-passing layers + global attention with structural encodings,
giving the best of both worlds at the cost of extra compute.

### Q5.4 If I doubled the number of attention heads in GAT, what would you
expect?

**Answer.**

- *Capacity* increases linearly (2x parameters in the projection /
  attention layers).
- *Expressivity* increases — different heads can attend to different
  patterns (e.g., one head focuses on short-range, another on hub
  neighbors).
- *Compute* doubles per layer.
- *Empirical performance*: typically diminishing returns past 4–8 heads
  on standard benchmarks. Diminishing returns may be due to redundancy
  between heads (multiple heads converge to similar attention
  patterns).

In practice, you'd profile compute, validate carefully, and probably
land at 4–8 heads in intermediate layers and 1 averaged head in the
final layer (the GAT paper's recipe).

### Q5.5 Suppose you have a directed graph (e.g., a citation network where
direction is "cited by"). How do GNNs handle this?

**Answer.** Several options, in increasing sophistication:

1. **Symmetrize**: replace *A* with *A* + *Aᵀ*. Throws away direction
   but lets standard GCN/GAT/GraphSAGE work. Often a strong baseline.
2. **Two parallel GNNs**: one with *A*, one with *Aᵀ*; concatenate
   outputs. Captures directionality at 2x compute.
3. **Directed-aware operators**: DGCN uses both first- and second-order
   proximity; MagNet uses a complex Hermitian Laplacian to encode
   direction in phase.
4. **Edge-typed message passing**: treat (*u*, *v*) and (*v*, *u*) as
   different edge types in a heterogeneous GNN (R-GCN, CompGCN).

Choice depends on whether direction is informative for the task. In
citation networks, symmetrizing often loses little because two papers
that cite each other are mutually relevant.

### Q5.6 How would you diagnose whether your GNN is using the graph
structure at all?

**Answer.**

1. **Ablation: replace adjacency with identity**. Train the same model
   but with *A* = 0 (no message passing), so nodes only see themselves.
   This is essentially an MLP on features. Compare performance.
2. **Ablation: replace adjacency with a random graph** of similar
   density. Performance should drop substantially if the real
   structure mattered.
3. **Look at gradient flow**: backprop a class-specific loss into the
   adjacency entries (treat edges as continuous weights) and see which
   edges the model considers important.
4. **Run GNNExplainer or similar**: ask "which edges are necessary for
   this prediction?" — sparse explanations indicate structure-driven
   predictions.
5. **Aggregator collapse check**: are aggregated messages varying
   meaningfully across nodes, or have they converged?

If the model performs nearly identically with no graph or random graph,
you have not learned anything from structure — either your model is wrong
for the task, or the graph is irrelevant.

### Q5.7 Can a GNN ever be exactly Turing-equivalent to a 1-WL test, even
in principle?

**Answer.** Yes — this is essentially what GIN proves. Concretely, given
sum aggregation and a sufficiently expressive (universal) function
approximator (an MLP) for the update step, a single GIN layer can
implement an injective hash on the multiset {*v*} ∪ N(*v*) of node
features. Stacking *L* such layers makes a GNN exactly as powerful as
*L* iterations of the 1-WL color refinement — and a sum-based readout
preserves the multiset of final colors, allowing graph-level
distinguishability whenever 1-WL can distinguish.

Caveat: this is an existential result (such weights exist), not a
guarantee that gradient-based training will find them on real data. In
practice, GIN often (but not always) approaches this ceiling on
benchmarks like ZINC and OGB-MolHIV.

### Q5.8 How do you adapt GNNs to a heterophilic graph?

**Answer.** Several recent architectures specifically target heterophily:

- **H2GCN** (Zhu et al. 2020): separates ego (self) and neighbor
  embeddings, uses higher-order neighborhoods (2-hop), and uses
  intermediate-layer aggregation (jumping knowledge).
- **FAGCN** (Bo et al. 2021): learns a per-edge sign — neighbors can
  contribute positively (homophilic) or negatively (heterophilic).
- **GPR-GNN** (Chien et al. 2021): learns generalized PageRank weights
  per layer, allowing the model to choose smoothing or sharpening per
  layer.
- **MixHop**: mixes powers of the adjacency to capture multi-hop
  information at a single layer.

You can also try: (1) use both *A* and *I* − *A* in propagation — letting
the model learn whether to follow or reverse neighbor signal; (2) use
labels of neighbors as additional features (label-aware methods like
CGNN, where label propagation is interleaved with message passing).

### Q5.9 What's the role of self-loops in message passing?

**Answer.** Self-loops ensure that a node's own representation is
included in the multiset that the aggregator sees. Without them:

- The node's previous state is dropped from the aggregation, and the
  layer can use it only via the UPDATE step (which often combines
  AGGREGATE's output with the previous state via concatenation or
  addition).
- For symmetric formulations (GCN's *D*⁻¹ᐟ² *A* *D*⁻¹ᐟ²), without
  self-loops the operator pulls a node's value entirely toward its
  neighbors, leading to faster collapse.

GCN explicitly adds self-loops via *Ã* = *A* + *I*. GraphSAGE includes
the self vector via concatenation. GAT typically includes self in the
neighborhood for attention, so the node attends to itself.

### Q5.10 What's the connection between message passing and belief
propagation?

**Answer.** Message passing in GNNs is a continuous, learned analog of
**belief propagation** (BP), the algorithm Pearl developed for exact
inference on tree-structured Bayesian networks (and approximate
inference on general graphs as **loopy BP**).

In BP, each node sends its neighbors a message that summarizes its
belief about a hidden variable, marginalizing over its own state. The
neighbor incorporates incoming messages (multiplying them, in the
probabilistic case) to update its belief.

In GNNs, the messages are continuous vectors (not probability
distributions); the aggregator is sum / mean / max (not multiplication
followed by normalization); the update is a learned MLP (not a Bayesian
update). But the architecture — local computation, iteratively updated
states — is the same. This connection has motivated several lines of
research: BP-aware GNNs, energy-based GNNs, and conditional random
field-style training of GNNs.

---

## Tier 6 — Synthesis Questions (designed to expose depth of
understanding)

### Q6.1 If you could only use one GNN architecture for everything you
ever did, which would it be, and why?

**Answer.** A defensible answer: **GIN with a small MLP and sum
aggregation**, plus a sum-readout for graph-level tasks.

- It achieves the maximum expressiveness allowed by the 1-WL bound, so
  it never *underperforms* a vanilla MPNN on representational grounds.
- It's simple to implement and fast.
- It can be extended easily with edge features and made deeper with
  residual connections.
- For inductive node-level tasks it works as well as any other MPNN.
- For very large graphs, sampling-based variants (GIN with neighborhood
  sampling) work too.

Caveats one should mention: it doesn't handle attention-style
neighbor weighting natively (might lose to GAT in some node-classification
tasks), and it doesn't address heterophily out of the box.

(A different defensible answer is GAT — for the interpretability and
neighbor-weighting flexibility — at the cost of more compute.)

### Q6.2 Given a benchmark on which GCN, GAT, GraphSAGE, and GIN all
perform within 1%, what does this tell you?

**Answer.** Several possibilities:

- The benchmark is **saturated** — all reasonable models are at the
  ceiling; the structure is sufficiently informative that minor
  architectural differences don't matter.
- The benchmark is **homophilic and small**, so the simple smoothing
  bias of GCN already captures the signal.
- The benchmark is **node-feature-driven**: structure matters little, so
  any architecture's MLP component dominates and they all converge.

Either way, it's a sign you should look at *harder* benchmarks (OGB,
heterophilic suites) to differentiate architectures meaningfully — and
that for this particular benchmark, simpler/faster architectures should
be preferred.

### Q6.3 Why do molecular GNNs often outperform sequence models (SMILES
RNN/Transformer) for property prediction?

**Answer.** SMILES is a *string serialization* of a molecular graph. Two
issues:

1. **Multiple SMILES per molecule**: there is no canonical way to write
   one (different "starting atom" choices give different strings), so a
   sequence model learning representations of the string is sensitive
   to a choice that has no chemical meaning. (Canonicalization helps
   but doesn't fully solve.)
2. **Loss of structural locality**: atoms that are spatially close in
   the molecule (forming a ring) can be far apart in the SMILES string,
   making it harder for a sequence model to capture local interactions
   without many layers.

GNNs, by being permutation-invariant and operating directly on the graph
structure, sidestep both issues. They do not need to learn a notion of
chemistry from a serialization choice; they are presented with the
chemistry directly.

That said, large pretrained molecular Transformers (ChemBERTa, etc.)
catch up by training on enormous unlabeled SMILES corpora — capacity and
data overcome representation issues. For limited-data regimes, GNNs
typically win.

### Q6.4 What does the future direction of GNN research look like, in
your view?

**Answer.** A few active threads:

- **Beyond message passing**: subgraph-based methods (NGNN, ID-GNN),
  higher-order GNNs (k-GNNs), graph transformers (Graphormer, GraphGPS)
  that combine local message passing with global attention.
- **Geometric & equivariant GNNs**: E(3)-equivariant networks (EGNN,
  Equiformer, NequIP) for 3D molecules and physics simulators.
- **Foundation models on graphs**: generalist models pretrained on many
  graphs across domains (analogous to LLMs for text).
- **Causality and counterfactuals on graphs**: "what would happen if we
  added/removed this edge?"
- **Robustness and adversarial defense**: especially in fraud and
  recommendation settings.
- **Scalability**: efficient distributed training on trillion-edge
  graphs.
- **Theoretical understanding**: tighter bounds beyond 1-WL,
  characterizing what tasks are or aren't solvable by what
  architectures.

---

## Quick Reference — Cheat Sheet

| Architecture | Aggregator | Self | Inductive | Notes |
|---|---|---|---|---|
| GCN | Degree-normalized sum | Self-loop | Transductive | Spectral derivation; smoothing |
| GraphSAGE | Mean / max-pool / LSTM | Concat | Inductive | Sampling-based; scales |
| GAT | Attention-weighted sum | Attends to self | Inductive | Per-edge weighting |
| GIN | Sum + MLP | (1+ε)·self in update | Inductive | Maximally expressive (1-WL) |
| MPNN | Custom message + sum/mean/etc. | In update | Inductive | Original framework |
| ChebNet | Chebyshev polynomial of L | (n/a) | Transductive | K-localized spectral filter |

| Concept | One-line explanation |
|---|---|
| Permutation invariance | f(PX, PAPᵀ) = f(X, A) for any permutation P |
| Permutation equivariance | f(PX, PAPᵀ) = P f(X, A) |
| 1-WL bound | MPNNs are at most as discriminative as 1-WL |
| Over-smoothing | Deep stacks → all node embeddings collapse |
| Over-squashing | Long-range info compressed through bottlenecks |
| Renormalization trick | Use D̃⁻¹ᐟ² Ã D̃⁻¹ᐟ² with Ã = A + I, D̃ recomputed |
| Inductive vs transductive | Generalize to new nodes/graphs vs predict on a fixed graph |
| Homophily vs heterophily | Connected nodes share / don't share labels |
| Sum vs mean vs max | Sum: most expressive, can blow up; mean: loses cardinality; max: loses everything but extremes |

---

*End of guide. This document covers the theoretical and practical
foundations of Graph Neural Networks — the message passing framework, the
spectral derivation of GCN, the spatial methods GAT and GraphSAGE, and
their major application domains. The interview section is meant to
exercise depth across foundational, mathematical, applied, debugging, and
synthesis questions, at the level expected of a machine learning engineer
or applied scientist interview.*

---

**[← Previous Chapter: Generative Models](generative_models_guide.md) | [Table of Contents](../README.md) | [Next Chapter: Optimization Theory →](optimization_theory.md)**
