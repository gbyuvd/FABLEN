"""
fablen.py - FABLEN: Fuzzy Adaptive Bilinear Logic Engine Networks
=================================================================
Standalone module for the FABLEN layer and stack.
Designed for tasks with boolean or rule-based feature interaction structure
(parity, digits classification, symbolic reasoning).

For sequence/language model integration, see fablen_lm.py.

Architecture overview
---------------------
Each FABLEN neuron:
  1. Maps input to [0,1] via learned affine + sigmoid (input_scale, input_shift)
  2. Selects two input features (a, b) via Sparsemax, sparse, near-discrete
  3. Picks a fuzzy Boolean operation via softmax over 16 ops
  4. Applies a learned sharpness γ to push output toward hard 0/1

The 16 ops are bilinear in (a, b):
    f(a,b) = c0 + c1*a + c2*b + c3*a*b
This means the softmax mixture collapses to a single bilinear form,
computed via one [16×4] matmul rather than 16 separate ops (OP_COEFF_TABLE).

Stacking: layers are connected via standard Pre-LN residual:
    x ← x + FABLENLayer(LayerNorm(x))

Dependencies: torch only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# SPARSEMAX
# =============================================================================

def sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Sparsemax projection onto the probability simplex.

    A drop-in alternative to softmax that produces *exact zeros* for
    low-scoring entries, giving genuinely sparse (not just peaked) distributions.
    Used for input selection so each FABLEN neuron commits to a small number
    of input features rather than blending all of them.

    Martins & Astudillo, "From Softmax to Sparsemax" (ICML 2016).

    Args:
        z:   Input tensor of any shape.
        dim: Dimension to project over (default: last).

    Returns:
        Tensor of same shape as z, non-negative, summing to 1 along `dim`,
        with exact zeros for entries below the threshold.
    """
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    z_cumsum    = torch.cumsum(z_sorted, dim=dim)

    # Build k = [1, 2, ..., D] broadcast-shaped for `dim`
    k       = torch.arange(1, z.shape[dim] + 1, dtype=z.dtype, device=z.device)
    k_shape = [1] * z.dim()
    k_shape[dim] = -1
    k = k.view(k_shape)

    # Find the support size: largest k s.t. 1 + k*z_sorted[k] > cumsum[k]
    condition = 1 + k * z_sorted > z_cumsum
    k_z = condition.sum(dim=dim, keepdim=True).clamp(min=1)

    # Threshold τ
    tau = (z_cumsum.gather(dim, k_z - 1) - 1.0) / k_z.float()

    return torch.clamp(z - tau, min=0.0)


# =============================================================================
# OPERATOR COEFFICIENT TABLE
# =============================================================================

# Every fuzzy Boolean op f(a,b) is bilinear: f = c0 + c1*a + c2*b + c3*a*b
# This table stores [c0, c1, c2, c3] for all 16 ops.
# The softmax mixture over ops then reduces to:
#   κ = softmax(op_logits) @ OP_COEFF_TABLE   →  shape [out_dim, 4]
#   y = κ[:,0] + κ[:,1]*a + κ[:,2]*b + κ[:,3]*a*b
# One matmul replaces 16 elementwise ops + a weighted sum.
OP_COEFF_TABLE = torch.tensor([
    # c0    c1    c2    c3     op name
    [ 0.,   0.,   0.,   0.],  # FALSE        = 0
    [ 0.,   0.,   0.,   1.],  # AND          = a·b
    [ 0.,   1.,   0.,  -1.],  # A_AND_NOT_B  = a - a·b
    [ 0.,   1.,   0.,   0.],  # A            = a
    [ 0.,   0.,   1.,  -1.],  # NOT_A_AND_B  = b - a·b
    [ 0.,   0.,   1.,   0.],  # B            = b
    [ 0.,   1.,   1.,  -2.],  # XOR          = a + b - 2a·b
    [ 0.,   1.,   1.,  -1.],  # OR           = a + b - a·b
    [ 1.,  -1.,  -1.,   1.],  # NOR          = 1 - a - b + a·b
    [ 1.,  -1.,  -1.,   2.],  # XNOR         = 1 - a - b + 2a·b
    [ 1.,   0.,  -1.,   0.],  # NOT_B        = 1 - b
    [ 1.,   0.,  -1.,   1.],  # A_OR_NOT_B   = 1 - b + a·b
    [ 1.,  -1.,   0.,   0.],  # NOT_A        = 1 - a
    [ 1.,  -1.,   0.,   1.],  # NOT_A_OR_B   = 1 - a + a·b
    [ 1.,   0.,   0.,  -1.],  # NAND         = 1 - a·b
    [ 1.,   0.,   0.,   0.],  # TRUE         = 1
], dtype=torch.float32)

# Human-readable op names, indexed to match OP_COEFF_TABLE rows.
OP_NAMES = [
    "FALSE", "AND", "A_AND_NOT_B", "A",
    "NOT_A_AND_B", "B", "XOR", "OR",
    "NOR", "XNOR", "NOT_B", "A_OR_NOT_B",
    "NOT_A", "NOT_A_OR_B", "NAND", "TRUE",
]


# =============================================================================
# FABLEN LAYER
# =============================================================================

def _init_diverse_sel(out_dim: int, in_dim: int, scale: float = 0.1) -> torch.Tensor:
    """
    Initialize selection logits so slot A and slot B start pointing at
    different inputs for every neuron.

    Strategy: for each neuron, slot A gets a small positive bump on one
    input index, slot B gets a small positive bump on a *different* index.
    The bumps are small enough that Sparsemax still starts near-uniform,
    but the asymmetry gives gradient descent a nudge toward distinct picks.

    No ongoing constraint, this is purely an initialization prior.
    """
    logits = torch.randn(out_dim, 2, in_dim) * scale

    # For each neuron, assign a preferred input to slot A and a different
    # one to slot B by cycling through input indices.
    for j in range(out_dim):
        idx_a = j % in_dim
        idx_b = (j + in_dim // 2) % in_dim   # offset by half the input dim
        logits[j, 0, idx_a] += scale * 3.0   # small nudge for slot A
        logits[j, 1, idx_b] += scale * 3.0   # different nudge for slot B

    return logits

class FABLENLayer(nn.Module):
    """
    A single FABLEN layer: D_out fuzzy logic neurons over D_in inputs.

    Each neuron independently:
      - Selects two inputs (a, b) from x via Sparsemax over learned logits
      - Computes a softmax-weighted mixture of 16 fuzzy Boolean ops
      - Applies learned sharpness γ to sharpen output toward {0, 1}

    Input values are first projected to [0,1] via a learned per-dimension
    affine transform + sigmoid. This is critical: without it, LayerNorm
    output (≈N(0,1)) maps to only (0.27, 0.73) - 46% of the [0,1] range
    the logic ops are defined over. 
    
    A single FABLEN layer: D_out fuzzy logic neurons over D_in inputs.

    Each neuron independently:
      - Selects two inputs (a, b) from x via Sparsemax over learned logits
      - Computes a softmax-weighted mixture of 16 fuzzy Boolean ops
      - Applies learned sharpness γ to sharpen output toward {0, 1}

    Input values are first projected to [0,1] via a learned per-dimension
    affine transform + sigmoid. This is critical: without it, LayerNorm
    output (≈N(0,1)) maps to only (0.27, 0.73) - 46% of the [0,1] range
    the logic ops are defined over. 
    
    Init Scale Recommendation:
    - init_scale=1.5 (Default): Expands initial range to ~64%. Provides 
      superior training stability and prevents late-stage loss spikes/collapse
      by avoiding premature sigmoid saturation.
    - init_scale=2.0: Expands initial range to ~76%. May accelerate early 
      learning but carries a higher risk of gradient vanishing and instability
      in deeper networks or complex logical tasks.
    
    Both scale and shift are learned during training.

    The bilinear decomposition (OP_COEFF_TABLE) means the forward pass
    runs as: one [out_dim×16] @ [16×4] matmul + 4 elementwise ops,
    rather than computing all 16 ops separately. Numerically identical;
    verified to max absolute error < 1e-5.

    Args:
        in_dim:     Number of input features.
        out_dim:    Number of output neurons.
        init_scale: Initial value of input_scale (affine pre-sigmoid).
                    Default 1.5 offers the best stability/coverage trade-off.
    """

    def __init__(self, in_dim: int, out_dim: int, init_scale: float = 1.5):
        super().__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim

        # Input selection logits: [out_dim, 2, in_dim]
        # Sparsemax over last dim selects sparse weights for slots A and B.
        # Initialization Strategy: Diversity Prior (Symmetry Breaking)
        # ------------------------------------------------------------------
        # We initialize slots A and B with small noise N(0, 0.1) but add a 
        # deterministic "nudge" to different input indices per neuron:
        #   - Slot A biased toward index: j % D_in
        #   - Slot B biased toward index: (j + D_in/2) % D_in
        #
        # Without this, both slots start with identical distributions.
        # Gradient descent has no pressure to separate them, often leading to
        # pathological "same-input" selections (e.g., XOR(x, x) ≈ 0) that 
        # stall learning. This prior forces initial diversity, encouraging the
        # model to explore binary interactions immediately.
        #
        # Note: The bias is small enough that Sparsemax can override it if the
        # task truly requires single-input operations (e.g., NOT_A(x, x)).
        self.sel_logits = nn.Parameter(_init_diverse_sel(out_dim, in_dim))

        # Op mixture logits: [out_dim, 16]
        # Softmax gives op probabilities. Zero-init = uniform prior over all
        # 16 ops at the start, the network hasn't committed to any op yet.
        self.op_logits = nn.Parameter(torch.zeros(out_dim, 16))

        # Sharpness: [out_dim]
        # gamma = exp(log_gamma), so log_gamma=0 → gamma=1 (identity on [0,1]).
        # As gamma grows, output is pushed toward hard {0,1} thresholding.
        self.log_gamma = nn.Parameter(torch.zeros(out_dim))

        # Affine input projection applied before sigmoid: [in_dim] each.
        # Expands the effective input range beyond the compressed (0.27, 0.73)
        # that plain sigmoid gives for unit-normal inputs.
        self.input_scale = nn.Parameter(torch.full((in_dim,), init_scale))
        self.input_shift = nn.Parameter(torch.zeros(in_dim))

        # Fixed coefficient table; not a parameter, just a lookup.
        self.register_buffer('coeff_table', OP_COEFF_TABLE)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., in_dim]  any leading batch/sequence dims, any value range.

        Returns:
            [..., out_dim]  same leading dims, values in [0, 1].
        """
        leading = x.shape[:-1]
        x_flat  = x.reshape(-1, self.in_dim)   # [N, in_dim]

        # --- Step 1: project input to [0,1] ---
        # Learned affine rescale before sigmoid so logic ops see the full range.
        x_logic = torch.sigmoid(x_flat * self.input_scale + self.input_shift)
        # x_logic: [N, in_dim], values in (0, 1)

        # --- Step 2: sparse input selection ---
        # sel_w: [out_dim, 2, in_dim]  Sparsemax over in_dim for each of 2 slots
        sel_w    = sparsemax(self.sel_logits, dim=-1)
        # selected: [N, out_dim, 2]  weighted combination of inputs for slots A, B
        selected = torch.einsum('ozi,ni->noz', sel_w, x_logic)
        a = selected[:, :, 0]   # [N, out_dim]  slot A values
        b = selected[:, :, 1]   # [N, out_dim]  slot B values

        # --- Step 3: bilinear op mixture ---
        # Collapse 16-op softmax mixture to 4 effective scalar coefficients.
        op_w = F.softmax(self.op_logits, dim=-1)   # [out_dim, 16]
        k    = op_w @ self.coeff_table             # [out_dim, 4]
        # Apply the single resulting bilinear form: c0 + c1*a + c2*b + c3*a*b
        y_raw = k[:, 0] + k[:, 1] * a + k[:, 2] * b + k[:, 3] * (a * b)
        # y_raw: [N, out_dim], values approximately in [0, 1]

        # --- Step 4: per-neuron sharpness ---
        # Pulls outputs toward hard {0,1}. Identity at init (gamma=1).
        gamma = torch.exp(self.log_gamma)                           # [out_dim]
        y     = torch.clamp(0.5 + gamma * (y_raw - 0.5), 0.0, 1.0)

        return y.reshape(*leading, self.out_dim)

    def extra_repr(self) -> str:
        return f'in_dim={self.in_dim}, out_dim={self.out_dim}'


# =============================================================================
# FABLEN STACK
# =============================================================================

class FABLENStack(nn.Module):
    """
    A stack of FABLENLayer blocks with Pre-LN residual connections.

    Takes real-valued inputs of any range, projects them to logic_dim via a
    learned linear bottleneck, then applies n_layers of:

        x ← x + FABLENLayer(LayerNorm(x))

    Pre-LN (LayerNorm before the logic layer, not after) is used for two
    reasons:
      1. Stability: LayerNorm output is unit-normal, giving the affine
         input projection in FABLENLayer a consistent baseline to work from.
      2. Residual geometry: the skip connection x + f(LN(x)) keeps the
         residual stream in its natural value range rather than being pulled
         toward [0,1] by the logic output.

    A learned linear head maps the final logic representation to output logits.

    Args:
        in_dim:     Input feature dimension (raw, before projection).
        logic_dim:  Width of the logic layers. Wider = more feature pairs
                    available per layer. Typical: 32–128 for structured tasks.
        out_dim:    Number of output classes / regression targets.
        n_layers:   Depth of the logic stack. Deeper stacks can compose more
                    complex boolean functions (e.g., parity needs chained XOR).
        init_scale: Passed to each FABLENLayer (affine input scale init).
    """

    def __init__(
        self,
        in_dim:     int,
        logic_dim:  int,
        out_dim:    int,
        n_layers:   int   = 4,
        init_scale: float = 1.5,
    ):
        super().__init__()

        # Linear projection from raw input space to logic space.
        # No sigmoid here; FABLENLayer handles its own input-to-[0,1] mapping.
        self.input_proj = nn.Linear(in_dim, logic_dim)

        # Logic stack with Pre-LN
        self.layers = nn.ModuleList([
            FABLENLayer(logic_dim, logic_dim, init_scale=init_scale)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(logic_dim)
            for _ in range(n_layers)
        ])

        # Linear classification/regression head.
        # Operates on the final logic representation (still real-valued after
        # residual accumulation; not clamped to [0,1] overall).
        self.head = nn.Linear(logic_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [*, in_dim]  batch of input vectors, any value range.

        Returns:
            [*, out_dim]  output logits (pre-softmax / pre-sigmoid).
        """
        # Project to logic space
        x = self.input_proj(x)   # [*, logic_dim]

        # Pre-LN residual stack
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(norm(x))

        return self.head(x)


# =============================================================================
# DIAGNOSTIC UTILITIES
# =============================================================================

@torch.no_grad()
def inspect_layer(layer: FABLENLayer, top_k: int = 5) -> dict:
    """
    Summarize the learned state of a FABLENLayer after training.

    Useful for verifying that the network discovered meaningful operations
    (e.g., XOR-heavy solutions on parity) rather than staying diffuse.

    Returns a dict with:
      - dominant_ops:    List of (op_name, mean_probability) for top_k ops,
                         averaged over all neurons. Shows which ops the layer
                         overall favored.
      - op_confidence:   Mean probability of each neuron's argmax op.
                         High → neurons have committed to specific ops.
                         Low  → neurons still mixing many ops (smooth-gate mode).
      - sel_sparsity:    Mean fraction of selection weight on the single
                         highest-weight input (avg over both slots, all neurons).
                         Near 1.0 → near-one-hot selection (committed).
                         Near 1/in_dim → fully diffuse.
      - gamma_stats:     Dict with mean, min, max of per-neuron sharpness γ.
                         High γ → outputs close to hard {0,1}.
                         γ ≈ 1  → outputs are soft/continuous.
    """
    op_probs   = F.softmax(layer.op_logits, dim=-1)     # [out_dim, 16]
    gamma      = torch.exp(layer.log_gamma)             # [out_dim]
    sel_w      = sparsemax(layer.sel_logits, dim=-1)    # [out_dim, 2, in_dim]

    # Top-k ops by mean probability across neurons
    mean_op_probs = op_probs.mean(dim=0)                # [16]
    topk_vals, topk_idx = mean_op_probs.topk(top_k)
    dominant_ops = [(OP_NAMES[i.item()], round(v.item(), 4))
                    for i, v in zip(topk_idx, topk_vals)]

    # Per-neuron op confidence = probability of its argmax op
    op_confidence = op_probs.max(dim=-1).values.mean().item()

    # Selection sparsity = mean max-weight over inputs (both slots)
    sel_sparsity = sel_w.max(dim=-1).values.mean().item()

    return {
        'dominant_ops':  dominant_ops,
        'op_confidence': round(op_confidence, 4),
        'sel_sparsity':  round(sel_sparsity, 4),
        'gamma_stats': {
            'mean': round(gamma.mean().item(), 3),
            'min':  round(gamma.min().item(),  3),
            'max':  round(gamma.max().item(),  3),
        },
    }


@torch.no_grad()
def inspect_stack(model: FABLENStack, top_k: int = 3) -> None:
    """
    Print a layer-by-layer diagnostic summary of a trained FABLENStack.

    Example output (parity task after training):
        Layer 0: dominant=[('XOR',0.41),('XNOR',0.28),...] conf=0.63 sparsity=0.99 γ_mean=8.5
        Layer 1: dominant=[('AND',0.55),('NAND',0.21),...] conf=0.71 sparsity=1.00 γ_mean=14.2

    Args:
        model:  A trained FABLENStack.
        top_k:  How many dominant ops to show per layer.
    """
    print(f"FABLENStack diagnostic ({len(model.layers)} layers, "
          f"logic_dim={model.layers[0].in_dim})")
    print("-" * 60)
    for i, layer in enumerate(model.layers):
        info = inspect_layer(layer, top_k=top_k)
        ops_str = ', '.join(f"{n}:{p}" for n, p in info['dominant_ops'])
        print(
            f"Layer {i}: "
            f"top_ops=[{ops_str}] | "
            f"conf={info['op_confidence']} | "
            f"sparsity={info['sel_sparsity']} | "
            f"γ mean={info['gamma_stats']['mean']} "
            f"max={info['gamma_stats']['max']}"
        )
    print("-" * 60)