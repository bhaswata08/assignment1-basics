# Fancy indexing: `W[X]` in PyTorch

## The base case: 1D index

$W$ has shape $(V, D)$ — $V$ rows, each a $D$-dimensional vector.

$X$ is a 1D LongTensor of shape $(N,)$, entries in $[0, V)$.

`W[X]` has shape $(N, D)$. Each entry of $X$ pulls out one row of $W$:

```
W[X][i] = W[X[i]]
```

## Why it's not a loop

`W[X]` is not computed with a Python for-loop. But it's mathematically the same as:

```
one_hot(X) @ W
```

Gather is used instead of the matmul because it's cheaper: gather only copies the $N$
rows you asked for ($O(N \cdot D)$ work), while the one-hot matmul also multiplies
against the $V - N$ rows you don't want, most of them zeroed out.

## Any-shape index: flatten, gather, reshape

If $X$ has more than one dimension — say shape $(B, T, S)$ — nothing about the rule
above changes. $X$ is still just a pile of scalar indices, each one still means "give
me this row of $W$." The only difference is that the indices are arranged in a grid
instead of a flat list.

So:

1. Flatten $X$ to 1D: shape $(N,)$ where $N = B \cdot T \cdot S$.
2. Gather: run the 1D case, shape $(N, D)$.
3. Reshape the result back into $X$'s original shape, with $D$ appended:
   $(B, T, S, D)$.

Flattening `X` doesn't change which integer sits at each position — it just relists
them in row-major order. Reshaping the output reverses that: the flat axis expands
back into $(B, T, S)$, and the new $D$ axis (which didn't exist before the gather)
sits on the end, untouched.

## General rule

$$\text{shape}(W[X]) = \text{shape}(X) + \text{shape}(W)[1:]$$

Concatenate $X$'s whole shape with $W$'s shape minus its first axis. That first axis
is the one being consumed by indexing — everything after it just rides along.

## The geometric picture

Think of $W$ as a stack of $V$ blocks, where each block has whatever shape is left
after dropping $W$'s first dimension.

- $W$ shape $(V, D)$ → $V$ blocks, each a $D$-vector.
- $W$ shape $(V, T, S)$ → $V$ blocks, each a $(T, S)$ matrix.
- $W$ shape $(V, A, T, S)$ → $V$ blocks, each a 3D $(A, T, S)$ tensor.

$X$ is a grid of index values, shape $(B, D')$ or any other shape. Every single entry
of $X$ pulls out one whole block from $W$ and drops it into the matching spot in the
output grid.

So if $W$ has shape $(V, T, S)$ and $X$ has shape $(B, D')$, then `W[X]` has shape
$(B, D', T, S)$: every one of $X$'s $B \times D'$ entries pulls a full $(T, S)$ block,
and those blocks sit in the same grid arrangement $X$ had.

This extends to any rank of $W$ — whatever's left after dropping $W$'s first
dimension gets appended whole onto $X$'s shape, no matter how many axes that leftover
part has.

## Applying it: `nn.Embedding.forward`

```python
x: Int[Tensor, "... seqlen"]                      # index tensor
W: Float[nn.Parameter, "vocab_size d_model"]      # embedding table
return self.W[x]
```

$W$ has shape $(V, D)$ — $V$ = vocab size, $D$ = embedding dim. Dropping $W$'s first
axis leaves just $(D,)$.

By the general rule:

$$\text{shape}(W[x]) = \text{shape}(x) + (D,)$$

If $x$ has shape `(..., seqlen)`, the output has shape `(..., seqlen, d_model)` —
matching the return type annotation exactly. Each token id in $x$ pulls out its
$D$-dimensional embedding vector, and the result keeps every batch dimension $x$ had,
with the embedding dimension tacked on at the end.

---

Here is a visual step-by-step breakdown of how `W[X]` works.

### 1. The Core Mental Model

Think of `W` as a **1D array/list of 5 pages**, where each page is a 3D block:

```
          [ Page 0 ]  -->  3D Tensor (shape 3 x 4 x 4)
          [ Page 1 ]  -->  3D Tensor (shape 3 x 4 x 4)
W  =      [ Page 2 ]  -->  3D Tensor (shape 3 x 4 x 4)
          [ Page 3 ]  -->  3D Tensor (shape 3 x 4 x 4)
          [ Page 4 ]  -->  3D Tensor (shape 3 x 4 x 4)

```

Now, your index tensor `X` is a **2D grid of numbers** (like a recipe telling PyTorch which page to copy):

$$X = \begin{bmatrix} 0 & 2 \\ 1 & 4 \end{bmatrix}$$

---

### 2. How PyTorch Builds the Result

PyTorch creates a new grid that has the **exact same 2D shape as `X**`, but replaces every number in `X`with the **entire 3D page** from`W`:

```
               +-----------------------+-----------------------+
Row 0:         |  W[0] (3D Tensor)     |  W[2] (3D Tensor)     |
               +-----------------------+-----------------------+
Row 1:         |  W[1] (3D Tensor)     |  W[4] (3D Tensor)     |
               +-----------------------+-----------------------+

```

---

### 3. Concrete Example with Smaller Numbers

Let's shrink the dimensions down so we can see actual numbers inside the tensors.

#### **Define `W` (Shape: `4, 2, 2`)**

Here `W` has 4 pages (index 0 to 3), and each page is a $2 \times 2$ matrix:

```python
# Page 0 (all 0s)
W[0] = [[0, 0],
        [0, 0]]

# Page 1 (all 1s)
W[1] = [[1, 1],
        [1, 1]]

# Page 2 (all 2s)
W[2] = [[2, 2],
        [2, 2]]

# Page 3 (all 3s)
W[3] = [[3, 3],
        [3, 3]]

```

#### **Define `X` (Shape: `2, 3`)**

`X` is a $2 \times 3$ grid of indices:

```python
X = [[3, 0, 1],
     [2, 1, 3]]

```

#### **The Output `W[X]` (Shape: `2, 3, 2, 2`)**

PyTorch builds a $2 \times 3$ arrangement of $2 \times 2$ matrices:

```
Row 0:  [  W[3]  ,  W[0]  ,  W[1]  ]
Row 1:  [  W[2]  ,  W[1]  ,  W[3]  ]

```

Plugging in the actual values from each page:

```python
result = [
  # --- Row 0 of X ---
  [
    [[3, 3], [3, 3]],  # grabbed W[3]
    [[0, 0], [0, 0]],  # grabbed W[0]
    [[1, 1], [1, 1]]   # grabbed W[1]
  ],
  # --- Row 1 of X ---
  [
    [[2, 2], [2, 2]],  # grabbed W[2]
    [[1, 1], [1, 1]],  # grabbed W[1]
    [[3, 3], [3, 3]]   # grabbed W[3]
  ]
]

```

---

### Summary Rule of Thumb

Whenever you do `Out = W[X]`:

1. The **first dimensions** of `Out` come directly from the shape of **`X`**.
2. The **remaining dimensions** of `Out` come from dimensions 1 to end of **`W`**.

$$\text{Shape}(W[X]) = (\text{Shape of } X) + (\text{Shape of } W[1:])$$
