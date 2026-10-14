# state_dict and load_state_dict

## What a state_dict is

A `state_dict` is a dict that maps a parameter name to a tensor. Nothing more.

The names are attribute paths through the module tree. PyTorch walks the modules
and builds the names. You never write them yourself.

```python
class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = Linear(4, 8)
        self.w2 = Linear(8, 4)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([FFN(), FFN()])
```

```
layers.0.w1.W                (8, 4)
layers.0.w2.W                (4, 8)
layers.1.w1.W                (8, 4)
layers.1.w2.W                (4, 8)
```

Read `layers.0.w1.W` as a path. It means attribute `layers`, element `0`,
attribute `w1`, parameter `W`. A dot separates each step.

Only parameters and buffers appear in the dict. A plain `Tensor` on
`self.something` does not appear. This is one reason to wrap a weight in
`nn.Parameter`.

A key name is the name of the attribute. Two implementations of the same layer
can use different attribute names. The keys then differ, but the tensors still
mean the same thing.

## What load_state_dict does

`load_state_dict` does the reverse. Give it a dict. It walks its own module tree.
It copies each tensor that matches a key into that parameter, in place.

Two things to know:

1. It returns a report of the keys that matched. It does not return the module.
   Do not chain it onto the constructor.
2. It is strict by default. A missing key, an extra key, or a wrong shape raises
   an error. It does not load part of the weights.

```python
m = Linear(4, 8)
m.load_state_dict({"W": weights})   # its own statement, not chained
```

## Why not assign the tensor directly

`m.W = weights` appears to work. It causes two problems.

1. It replaces the `nn.Parameter` with a plain `Tensor`. The weight then
   disappears from `.parameters()` and no longer tracks gradients.
2. Nothing checks the shape. A transposed matrix loads without an error.

`load_state_dict` prevents both problems.

## Save and load a model

```python
torch.save(model.state_dict(), path)
model.load_state_dict(torch.load(path))
```

PyTorch checkpoints work this way. Save the `state_dict`, not the model object.
The dict holds only tensors. It does not depend on your class definitions.

The optimizer has its own `state_dict`. It holds values such as the Adam moment
estimates. To resume training at the exact point where you stopped, save the
optimizer `state_dict` too.

## Load weights into a module you did not train

Use the same three steps every time.

1. Build the module with the correct dimensions.
2. Load the weights into the module with `load_state_dict`.
3. Run the module.

If the weights arrive as a bare tensor with no name, build the dict yourself. Use
the key that matches your own parameter.

## Worth running once

```python
m.state_dict()
list(m.named_parameters())
m.load_state_dict({"W": torch.zeros(4, 8)})   # wrong shape, read the error
```

Rename the attribute and watch the key change. This makes the naming rule clear.
