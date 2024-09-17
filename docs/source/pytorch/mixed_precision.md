# Mixed Precision

### References
- [PyTorch Doc: Automatic Mixed Precision package - torch.amp](https://pytorch.org/docs/stable/amp.html#gradient-scaling)
- [PyTorch Doc: Automatic Mixed Precision](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
- [Automatic Mixed Precision examples](https://pytorch.org/docs/stable/notes/amp_examples.html)

torch.amp provides convenience methods for mixed precision, where some operations use the torch.float32 (float) datatype and other operations use lower precision floating point datatype (lower_precision_fp): torch.float16 (half) or torch.bfloat16. 

- Mixed precision tries to match each op to its appropriate datatype.
	- Some ops, like linear layers and convolutions, are much faster in lower_precision_fp.
	- Other ops, like reductions, often require the dynamic range of float32. 

- Mixed precision primarily benefits Tensor Core-enabled architectures (Volta, Turing, Ampere). This recipe should show significant (2-3X) speedup on those architectures. On earlier architectures (Kepler, Maxwell, Pascal), you may observe a modest speedup.

- Typically, mixed precision provides the greatest speedup when the GPU is saturated. Small networks may be CPU bound, in which case mixed precision won’t improve performance. Sizes are also chosen such that linear layers’ participating dimensions are multiples of 8, to permit Tensor Core usage on Tensor Core-capable GPUs. batch_size, in_size, out_size, and num_layers are chosen to be large enough to saturate the GPU with work.


### Gradient Scaling

- If the forward pass for a particular op has float16 inputs, the backward pass for that op will produce float16 gradients. Gradient values with small magnitudes may not be representable in float16. These values will flush to zero (“underflow”), so the update for the corresponding parameters will be lost.

- To prevent underflow, “gradient scaling” multiplies the network’s loss(es) by a scale factor and invokes a backward pass on the scaled loss(es). Gradients flowing backward through the network are then scaled by the same factor. In other words, gradient values have a larger magnitude, so they don’t flush to zero.

- Each parameter’s gradient (.grad attribute) should be unscaled before the optimizer updates the parameters, so the scale factor does not interfere with the learning rate.


### CUDA Op-Specific Behavior

##### Some of the CUDA Ops that can autocast to float16
- __matmul__, addbmm, bmm, chain_matmul, multi_dot, conv1d, conv2d, conv3d,  GRUCell, linear, LSTMCell, matmul, mm, mv, prelu, RNNCell

##### Some of the CUDA Ops that can autocast to float32
- binary_cross_entropy_with_logits, cosine_similarity, cross_entropy, cumsum, dist, l1_loss, layer_norm, log, log_softmax, mse_loss, nll_loss, norm, normalize,pow, prod, soft_margin_loss, softmax, softmin, softplus, sum

##### Some of the CUDA Ops that promote to the widest input type
- These ops don’t require a particular dtype for stability, but take multiple inputs and require that the inputs’ dtypes match. If all of the inputs are float16, the op runs in float16. If any of the inputs is float32, autocast casts all inputs to float32 and runs the op in float32.
- addcdiv, addcmul, bilinear, cross, dot, grid_sample, scatter_add, tensordot



### Code Sample

```python
import torch, time, gc

# Timing utilities
start_time = None

'''
While measuring time, CPU will send tasks to GPU, immediately reports time whereas those tasks on GPU may not yet have been picked up / completed. To avoid this, use torch.cuda.synchronised() and then measure time
'''
def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()

def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))
```

```python
def make_model(in_size, out_size, num_layers):
    layers = []
    for _ in range(num_layers - 1):
        layers.append(torch.nn.Linear(in_size, in_size))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(in_size, out_size))
    return torch.nn.Sequential(*tuple(layers)).cuda()

batch_size = 512 # Try, for example, 128, 256, 513.
in_size = 4096
out_size = 4096
num_layers = 3
num_batches = 50
epochs = 3

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)

# Creates data in default precision.
# The same data is used for both default and mixed precision trials below.
# Don't need to manually change inputs' ``dtype`` when enabling mixed precision.
data = [torch.randn(batch_size, in_size) for _ in range(num_batches)]
targets = [torch.randn(batch_size, out_size) for _ in range(num_batches)]

loss_fn = torch.nn.MSELoss().cuda()
```

##### With Default Precision

```python
net = make_model(in_size, out_size, num_layers)
opt = torch.optim.SGD(net.parameters(), lr=0.001)

start_timer()
for epoch in range(epochs):
    for input, target in zip(data, targets):
        output = net(input)
        loss = loss_fn(output, target)
        loss.backward()
        opt.step()
        opt.zero_grad() # set_to_none=True here can modestly improve performance
end_timer_and_print("Default precision:")
```

##### Adding torch.autocast
Instances of torch.autocast serve as context managers that allow regions of your script to run in mixed precision.
In these regions, CUDA ops run in a dtype chosen by autocast to improve performance while maintaining accuracy.

```python
for epoch in range(0):
    for input, target in zip(data, targets):
        # Runs the forward pass under ``autocast``.
        with torch.autocast(device_type=device, dtype=torch.float16):
            output = net(input)
            # output is float16 because linear layers ``autocast`` to float16.
            assert output.dtype is torch.float16

            loss = loss_fn(output, target)
            # loss is float32 because ``mse_loss`` layers ``autocast`` to float32.
            assert loss.dtype is torch.float32

        # Exits ``autocast`` before backward().
        # Backward passes under ``autocast`` are not recommended.
        # Backward ops run in the same ``dtype`` ``autocast`` chose for corresponding forward ops.
        loss.backward()
        opt.step()
        opt.zero_grad() # set_to_none=True here can modestly improve performance
```


##### Adding GradScaler
Gradient scaling helps prevent gradients with small magnitudes from flushing to zero (“underflowing”) when training with mixed precision.

```python
# Constructs a ``scaler`` once, at the beginning of the convergence run, using default arguments.
# The same ``GradScaler`` instance should be used for the entire convergence run.
# If you perform multiple convergence runs in the same script, each run should use
# a dedicated fresh ``GradScaler`` instance. ``GradScaler`` instances are lightweight.
scaler = torch.cuda.amp.GradScaler()

for epoch in range(0):
    for input, target in zip(data, targets):
        with torch.autocast(device_type=device, dtype=torch.float16):
            output = net(input)
            loss = loss_fn(output, target)

        # Scales loss. Calls ``backward()`` on scaled loss to create scaled gradients.
        scaler.scale(loss).backward()

        # ``scaler.step()`` first unscales the gradients of the optimizer's assigned parameters.
        # If these gradients do not contain ``inf``s or ``NaN``s, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(opt)

        # Updates the scale for next iteration.
        scaler.update()

        opt.zero_grad() # set_to_none=True here can modestly improve performance
```


##### Typical Automatic Mixed Precision Training 

The following also demonstrates enabled, an optional convenience argument to autocast and GradScaler.
```python
use_amp = True

net = make_model(in_size, out_size, num_layers)
opt = torch.optim.SGD(net.parameters(), lr=0.001)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

start_timer()
for epoch in range(epochs):
    for input, target in zip(data, targets):
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            output = net(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()

        ''' Optional Gradient clipping (on unscaled gradients)
        # Unscales the gradients of optimizer's assigned parameters in-place
        scaler.unscale_(opt)

        # Since the gradients of optimizer's assigned parameters are now unscaled, clips as usual.
        # You may use the same value for max_norm here as you would without gradient scaling.
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.1)
		'''

        scaler.step(opt)
        scaler.update()
        opt.zero_grad() # set_to_none=True here can modestly improve performance
end_timer_and_print("Mixed precision:")
```

##### Gradient accumulation with Scaled Gradients
- Gradient accumulation adds gradients over an effective batch of size batch_per_iter * iters_to_accumulate (* num_procs if distributed). 
- The scale should be calibrated for the effective batch, which means inf/NaN checking, step skipping if inf/NaN grads are found, and scale updates should occur at effective-batch granularity. Also, grads should remain scaled, and the scale factor should remain constant, while grads for a given effective batch are accumulated. If grads are unscaled (or the scale factor changes) before accumulation is complete, the next backward pass will add scaled grads to unscaled grads (or grads scaled by a different factor) after which it’s impossible to recover the accumulated unscaled grads step must apply.
- Also, only call update at the end of iterations where you called step for a full effective batch:

```python
scaler = GradScaler()

for epoch in epochs:
    for i, (input, target) in enumerate(data):
        with autocast(device_type='cuda', dtype=torch.float16):
            output = model(input)
            loss = loss_fn(output, target)
            loss = loss / iters_to_accumulate

        # Accumulates scaled gradients.
        scaler.scale(loss).backward()

        if (i + 1) % iters_to_accumulate == 0:
            # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
```

##### Gradient penalty with Scaled Gradients
- A gradient penalty implementation commonly creates gradients using torch.autograd.grad(), combines them to create the penalty value, and adds the penalty value to the loss.
- To implement a gradient penalty with gradient scaling, the outputs Tensor(s) passed to torch.autograd.grad() should be scaled. The resulting gradients will therefore be scaled, and should be unscaled before being combined to create the penalty value.
- Refer [Gradient penalty](https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-penalty) for code sample


### Troubleshooting

##### Speedup with Amp is minor
- Network may fail to saturate the GPU(s) with work, and is therefore CPU bound. Amp’s effect on GPU performance won’t matter.
	- A rough rule of thumb to saturate the GPU is to increase batch and/or network size(s) as much as you can without running OOM.
	- Try to avoid excessive CPU-GPU synchronization (.item() calls, or printing values from CUDA tensors).
	- Try to avoid sequences of many small CUDA ops (coalesce these into a few large CUDA ops if you can).

- Network may be GPU compute bound (lots of matmuls/convolutions) but your GPU does not have Tensor Cores. In this case a reduced speedup is expected.

- The matmul dimensions are not Tensor Core-friendly. Make sure matmuls participating sizes are multiples of 8.


##### Loss is inf/NaN
- Disable autocast or GradScaler individually (by passing enabled=False to their constructor) and see if infs/NaNs persist.



### References
- [PyTorch Doc: Automatic Mixed Precision package - torch.amp](https://pytorch.org/docs/stable/amp.html#gradient-scaling)
- [PyTorch Doc: Automatic Mixed Precision](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
- [Automatic Mixed Precision examples](https://pytorch.org/docs/stable/notes/amp_examples.html)














