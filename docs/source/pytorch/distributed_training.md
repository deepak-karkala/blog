# Distributed Training with PyTorch and DeepSpeed

Training large-scale deep learning models has become essential for achieving state-of-the-art results, but it presents significant challenges in terms of computational resources and memory requirements. Distributed training, which spreads the workload across multiple processing units (GPUs), is the key to overcoming these limitations. This guide provides a comprehensive overview of distributed training techniques, starting from the fundamentals in PyTorch and progressing to advanced, large-scale strategies using native PyTorch features and the DeepSpeed library.

## 1. Fundamentals of Distributed Training in PyTorch

At the core of distributed training in PyTorch is the `torch.distributed` package, which enables computations to be parallelized across processes and clusters of machines.

### 1.1. Setup and Initialization

To begin, a distributed environment must be established where multiple processes can communicate with each other. This involves two key steps: setting up the process group and running the processes.

**Process Group Initialization**
The `dist.init_process_group()` function is the entry point for setting up the distributed environment. Each process in the group is assigned a unique identifier called a **rank**, and the total number of processes is the **world size**.

Every process needs to coordinate through a master process. This is typically achieved by providing the master's IP address and a free port.

```python
import torch.distributed as dist
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
```

**Launching Processes**
PyTorch provides `torch.multiprocessing.spawn` for launching multiple processes on a single machine. For multi-machine setups, `torchrun` (PyTorch Elastic) is the recommended tool. `torchrun` automatically manages environment variables like `RANK` and `WORLD_SIZE`, simplifying the launch process.

Example launch command with `torchrun`:
```bash
torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 my_script.py
```

### 1.2. Communication Backends

PyTorch supports several communication backends, each with its own strengths:
*   **Gloo**: A highly portable backend that works on both CPU and GPU. It's included in pre-compiled PyTorch binaries and is a good default for getting started.
*   **NCCL (NVIDIA Collective Communications Library)**: Provides the best performance for collective operations on NVIDIA GPUs. It is the recommended backend for multi-GPU training.
*   **MPI (Message Passing Interface)**: A standard in high-performance computing (HPC). It's useful on large computer clusters where MPI is already installed and optimized for the specific hardware.

### 1.3. Communication Primitives

Communication is the foundation of distributed computing. `torch.distributed` provides two main types of communication patterns.

**Point-to-Point Communication**
This involves sending a tensor from one specific process to another.
*   `dist.send(tensor, dst)`: Sends a tensor to a destination rank (blocking).
*   `dist.recv(tensor, src)`: Receives a tensor from a source rank (blocking).
*   `dist.isend()` and `dist.irecv()`: Non-blocking (asynchronous) versions that return a request object, which can be waited upon using `req.wait()`.

**Collective Communication**
Collectives involve communication across all processes in a group. A group is a subset of all processes, with the default group being the entire world.
*   `dist.broadcast(tensor, src)`: Copies a tensor from the `src` rank to all other processes.
*   `dist.all_reduce(tensor, op)`: Aggregates tensors from all processes and distributes the result back to all of them. The `op` can be `SUM`, `AVG`, `MAX`, etc.
*   `dist.reduce(tensor, dst, op)`: Aggregates tensors from all processes but only stores the result on the `dst` rank.
*   `dist.all_gather(tensor_list, tensor)`: Gathers tensors from all processes into a list on every process.
*   `dist.gather(tensor, gather_list, dst)`: Gathers tensors from all processes into a list, but only on the `dst` rank.
*   `dist.scatter(tensor, scatter_list, src)`: Scatters a list of tensors from the `src` rank to all other processes.
*   `dist.barrier()`: A synchronization point that blocks processes until all of them have reached the barrier.

## 2. Data Parallelism Techniques

Data parallelism is the most common distributed training strategy. The dataset is split across multiple processes, and each process trains on its own subset of data using a complete copy of the model.

### 2.1. DistributedDataParallel (DDP)

`torch.nn.parallel.DistributedDataParallel` (DDP) is the standard and recommended module for data parallelism in PyTorch. It is generally faster than the older `torch.nn.DataParallel` because it is multi-process (avoiding Python's GIL contention) and works across multiple machines.

**How DDP Works**
1.  **Construction**: The model is replicated on each process. The constructor ensures all model replicas start with the exact same state by broadcasting the parameters from rank 0.
2.  **Forward Pass**: Each process performs a forward pass on its local model with its slice of the input data.
3.  **Backward Pass**: When `loss.backward()` is called, DDP registers autograd hooks on the model's parameters. As gradients are computed, these hooks fire and trigger an `all_reduce` operation to sum the gradients across all processes. This communication is overlapped with the gradient computation, making it highly efficient.
4.  **Optimizer Step**: After the backward pass, each model replica has the same averaged gradients. The optimizer then updates the local model parameters, and because they all started from the same state and received the same gradients, they remain synchronized.

**Basic DDP Usage**
```python
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

# In your setup function for each process 'rank'
# 1. Setup process group
setup(rank, world_size)

# 2. Create model and move it to the correct device
model = MyModel().to(rank)

# 3. Wrap the model with DDP
ddp_model = DDP(model, device_ids=[rank])

# --- Training loop ---
# loss_fn(ddp_model(inputs), labels).backward()
# optimizer.step()
```

### 2.2. Fully Sharded Data Parallel (FSDP)

While DDP is efficient, it requires each GPU to store a full copy of the model's parameters, gradients, and optimizer states. For very large models, this becomes a memory bottleneck. **Fully Sharded Data Parallel (FSDP)** solves this by sharding (partitioning) these three components across the data-parallel workers.

**How FSDP Works**
FSDP effectively decomposes DDP's `all_reduce` into `reduce_scatter` and `all_gather` operations.
1.  **Storage**: At rest, each GPU holds only a shard of the model's parameters, gradients, and optimizer states.
2.  **Forward Pass**: Before a layer is executed, FSDP performs an `all_gather` to collect the full, unsharded parameters for that specific layer onto all GPUs. After the layer's forward computation, the full parameters are immediately discarded, freeing up memory.
3.  **Backward Pass**: The process is reversed. Gradients are computed on the full parameters, but then a `reduce_scatter` operation is used to sum and shard the gradients, so each GPU only stores its corresponding portion of the final averaged gradients.
4.  **Optimizer Step**: The optimizer updates only the local shard of the parameters using the local shard of the gradients.

This "gather-compute-discard" strategy ensures that the full model parameters for only one layer exist in GPU memory at any given time, significantly reducing the peak memory footprint.

**PyTorch FSDP (FSDP2) Usage**
The modern FSDP API in PyTorch is declarative and uses the `fully_shard` function. It is recommended to apply `fully_shard` hierarchically to submodules (like Transformer blocks) as well as the root model. This allows for fine-grained control and better communication-computation overlap.

```python
from torch.distributed.fsdp import fully_shard

# Initialize the model (often on a 'meta' device to save CPU memory)
model = Transformer()

# Apply FSDP wrapping to each submodule (e.g., each transformer layer)
for layer in model.layers:
    fully_shard(layer)

# Apply FSDP to the root model
fully_shard(model)
model.to(device) # Move the sharded model to the training device

# The training loop remains the same
# optim = torch.optim.Adam(model.parameters(), ...)
# loss = model(x).sum()
# loss.backward()
# optim.step()
```

**Key FSDP Features**
*   **Mixed Precision**: FSDP provides a flexible `MixedPrecisionPolicy` to cast parameters to lower precision (e.g., `bfloat16`) for computation while keeping gradients in higher precision (`float32`) for numerical stability.
*   **CPU Offloading**: For extremely large models, FSDP can offload sharded parameters to CPU memory, gathering them to the GPU only when needed. This further reduces GPU memory usage at the cost of CPU-GPU data transfer overhead.
*   **DTensor Integration**: FSDP2 represents sharded parameters as `DTensor`s, a powerful abstraction that simplifies checkpointing and integration with other parallelism techniques.
*   **Prefetching**: FSDP automatically prefetches the parameters for the next layer while the current layer's computation is ongoing, hiding communication latency. Both implicit (default) and explicit prefetching are available for fine-tuning performance.

## 3. Beyond Data Parallelism: Advanced Techniques

For trillion-parameter models, even FSDP might not be enough. Advanced techniques combine FSDP with other forms of parallelism to scale further. PyTorch provides native APIs for these, and libraries like DeepSpeed offer a tightly integrated system.

### 3.1. Pipeline Parallelism (PP)

Pipeline Parallelism splits a model's layers into sequential stages, with each stage residing on a different device or set of devices. A mini-batch of data is broken down into even smaller **micro-batches**. These micro-batches are then fed through the pipeline, allowing different stages to work on different micro-batches concurrently. This technique is especially effective in clusters with limited network bandwidth between nodes, as communication only happens between adjacent stages and involves only activations, not entire gradients.

**PyTorch Pipeline Parallelism (`torch.distributed.pipelining`)**
This native API provides tools to automatically split a model and schedule the execution of micro-batches.
1.  **Splitting the Model**: The model can be split manually (by creating separate `nn.Module`s for each stage) or automatically using a tracer-based API that analyzes the model's forward pass.
2.  **Creating a `PipelineStage`**: Each model partition is wrapped in a `PipelineStage`, which manages the communication logic for that stage.
3.  **Using a `PipelineSchedule`**: A schedule like `ScheduleGPipe` (all forwards then all backwards) or `Schedule1F1B` (interleaved forward-backward) orchestrates the flow of micro-batches through the pipeline.

```python
# Assuming 'stage_model' is the nn.Module for the current rank's stage
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

stage = PipelineStage(stage_model, rank, num_stages, device)
schedule = ScheduleGPipe(stage, n_microbatches=4, loss_fn=my_loss_fn)

# In the training loop
if rank == 0: # First stage receives input data
    schedule.step(inputs)
elif rank == num_stages - 1: # Last stage receives targets and calculates loss
    outputs = schedule.step(targets=labels)
else: # Intermediate stages just execute
    schedule.step()
```

### 3.2. Tensor Parallelism (TP)

Tensor Parallelism is an intra-layer model parallelism technique that shards individual weight matrices within a layer (e.g., a `Linear` layer or the attention mechanism in a Transformer) across multiple devices. This is in contrast to pipeline parallelism, which splits layers between devices.

For a linear layer `Y = XA`, the weight matrix `A` can be split column-wise (`A = [A1, A2]`) or row-wise.
*   **Column-wise Parallelism**: `Y = X[A1, A2] = [XA1, XA2]`. Each device computes part of the output, and the results are concatenated.
*   **Row-wise Parallelism**: If `A` is split row-wise, the input `X` must also be split. After local matrix multiplication, an `all_reduce` is needed to sum the partial results.

In a Transformer, TP is typically applied by sharding the query, key, and value projection matrices column-wise and the output projection matrix row-wise. This strategy requires only two collective communications per Transformer block.

**PyTorch Tensor Parallel**
The native TP API uses `parallelize_module` to apply sharding strategies to a model.
*   `ColwiseParallel()` and `RowwiseParallel()` specify how `nn.Linear` layers should be sharded.
*   `SequenceParallel` is a variant that shards activations along the sequence dimension, further saving memory.

## 4. Composing Parallelism Strategies: 2D & 3D Parallelism

The true power for extreme-scale training comes from combining these techniques. A common and highly effective strategy is **2D Parallelism**:
*   **Tensor Parallelism (TP)** is used *within* a node (intra-node) to take advantage of high-speed interconnects like NVLink.
*   **Data Parallelism (FSDP)** is used *across* nodes (inter-node).

This reduces the size of the FSDP communication group, mitigating the communication bottleneck that arises when using FSDP with thousands of GPUs.

**PyTorch `DeviceMesh`**
Managing the process groups for these multi-dimensional strategies can be complex. `torch.distributed.device_mesh` is an abstraction that simplifies this. A user can define a multi-dimensional mesh of devices and then create sub-meshes for each parallelism dimension.

```python
from torch.distributed.device_mesh import init_device_mesh

# Create a 2D mesh for 64 GPUs: 8-way Data Parallel, 8-way Tensor Parallel
mesh_2d = init_device_mesh("cuda", (8, 8), mesh_dim_names=("dp", "tp"))

# Get sub-meshes for each dimension
dp_mesh = mesh_2d["dp"]
tp_mesh = mesh_2d["tp"]

# Apply TP using the tp_mesh
model = parallelize_module(model, tp_mesh, tp_plan)

# Apply FSDP using the dp_mesh
model = fully_shard(model, mesh=dp_mesh)
```

## 5. DeepSpeed: An Integrated Library for Scale

DeepSpeed is an open-source library from Microsoft that provides a suite of system optimizations for large-scale training, often packaging the above concepts into a user-friendly and highly optimized engine.

**Key DeepSpeed Technologies**
*   **ZeRO (Zero Redundancy Optimizer)**: DeepSpeed's powerful implementation of FSDP. It comes in several stages:
    *   **ZeRO-1**: Shards optimizer states.
    *   **ZeRO-2**: Shards optimizer states and gradients.
    *   **ZeRO-3**: Shards optimizer states, gradients, and model parameters (equivalent to FSDP).
*   **3D Parallelism**: DeepSpeed seamlessly integrates its ZeRO-powered data parallelism with pipeline parallelism and tensor-slicing model parallelism to train trillion-parameter models efficiently. It intelligently maps each parallelism dimension to the hardware topology to maximize communication efficiency.
*   **ZeRO-Offload**: Extends ZeRO by offloading optimizer states and gradients to CPU memory, allowing users to train multi-billion parameter models on a single GPU.
*   **Communication Compression**:
    *   **1-bit Adam / 1-bit LAMB**: These algorithms reduce communication volume by up to 5x by compressing optimizer states before communication, accelerating training on clusters with limited network bandwidth.
*   **DeepSpeed Sparse Attention**: Provides efficient kernels to support extremely long input sequences by using block-sparse attention patterns, reducing the quadratic complexity of standard attention.
*   **DeepSpeed Inference**: A suite of tools to optimize inference latency and cost for large models, including inference-adapted parallelism, optimized CUDA kernels, and support for quantization-aware training.
*   **Performance Tools**: Includes a flops profiler to identify performance bottlenecks and a compressed training feature called **Progressive Layer Dropping** to accelerate convergence.

DeepSpeed provides a powerful, all-in-one solution for researchers and engineers looking to push the boundaries of model scale and training efficiency.