# DeepSpeed ZeRO

The advent of massive deep learning models has introduced significant challenges in terms of memory and computational resources. Microsoft's DeepSpeed library, with its Zero Redundancy Optimizer (ZeRO), has been at the forefront of addressing these challenges. This document details the evolution of ZeRO, starting from its foundational principles and progressing through its subsequent enhancements: ZeRO-Offload, ZeRO-Infinity, and ZeRO++.

### 1. The Foundation: DeepSpeed ZeRO

The initial ZeRO paper, "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models," laid the groundwork for a new paradigm in distributed training. It addressed the memory limitations of traditional data parallelism, where each GPU holds a full copy of the model's parameters, gradients, and optimizer states. ZeRO introduced a novel approach to eliminate this redundancy by partitioning these model states across the available GPUs.

#### Key Innovations of the Original ZeRO:

*   **Partitioning of Model States:** ZeRO is characterized by its three-stage approach to partitioning, which progressively reduces memory consumption:
    *   **Stage 1 (Optimizer State Partitioning):** This stage partitions the optimizer states (e.g., momentum and variance in Adam) across the data-parallel processes. Each GPU only stores and updates a fraction of the optimizer states. This results in a significant memory reduction, especially for optimizers like Adam that store multiple states for each parameter.
    *   **Stage 2 (Gradient and Optimizer State Partitioning):** Building upon Stage 1, this stage also partitions the gradients. During the backward pass, gradients are reduced to the specific GPU that owns the corresponding optimizer state partition. This further decreases the memory footprint on each GPU.
    *   **Stage 3 (Parameter, Gradient, and Optimizer State Partitioning):** The most memory-efficient stage, Stage 3 partitions the model parameters themselves. During the forward and backward passes, the necessary full parameter set for each layer is dynamically reconstructed on-the-fly via an `all-gather` collective operation, and then discarded immediately after use. This allows for training models of immense scale, as the memory required per GPU for the model parameters is divided by the number of GPUs.

*   **Dynamic Communication Schedule:** To maintain computational efficiency, ZeRO employs a dynamic communication schedule. This ensures that the necessary model states (parameters and gradients) are available on the correct devices just when they are needed for computation, and are then released to free up memory. This "just-in-time" approach minimizes the communication overhead while maximizing memory savings.

*   **Preserving Simplicity:** A significant advantage of ZeRO is its ease of use. Unlike model parallelism, which often requires significant code refactoring to split the model architecture across devices, ZeRO works with standard data-parallel training scripts. This makes it accessible to a broader range of researchers and practitioners who can scale their models without intricate code modifications.

In essence, the original ZeRO paper established a powerful and user-friendly framework for training massive models by intelligently managing memory resources across a distributed system.

### 2. ZeRO-Offload: Democratizing Billion-Scale Model Training

While the original ZeRO was highly effective for large GPU clusters, it still required a substantial number of GPUs to train billion-parameter models. "ZeRO-Offload: Democratizing Billion-Scale Model Training" introduced a groundbreaking enhancement that leverages the host CPU's memory and compute resources, making large-scale model training more accessible.

#### Core Contributions of ZeRO-Offload:

*   **Heterogeneous Memory Utilization:** The primary innovation of ZeRO-Offload is its ability to offload model states and computations to the CPU. Specifically, it partitions the model states and offloads the gradients and optimizer states to the host's main memory (DRAM). The model parameters remain on the GPU for the forward and backward passes.

*   **Optimal Offload Strategy:** The paper presents a principled analysis to determine the most efficient way to partition data and computation between the GPU and CPU. It concludes that offloading the optimizer states and gradients to the CPU, while keeping the parameters and the forward/backward passes on the GPU, provides the best balance of memory savings and computational efficiency. This strategy minimizes the communication volume between the GPU and CPU and avoids making the CPU a computational bottleneck.

*   **Optimized CPU Adam Optimizer:** To mitigate the performance impact of moving the optimizer step to the CPU, ZeRO-Offload includes a highly optimized Adam optimizer implementation. This custom optimizer leverages techniques like SIMD instructions and multi-threading to achieve a significant speedup over standard CPU-based Adam implementations.

*   **Delayed Parameter Update:** For scenarios with very small batch sizes where the CPU computation could still become a bottleneck, ZeRO-Offload introduces a "one-step delayed parameter update" schedule. This allows the CPU's parameter update to be overlapped with the GPU's forward and backward passes of the next training step, effectively hiding the CPU compute time.

ZeRO-Offload dramatically lowered the barrier to entry for training large models, enabling researchers with access to a single high-end GPU to train models with tens of billions of parameters—a task that previously required a large cluster.

### 3. ZeRO-Infinity: Breaking the GPU Memory Wall

Building on the success of ZeRO-Offload, "ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning" pushed the boundaries of model scale even further by incorporating Non-Volatile Memory express (NVMe) into the memory hierarchy. This allows for the training of models with tens of trillions of parameters.

#### Key Advancements in ZeRO-Infinity:

*   **Leveraging NVMe Storage:** ZeRO-Infinity extends the heterogeneous memory concept of ZeRO-Offload to include fast NVMe solid-state drives (SSDs). By offloading partitioned model states to NVMe, ZeRO-Infinity can accommodate models that are orders of magnitude larger than what can fit in GPU and CPU memory combined.

*   **Bandwidth-Centric Partitioning:** To effectively utilize the lower bandwidth of NVMe and CPU memory, ZeRO-Infinity introduces a novel data partitioning strategy. Instead of a single GPU being responsible for a parameter partition, the parameters are striped across all GPUs. This allows for parallel data transfer from the slower memory tiers to all GPUs simultaneously, effectively multiplying the available bandwidth.

*   **Overlap-Centric Design:** ZeRO-Infinity employs a sophisticated prefetching and overlapping mechanism to hide the latency of data movement between the different memory tiers (NVMe, CPU, and GPU). It intelligently schedules the transfer of the necessary model parameters so that they are available in GPU memory just in time for computation.

*   **Memory-Centric Tiling:** To handle extremely large individual layers that might not fit in GPU memory even after partitioning, ZeRO-Infinity introduces "memory-centric tiling." This technique breaks down large linear layers into smaller, sequential tiles that are processed one at a time, further reducing the peak memory requirement on the GPU.

ZeRO-Infinity represents a significant leap in large-scale model training, effectively breaking the "GPU memory wall" and paving the way for training models of unprecedented scale. It demonstrates that by intelligently managing a hierarchy of memory resources, the limitations of GPU memory can be overcome.

### 4. ZeRO++: Extremely Efficient Collective Communication for Giant Model Training

While ZeRO-Infinity addressed the memory capacity challenge, communication efficiency, especially in low-bandwidth environments or at very large scales with small per-GPU batch sizes, remained a bottleneck. "ZeRO++: Extremely Efficient Collective Communication for Giant Model Training" tackles this by introducing a suite of techniques to drastically reduce the communication volume.

#### Core Innovations of ZeRO++:

*   **Quantized Weight Communication (qwZ):** To reduce the communication overhead of the `all-gather` operation for model parameters, ZeRO++ employs block-based quantization. Before being communicated, the FP16 weights are quantized to INT8, halving the data volume. To maintain accuracy, the quantization is performed in blocks, with each block having its own quantization scale.

*   **Hierarchical Weight Partitioning (hpZ):** This technique trades a small amount of memory for a significant reduction in inter-node communication during the backward pass. It creates a secondary partition of the model weights within each node, allowing the `all-gather` operation for the backward pass to be performed using the high-speed intra-node interconnect (like NVLink), completely eliminating the need for slower inter-node communication for this step.

*   **Quantized Gradient Communication (qgZ):** ZeRO++ introduces a novel all-to-all based collective for gradients that uses INT4 quantization to reduce the communication volume by 75%. This approach is designed to preserve accuracy by performing the reduction on the full-precision de-quantized gradients. It also uses a hierarchical approach to minimize inter-node traffic.

Collectively, these optimizations in ZeRO++ reduce the communication volume of ZeRO by up to 4x, leading to significant throughput improvements, especially in clusters with limited network bandwidth. This makes large-scale training more efficient and feasible in a wider range of hardware environments.

In conclusion, the evolution from the original ZeRO to ZeRO++ represents a continuous and remarkable effort to push the boundaries of large-scale deep learning. Each iteration has introduced innovative solutions to overcome critical bottlenecks, from memory capacity to communication efficiency, making the training of increasingly massive models a reality for a growing community of researchers and developers.