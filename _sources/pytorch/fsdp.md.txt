# FSDP

- Mar 2022: With PyTorch 1.11 we’re adding native support for Fully Sharded Data Parallel (FSDP). Its implementation heavily borrows from FairScale’s version while bringing more streamlined APIs and additional performance improvements. Realized performance in our experiments reached 84 TFLOPS per A100 GPU for GPT 1T model and 159 TFLOPS per A100 GPU for GPT 175B model on AWS cluster. 


#### How FSDP Works

- FSDP is a type of data-parallel training, but unlike traditional data-parallel, which maintains a per-GPU copy of a model’s parameters, gradients and optimizer states, it shards all of these states across data-parallel workers and can optionally offload the sharded model parameters to CPUs.

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-4 mb-4">
        <img src="../../_static/distributed_training_and_pytorch/pytorch/fsdp/fsdp_workflow.png"></img>
      </div>
    </div>
</div>


- <b>Usually, model layers are wrapped with FSDP in a nested way, so that only layers in a single FSDP instance need to gather the full parameters to a single device during forward or backward computations. The gathered full parameters will be freed immediately after computation, and the freed memory can be used for the next layer’s computation. In this way, peak GPU memory could be saved and thus training can be scaled to use a larger model size or larger batch size. To further maximize memory efficiency, FSDP can offload the parameters, gradients and optimizer states to CPUs when the instance is not active in the computation.</b>


#### FSDP VS DDP
- In DistributedDataParallel, (DDP) training, each process/ worker owns a replica of the model and processes a batch of data, finally it uses all-reduce to sum up gradients over different workers. In DDP the model weights and optimizer states are replicated across all workers. FSDP is a type of data parallelism that shards model parameters, optimizer states and gradients across DDP ranks.

- When training with FSDP, the GPU memory footprint is smaller than when training with DDP across all workers. This makes the training of some very large models feasible by allowing larger models or batch sizes to fit on device. This comes with the cost of increased communication volume. The communication overhead is reduced by internal optimizations like overlapping communication and computation.

- At a high level FSDP works as follow:

	- In constructor
		- Shard model parameters and each rank only keeps its own shard
	- In forward path
		- Run all_gather to collect all shards from all ranks to recover the full parameter in this FSDP unit
		- Run forward computation
		- Discard parameter shards it has just collected
	- In backward path
		- Run all_gather to collect all shards from all ranks to recover the full parameter in this FSDP unit
		- Run backward computation
		- Run reduce_scatter to sync gradients
		- Discard parameters.

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-4 mb-4">
        <img src="../../_static/distributed_training_and_pytorch/pytorch/fsdp/fsdp_sharding.png"></img>
      </div>
    </div>
</div>

- <b> One way to view FSDP’s sharding is to decompose the DDP gradient all-reduce into reduce-scatter and all-gather. Specifically, during the backward pass, FSDP reduces and scatters gradients, ensuring that each rank possesses a shard of the gradients. Then it updates the corresponding shard of the parameters in the optimizer step. Finally, in the subsequent forward pass, it performs an all-gather operation to collect and combine the updated parameter shards.
</b>

#### References

- [Introducing PyTorch Fully Sharded Data Parallel (FSDP) API](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)