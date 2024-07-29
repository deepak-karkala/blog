# Distributed Training

#### Learning Resources
1. Learn to read PyTorch traces - What really happens when you call .forward, .backward, and .step? -
[How to use PyTorch Profiler with W&B](https://wandb.ai/wandb/trace/reports/A-Public-Dissection-of-a-PyTorch-Training-Step--Vmlldzo5MDE3NjU)




## Single GPU
	
### Automatic Mixed Precision


### Static Graphs with Torch.Compile
<p>
	torch.compile makes PyTorch code run faster by JIT-compiling PyTorch code into optimized kernels, all while requiring minimal code changes.
</p>
<ul>
	<li>Arbitrary Python functions can be optimized by passing the callable to torch.compile. We can then call the returned optimized function in place of the original function.
		<p>
			<code>opt_func = torch.compile(func)</code>
		</p>
	</li>
	<li>
		We can decorate the function.
		<p>
			<code>@torch.compile</code>
		</p>
	</li>
	<li>
		We can  optimize torch.nn.Module instances. This compiled_model holds a reference to the model and compiles the forward function to a more optimized version
		<p>
			<code>compiled_model = torch.compile(module)</code>
		</p>
	</li>
</ul>

##### How torch.compile works ?
into three parts:
<ul>
	<li>graph acquisition</li>
	<li>graph lowering</li>
	<li>graph compilation</li>
</ul>

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-6 mb-6">
        <img src="../_static/dl_performance/distributed_training/pytorch-compile.jpg"></img>
      </div>
    </div>
</div>
Image Source:<a href="https://pytorch.org/get-started/pytorch-2.0/">PYTORCH 2.X: FASTER, MORE PYTHONIC AND AS DYNAMIC AS EVER</a>

torch.compile takes a lot longer to complete compared to eager. This is because torch.compile compiles the model into optimized kernels as it executes. If the structure of the model doesn’t change, and so recompilation is not needed. If we run the optimized model several more times, there will be a significant improvement compared to eager.

Speedup mainly comes from reducing Python overhead and GPU read/writes, and so the observed speedup may vary on factors such as model architecture and batch size. For example, if a model’s architecture is simple and the amount of data is large, then the bottleneck would be GPU compute and the observed speedup may be less significant.

The "reduce-overhead" mode uses CUDA graphs to further reduce the overhead of Python. The second time model is run with torch.compile, is significantly slower than the other runs, although it is much faster than the first run. This is because the "reduce-overhead" mode runs a few warm-up iterations for CUDA graphs.

##### TorchDynamo and FX Graphs

TorchDynamo is responsible for JIT compiling arbitrary Python code into FX graphs, which can then be further optimized. TorchDynamo extracts FX graphs by analyzing Python bytecode during runtime and detecting calls to PyTorch operations.

TorchInductor, another component of torch.compile, further compiles the FX graphs into optimized kernels, but TorchDynamo allows for different backends to be used.

When TorchDynamo encounters unsupported Python features, such as data-dependent control flow, it breaks the computation graph, lets the default Python interpreter handle the unsupported code, then resumes capturing the graph. This highlights a major difference between TorchDynamo and previous PyTorch compiler solutions. When encountering unsupported Python features, previous solutions either raise an error or silently fail. TorchDynamo, on the other hand, will break the computation graph.


## Multiple GPUs

### Distributed Data Parallel

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-6 mb-6">
        <img src="../_static/dl_performance/distributed_training/multi-gpu.png"></img>
      </div>
    </div>
</div>
Image Source:<a href="https://sebastianraschka.com/blog/2023/pytorch-faster.html"> Some Techniques To Make Your PyTorch Models Train (Much) Faster</a>

### DeepSpeed



### Glossary

##### NCCL 
NCCL (pronounced "Nickel") is a stand-alone library of standard communication routines for GPUs, implementing all-reduce, all-gather, reduce, broadcast, reduce-scatter, as well as any send/receive based communication pattern. It has been optimized to achieve high bandwidth on platforms using PCIe, NVLink, NVswitch, as well as networking using InfiniBand Verbs or TCP/IP sockets. NCCL supports an arbitrary number of GPUs installed in a single node or across multiple nodes, and can be used in either single- or multi-process (e.g., MPI) applications.

##### Gloo
Gloo is a collective communications library. It comes with a number of collective algorithms useful for machine learning applications. These include a barrier, broadcast, and allreduce.

##### MPI
MPI (Message Passing Interface) is a standardized and portable API for communicating data via messages (both point-to-point & collective) between distributed processes. MPI is frequently used in HPC to build applications that can scale on multi-node computer clusters. In most MPI implementations, library routines are directly callable from C, C++, and Fortran, as well as other languages able to interface with such libraries.



