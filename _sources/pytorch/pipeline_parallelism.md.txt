# Pipeline Parallelism

### References
- [PyTorch Doc: Pipeline Parallelism](https://pytorch.org/docs/main/distributed.pipelining.html)
- [PyTorch Doc: Introduction to Distributed Pipeline Parallelism](https://pytorch.org/tutorials/intermediate/pipelining_tutorial.html)


### Why Pipeline Parallel?
- Pipeline Parallelism is one of the primitive parallelism for deep learning. It allows the execution of a model to be partitioned such that multiple micro-batches can execute different parts of the model code concurrently. Pipeline parallelism can be an effective technique for:
	- large-scale training
	- bandwidth-limited clusters
	- large model inference


### What is torch.distributed.pipelining?
- While promising for scaling, pipelining is often difficult to implement because it needs to partition the execution of a model in addition to model weights. The partitioning of execution often requires intrusive code changes to your model. Another aspect of complexity comes from scheduling micro-batches in a distributed environment, with data flow dependency considered.

- The pipelining package provides a toolkit that does said things automatically which allows easy implementation of pipeline parallelism on general models.

- It consists of two parts: a splitting frontend and a distributed runtime.
	- The splitting frontend takes your model code as-is, splits it up into “model partitions”, and captures the data-flow relationship.
	- The distributed runtime executes the pipeline stages on different devices in parallel, handling things like micro-batch splitting, scheduling, communication, and gradient propagation, etc.

- Overall, the pipelining package provides the following features:
	- Splitting of model code based on simple specification.

	- Rich support for pipeline schedules, including GPipe, 1F1B, Interleaved 1F1B and Looped BFS, and providing the infrastruture for writing customized schedules.

	- First-class support for cross-host pipeline parallelism, as this is where PP is typically used (over slower interconnects).

	- Composability with other PyTorch parallel techniques such as data parallel (DDP, FSDP) or tensor parallel.


##### Step 1: build PipelineStage
Before we can use a PipelineSchedule, we need to create PipelineStage objects that wrap the part of the model running in that stage. The PipelineStage is responsible for allocating communication buffers and creating send/recv ops to communicate with its peers. It manages intermediate buffers e.g. for the outputs of forward that have not been consumed yet, and it provides a utility for running the backwards for the stage model.

A PipelineStage needs to know the input and output shapes for the stage model, so that it can correctly allocate communication buffers. The shapes must be static, e.g. at runtime the shapes can not change from step to step.

##### Step 2: use PipelineSchedule for execution
We can now attach the PipelineStage to a pipeline schedule, and run the schedule with input data. 

```python
from torch.distributed.pipelining import ScheduleGPipe

# Create a schedule
schedule = ScheduleGPipe(stage, n_microbatches)

# Input data (whole batch)
x = torch.randn(batch_size, in_dim, device=device)

# Run the pipeline with input `x`
# `x` will be divided into microbatches automatically
if rank == 0:
    schedule.step(x)
else:
    output = schedule.step()
```

### Options for Splitting a Model

##### Option 1: splitting a model manually
- To directly construct a PipelineStage, the user is responsible for providing a single nn.Module instance that owns the relevant nn.Parameters and nn.Buffers, and defines a forward() method that executes the operations relevant for that stage. 

```python
with torch.device("meta"):
    assert num_stages == 2, "This is a simple 2-stage example"

    # we construct the entire model, then delete the parts we do not need for this stage
    # in practice, this can be done using a helper function that automatically divides up layers across stages.
    model = Transformer()

    if stage_index == 0:
        # prepare the first stage model
        del model.layers["1"]
        model.norm = None
        model.output = None

    elif stage_index == 1:
        # prepare the second stage model
        model.tok_embeddings = None
        del model.layers["0"]

    from torch.distributed.pipelining import PipelineStage
    stage = PipelineStage(
        model,
        stage_index,
        num_stages,
        device,
        input_args=example_input_microbatch,
    )
```

- The PipelineStage requires an example argument input_args representing the runtime input to the stage, which would be one microbatch worth of input data. This argument is passed through the forward method of the stage module to determine the input and output shapes required for communication.


##### Option 2: splitting a model automatically

```python
from torch.distributed.pipelining import pipeline, SplitPoint

# An example micro-batch input
x = torch.LongTensor([1, 2, 4, 5])

pipe = pipeline(
    module=mod,
    mb_args=(x,),
    split_spec={
        "layers.1": SplitPoint.BEGINNING,
    }
)
```

- The pipeline API splits your model given a split_spec, where SplitPoint.BEGINNING stands for adding a split point before execution of certain submodule in the forward function, and similarly, SplitPoint.END for split point after such. If we print(pipe), we can see:


```python
GraphModule(
  (submod_0): GraphModule(
    (emb): InterpreterModule()
    (layers): Module(
      (0): InterpreterModule(
        (lin): InterpreterModule()
      )
    )
  )
  (submod_1): GraphModule(
    (layers): Module(
      (1): InterpreterModule(
        (lin): InterpreterModule()
      )
    )
    (lm): InterpreterModule(
      (proj): InterpreterModule()
    )
  )
)

def forward(self, x):
    submod_0 = self.submod_0(x);  x = None
    submod_1 = self.submod_1(submod_0);  submod_0 = None
    return (submod_1,)
```

- The “model partitions” are represented by submodules (submod_0, submod_1), each of which is reconstructed with original model operations, weights and hierarchies. In addition, a “root-level” forward function is reconstructed to capture the data flow between those partitions. Such data flow will be replayed by the pipeline runtime later, in a distributed fashion.


### How does the pipeline API split a model?
- First, the pipeline API turns our model into a directed acyclic graph (DAG) by tracing the model. It traces the model using torch.export – a PyTorch 2 full-graph capturing tool.
- Then, it groups together the operations and parameters needed by a stage into a reconstructed submodule: submod_0, submod_1, …
- Different from conventional submodule access methods like Module.children(), the pipeline API does not only cut the module structure of your model, but also the forward function of your model.


### Implementing Your Own Schedule
- You can implement your own pipeline schedule by extending one of the following two class:
	- PipelineScheduleSingle: for schedules that assigns only one stage per rank
	- PipelineScheduleMulti: for schedules that assigns multiple stages per rank.

### Code Sample
- Refer to [Introduction to Distributed Pipeline Parallelism](https://pytorch.org/tutorials/intermediate/pipelining_tutorial.html) where a gpt-style transformer model is trained using distributed pipeline parallelism with torch.distributed.pipelining APIs.