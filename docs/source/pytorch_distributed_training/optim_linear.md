# Optimising Linear Layers

## Checklist
- Choose the batch size and the number of inputs and outputs to be divisible by 4 (TF32) / 8 (FP16) / 16 (INT8) to run efficiently on Tensor Cores.
	- For best efficiency on A100, choose these parameters to be divisible by 32 (TF32) / 64 (FP16) / 128 (INT8)
- Batch size:
	- Larger values for batch size and the number of inputs and outputs improve parallelization and efficiency
	- As a rough guideline, choose batch sizes and neuron counts greater than 128 to avoid being limited by memory bandwidth