# DP vs DDP

- DataParallel is single-process, multi-thread, and only works on a single machine, while DistributedDataParallel is multi-process and works for both single- and multi- machine training.
- DataParallel is usually slower than DistributedDataParallel even on a single machine due to GIL contention across threads, per-iteration replicated model, and additional overhead introduced by scattering inputs and gathering outputs.
- DistributedDataParallel works with model parallel; DataParallel does not at this time. When DDP is combined with model parallel, each DDP process would use model parallel, and all processes collectively would use data parallel.