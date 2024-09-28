# Multi-GPU LLM Recipes

This repository contains recipes for running inference and training on Large Language Models (LLMs) using PyTorch's multi-GPU support. It demonstrates how to set up parallelism using `torch.distributed` and optimizes inference pipelines for large models across multiple GPUs.

## Table of Contents
- [Introduction](#introduction)
- [Fundamentals of Multi-GPU Code](#fundamentals-of-multi-gpu-code)
- [How Multi-GPU Parallelism Works in PyTorch](#how-multi-gpu-parallelism-works-in-pytorch)
- [Backend Initialization and Process Communication](#backend-initialization-and-process-communication)
- [Key PyTorch Functions for Distributed Processing](#key-pytorch-functions-for-distributed-processing)
- [Setting up Multi-GPU Distributed Training or Inference](#setting-up-multi-gpu-distributed-training-or-inference)
- [How to Run](#how-to-run)
- [References](#references)

## Introduction

When working with large models, such as LLMs, it often becomes necessary to leverage multiple GPUs to distribute the memory and computation load. PyTorch provides a powerful distributed API to facilitate multi-GPU operations, making it easier to parallelize training or inference across GPUs or even across multiple machines.

This repository demonstrates setting up an inference pipeline with multiple GPUs for running LLMs using distributed processing.

## Fundamentals of Multi-GPU Code

### Why Multi-GPU?
- **Memory Limitation**: Large models can exceed the memory limits of a single GPU.
- **Performance Optimization**: Distributing workloads across multiple GPUs can significantly speed up training or inference.
- **Scalability**: Multi-GPU setups allow for greater scalability, especially important for large datasets or complex model architectures.

### Types of Parallelism:
1. **Data Parallelism**: This strategy simultaneously processes data segments on different GPUs, speeding up computations. You can see the example of data parallelism in the `multi-gpu-data-parallel.py` script.
2. **Model Parallelism**: The model itself is split across GPUs (typically layer-wise), with each GPU responsible for a portion of the model. This is useful when the model is too large to fit on a single GPU.
3. **Pipeline Parallelism**: A combination of data and model parallelism where different parts of the model and data batches process on different GPUs concurrently. It is typically more efficient and can lead to faster training.
4. **Tensor Parallelism**: The tensors are split across devices, with operations parallelized across GPUs.

Source: [How to Use Multiple GPUs in PyTorch](https://saturncloud.io/blog/how-to-use-multiple-gpus-in-pytorch/)

### Backend Initialization
When using `torch.distributed`, it's necessary to initialize a communication backend, which sets up the distributed environment for GPUs and facilitates communication between the nodes.

### Process Group
A process group is a collection of processes that can communicate with each other. Each process is assigned a unique rank, which helps identify the GPUs or machines involved in the computation.

---

## How Multi-GPU Parallelism Works in PyTorch

### What Happens in Multi-GPU Parallelism?

1. **Rank Assignment**: Each GPU is assigned a unique rank. Rank `0` is typically the master process, and other ranks are worker processes.
2. **World Size**: The world size is the total number of GPUs across all nodes. Each process runs on a specific GPU and communicates with others to distribute the workload.
3. **Backend Setup**: The backend (e.g., `NCCL`, `GLOO`, `MPI`) is initialized to manage the communication between GPUs.
4. **Synchronization**: Operations like `all_reduce`, `scatter`, `gather`, and `broadcast` are used to ensure synchronization of data and computation across GPUs.
5. **Gradients Sharing (for training)**: Each GPU computes its own gradients, and these are averaged (or reduced) across all processes so that each GPU can update its copy of the model.

### Example of Distributed Setup in PyTorch:

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def init_process(rank, world_size):
    # Initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
    # Assign the process to the GPU with the same rank
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def run_parallel_task(rank, world_size):
    init_process(rank, world_size)
    
    # Your model or task setup goes here...
    # Distribute work across GPUs
    
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()  # Number of available GPUs
    mp.spawn(run_parallel_task, args=(world_size,), nprocs=world_size, join=True)

```

### Key PyTorch Functions for Distributed Processing
- `dist.init_process_group()`: Initializes the distributed environment.
- `dist.destroy_process_group()`: Cleans up the distributed environment.
- `dist.all_reduce()`: Sums the input tensor across all processes.
- `dist.scatter()`: Scatters the input tensor to all processes.
- `dist.gather()`: Gathers tensors from all processes.
- `dist.broadcast()`: Broadcasts the input tensor to all processes.

---
### Setting up Multi-GPU Distributed Training or Inference
It can be challenging to set up multi-GPU distributed training or inference, but PyTorch provides a robust API to simplify the process. Here are the key steps:
1 . **Initialize the Process Group**: Use `dist.init_process_group()` to set up the distributed environment.
2. **Assign GPUs to Processes**: Use `torch.cuda.set_device(rank)` to assign each process to a specific GPU.
3. **Distribute Work**: Distribute the workload across GPUs, ensuring synchronization using `dist.all_reduce()`, `dist.scatter()`, `dist.gather()`, etc.
4. **Cleanup**: Use `dist.destroy_process_group()` to clean up the distributed environment.

---

## How to Run
```bash
torchrun -n 2 python multi-gpu-data-parallel.py
```
