# Compute notes

#TODO:
- Exp of multi node -- > multi gpu
- Exp of Single node -- > multi gpu

## How to decide the distributed inference strategy?[](https://docs.vllm.ai/en/latest/serving/distributed_serving.html#how-to-decide-the-distributed-inference-strategy "Permalink to this heading")

Before going into the details of distributed inference and serving, let’s first make it clear when to use distributed inference and what are the strategies available. The common practice is:

- **Single GPU (no distributed inference)**: If your model fits in a single GPU, you probably don’t need to use distributed inference. Just use the single GPU to run the inference.
    
- **Single-Node Multi-GPU (tensor parallel inference)**: If your model is too large to fit in a single GPU, but it can fit in a single node with multiple GPUs, you can use tensor parallelism. The tensor parallel size is the number of GPUs you want to use. For example, if you have 4 GPUs in a single node, you can set the tensor parallel size to 4.
    
- **Multi-Node Multi-GPU (tensor parallel plus pipeline parallel inference)**: If your model is too large to fit in a single node, you can use tensor parallel together with pipeline parallelism. The tensor parallel size is the number of GPUs you want to use in each node, and the pipeline parallel size is the number of nodes you want to use. For example, if you have 16 GPUs in 2 nodes (8 GPUs per node), you can set the tensor parallel size to 8 and the pipeline parallel size to 2.
    

In short, you should increase the number of GPUs and the number of nodes until you have enough GPU memory to hold the model. The tensor parallel size should be the number of GPUs in each node, and the pipeline parallel size should be the number of nodes.

After adding enough GPUs and nodes to hold the model, you can run vLLM first, which will print some logs like `# GPU blocks: 790`. Multiply the number by `16` (the block size), and you can get roughly the maximum number of tokens that can be served on the current configuration. If this number is not satisfying, e.g. you want higher throughput, you can further increase the number of GPUs or nodes, until the number of blocks is enough.

Note:
	There is one edge case: if the model fits in a single node with multiple GPUs, but the number of GPUs cannot divide the model size evenly, you can use pipeline parallelism, which splits the model along layers and supports uneven splits. In this case, the tensor parallel size should be 1 and the pipeline parallel size should be the number of GPUs.


Note that in the following we will have examples on how to use vllm on multi-node set up and all that, but to be consistant with the context, in the following we will be focused on using GRPO trainer for generation and training. 

Background:
Based on the experiments of the team, we came to conclude that if we seperate out training vs inference, it would lead to significant speed gain! 
This seperation means that for instance in single node--multi gpu setup, we will reserve one gpu for the vllm server then the rest of gpus for training. For intstance, imagine we have 8 gpus on a single Node, (0,1,2,3,4,5,6,7) so we take the first seven, 0,1,2,3,4,5,6 for training and 7ish for vllm to do the inference. we will see this in practice in below;

## Below is an example SLURM script to train a 70B model with GRPO on different setups (Single Node -- Multi GPU and Multi Node --Multi GPU).
# Running vLLM on a single node

 **1. Single Node -- Single GPU**
Pratically, if you think about it single gpu for both training and inference, it is going to be very messy! Cuz it has to allocate a fraction of same gpu memory to the vllm and a fraction for training, which will be messy eventually and very high risk of OOM for training even small model! This setup is deprecated in most of the libraries supporting vllm, including trl!
2.**Single Node -- Multi GPU**

```bash
#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --gres=gpu:5 

# Specify which GPUs to use for training and vLLM TRAIN_GPUS="0,1,2,3" VLLM_GPU="4" 

# Run training CUDA_VISIBLE_DEVICES=$TRAIN_GPUS srun accelerate launch \ --config_file examples/accelerate_configs/deepspeed_zero3.yaml \ 
--num_processes 4 \ 
--num_machines 1 \ 
--machine_rank 0 \ 
--rdzv_backend c10d \ 
train_grpo.py & 

# Run vLLM server 
CUDA_VISIBLE_DEVICES=$VLLM_GPU srun trl vllm-serve --model Qwen/Qwen2-7B & wait
```

3. Multi Node Multi GPU
4

```python
bst = bs * seq
```
## Highlevel overview

the default nowadays is to use "mixed precision" for full training--- meaning some components in full n some in half. if you do so then to calculate the memory needed for model you can come to in half precision;
m_param = 2 * N
m_grad = 2 * N


## Activation recompute
- full
- selective (in pytorch)
	- linear
	- attention 
## grad accum
