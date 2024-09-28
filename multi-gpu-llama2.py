import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'  # Set the master address
    os.environ['MASTER_PORT'] = '12355'      # Set a port (this needs to be free on your system)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def shard_dataset(dataset, rank, world_size):
    """Shard the dataset across GPUs"""
    total_samples = len(dataset)
    per_gpu_samples = total_samples // world_size
    start_idx = rank * per_gpu_samples
    end_idx = start_idx + per_gpu_samples if rank != world_size - 1 else total_samples
    return dataset.select(range(start_idx, end_idx))

def batch_loader(dataset, batch_size):
    """Create batches from dataset"""
    for i in range(0, len(dataset), batch_size):
        yield dataset.select(range(i, min(i + batch_size, len(dataset))))

def inference(rank, world_size, model_name, dataset, batch_size=1):
    setup(rank, world_size)

    # Load model and tokenizer
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(f'cuda:{rank}')
    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    # Shard dataset for each GPU
    dataset_shard = shard_dataset(dataset, rank, world_size)

    # Loop through the dataset shard in batches and prompt the model for each caption
    for batch in batch_loader(dataset_shard, batch_size):
        for sample in batch:
            selected_caption = sample['selected_caption']
            input_text = f"The following caption seems weird: '{selected_caption}'. Explain why it feels unusual."

            inputs = tokenizer(input_text, return_tensors="pt").to(f'cuda:{rank}')

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    outputs = model.generate(inputs['input_ids'], max_new_tokens=50)

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Rank {rank} - Caption: {selected_caption}\nGenerated Explanation: {generated_text}\n")

            # Clear memory after each iteration
            torch.cuda.empty_cache()

    cleanup()

def main(model_name, batch_size):
    world_size = torch.cuda.device_count()

    dataset_name = 'nlphuji/whoops'
    dataset = load_dataset(dataset_name)['test']
    
    # Use multiprocessing to spawn processes for each GPU
    mp.spawn(inference, args=(world_size, model_name, dataset, batch_size), nprocs=world_size, join=True)

if __name__ == "__main__":
    os.environ['NCCL_DEBUG'] = 'INFO'  # Enable NCCL debugging for more detailed logs
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker'  # Disable semaphore warnings
    model_name = "meta-llama/Llama-2-7b-hf"
    batch_size = 8  # Define your batch size (adjust based on GPU memory)
    main(model_name, batch_size)