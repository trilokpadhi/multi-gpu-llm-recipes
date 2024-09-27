import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

def setup(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def inference(rank, world_size, model_name, input_text):
    setup(rank, world_size)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name).to(f'cuda:{rank}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer(input_text, return_tensors="pt").to(f'cuda:{rank}')

    # Perform inference
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_new_tokens=50)

    print(f"Rank {rank} output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

    cleanup()

def main(model_name, input_text):
    world_size = torch.cuda.device_count()
    mp.spawn(inference, args=(world_size, model_name, input_text), nprocs=world_size, join=True)

if __name__ == "__main__":
    model_name = "meta-llama/Llama-2-7b"  # Replace with any other LLM
    input_text = "Write for me a motivational caption"
    main(model_name, input_text)