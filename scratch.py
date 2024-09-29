# import torch
# import torch.distributed as dist
# from transformers import LlamaTokenizer, LlamaForCausalLM
# from datasets import load_dataset
# import os
# import json
# from tqdm import tqdm

# def setup_distributed():
#     """
#     Initialize the distributed environment.
#     """
#     try:
#         rank = int(os.environ['RANK'])
#         world_size = int(os.environ['WORLD_SIZE'])
#         local_rank = int(os.environ['LOCAL_RANK'])
#         dist_backend = 'nccl'

#         # Initialize the process group
#         dist.init_process_group(backend=dist_backend, rank=rank, world_size=world_size)
#         torch.cuda.set_device(local_rank)
#         device = torch.device(f"cuda:{local_rank}")
#         print(f"Rank {rank}: Distributed process group initialized.")
#         return rank, world_size, local_rank, device
#     except KeyError as e:
#         print(f"Rank {rank}: Environment variable {e} not set.")
#         raise
#     except Exception as e:
#         print(f"Rank {rank}: Failed to initialize distributed environment - {str(e)}")
#         raise

# def cleanup_distributed():
#     """
#     Clean up the distributed environment.
#     """
#     try:
#         dist.destroy_process_group()
#         print("Distributed process group destroyed.")
#     except Exception as e:
#         print(f"Error during distributed cleanup: {str(e)}")

# def shard_dataset(dataset, rank, world_size):
#     """
#     Shard the dataset for each process.
#     """
#     try:
#         total_samples = len(dataset)
#         per_gpu = total_samples // world_size
#         start = rank * per_gpu
#         # Ensure the last GPU gets any remaining samples
#         end = start + per_gpu if rank != world_size - 1 else total_samples
#         dataset_shard = dataset.select(range(start, end))
#         return dataset_shard
#     except Exception as e:
#         print(f"Rank {rank}: Failed to shard dataset - {str(e)}")
#         raise

# # def batch_loader(dataset, batch_size):
# #     """
# #     Generator to yield batches from the dataset.
# #     Each batch is a list of dictionaries representing individual samples.
# #     """
# #     try:
# #         for i in range(0, len(dataset), batch_size):
# #             batch = dataset[i : i + batch_size]
            
# #             # Each 'sample' in batch is already a dict
# #             batch_dicts = [sample for sample in batch]
            
# #             indices = list(range(i, min(i + batch_size, len(dataset))))
# #             yield batch_dicts, indices
# #     except Exception as e:
# #         print(f"Error in batch_loader: {str(e)}")
# #         raise

# def batch_loader(dataset, batch_size):
#     """
#     Generator to yield batches from the dataset.
#     Each batch is a list of dictionaries representing individual samples.
#     """
#     for i in range(0, len(dataset), batch_size):
#         # Select the batch range from the dataset
#         batch = dataset[i : i + batch_size]
        
#         # Convert the batch to a list of dictionaries directly from the dataset
#         batch_dicts = [{key: batch[key][j] for key in batch.keys()} for j in range(len(batch['selected_caption']))]

#         indices = list(range(i, min(i + batch_size, len(dataset))))
#         yield batch_dicts, indices

# def inference(rank, world_size, local_rank, device, model_name, dataset_shard, batch_size, output_dir):
#     """
#     Inference function for each process.
#     """
#     try:
#         # Load model and tokenizer
#         model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
#         model.to(device)
#         model.eval()
#         print(f"Rank {rank}: Model loaded on {device}")

#         tokenizer = LlamaTokenizer.from_pretrained(model_name)
#         print(f"Rank {rank}: Tokenizer loaded")

#         # Shard the dataset
#         shard_size = len(dataset_shard)
#         print(f"Rank {rank}: Dataset sharded, size: {shard_size}")

#         # Sanity Check: Print the first sample
#         if shard_size > 0:
#             first_sample = dataset_shard[0]
#             print(f"Rank {rank}: First sample: {first_sample}")

#         results = []

#         # Calculate the number of batches
#         batch_count = (shard_size + batch_size - 1) // batch_size

#         for batch_idx, (batch, indices) in enumerate(tqdm(batch_loader(dataset_shard, batch_size), 
#                                                             total=batch_count, 
#                                                             desc=f"Rank {rank} Processing")):
#             print(f"Rank {rank}: Processing batch {batch_idx + 1}/{batch_count}")
#             for sample, idx in zip(batch, indices):
#                 try:
#                     selected_caption = sample['selected_caption']
#                 except KeyError:
#                     print(f"Rank {rank}: 'selected_caption' not found in sample {idx}. Skipping.")
#                     continue

#                 input_text = f"The following caption seems weird: '{selected_caption}'. Explain why it feels unusual."

#                 inputs = tokenizer(input_text, return_tensors="pt").to(device)

#                 with torch.no_grad():
#                     outputs = model.generate(inputs['input_ids'], max_new_tokens=50)

#                 generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
#                 results.append({
#                     "id": idx,
#                     "input_prompt": input_text,
#                     "caption": selected_caption,
#                     "generated_response": generated_text
#                 })

#         # Save results to a JSON file specific to this rank
#         output_file = os.path.join(output_dir, f"results_rank_{rank}.json")
#         with open(output_file, 'w') as f:
#             json.dump(results, f, indent=4)
        
#         print(f"Rank {rank}: Results written to {output_file}")

#     except Exception as e:
#         print(f"Rank {rank}: Error during inference - {str(e)}")
#         raise

# def main():
#     """
#     Main function to perform distributed inference.
#     """
#     # Initialize distributed environment
#     try:
#         rank, world_size, local_rank, device = setup_distributed()
#     except Exception as e:
#         print(f"Rank {rank if 'rank' in locals() else 'Unknown'}: Initialization failed - {str(e)}")
#         return

#     # Each rank loads the dataset independently
#     dataset_name = 'nlphuji/whoops'
#     try:
#         dataset = load_dataset(dataset_name)['test']
#         print(f"Rank {rank}: Dataset loaded, size: {len(dataset)}")
#     except Exception as e:
#         print(f"Rank {rank}: Failed to load dataset - {str(e)}")
#         cleanup_distributed()
#         return

#     # Shard the dataset
#     try:
#         dataset_shard = shard_dataset(dataset, rank, world_size)
#         print(f"Rank {rank}: Dataset sharded, size: {len(dataset_shard)}")
#     except Exception as e:
#         print(f"Rank {rank}: Failed to shard dataset - {str(e)}")
#         cleanup_distributed()
#         return

#     # Define model and batch size
#     model_name = "meta-llama/Llama-2-7b-hf"
#     batch_size = 2

#     # Create output directory
#     output_dir = 'results'
#     try:
#         if rank == 0:
#             os.makedirs(output_dir, exist_ok=True)
#             print(f"Rank {rank}: Created output directory '{output_dir}'.")
#     except Exception as e:
#         print(f"Rank {rank}: Failed to create output directory - {str(e)}")
#         cleanup_distributed()
#         return

#     # Ensure all ranks wait until directory is created
#     try:
#         dist.barrier()
#         print(f"Rank {rank}: Passed the output directory barrier.")
#     except Exception as e:
#         print(f"Rank {rank}: Barrier synchronization failed - {str(e)}")
#         cleanup_distributed()
#         return

#     # Perform inference
#     try:
#         inference(rank, world_size, local_rank, device, model_name, dataset_shard, batch_size, output_dir)
#     except Exception as e:
#         print(f"Rank {rank}: Inference failed - {str(e)}")
#         cleanup_distributed()
#         return

#     # Ensure all processes have finished inference before aggregating
#     try:
#         dist.barrier()
#         print(f"Rank {rank}: Passed the inference barrier.")
#     except Exception as e:
#         print(f"Rank {rank}: Barrier synchronization failed - {str(e)}")
#         cleanup_distributed()
#         return

#     # Only rank 0 will aggregate results
#     if rank == 0:
#         combined_results = []
#         for r in range(world_size):
#             output_file = os.path.join(output_dir, f"results_rank_{r}.json")
#             if os.path.exists(output_file):
#                 try:
#                     with open(output_file, 'r') as f:
#                         rank_results = json.load(f)
#                         combined_results.extend(rank_results)
#                     print(f"Rank {rank}: Loaded results from Rank {r}.")
#                 except Exception as e:
#                     print(f"Rank {rank}: Failed to load results from Rank {r} - {str(e)}")
#             else:
#                 print(f"Rank {rank}: Result file {output_file} not found.")

#         # Write the combined results to a single JSON file
#         final_output_file = 'results_combined.json'
#         try:
#             with open(final_output_file, 'w') as f:
#                 json.dump(combined_results, f, indent=4)
#             print(f"Rank {rank}: Combined results written to {final_output_file}")
#         except Exception as e:
#             print(f"Rank {rank}: Failed to write combined results - {str(e)}")

#     # Cleanup distributed environment after all operations
#     cleanup_distributed()

# if __name__ == "__main__":
#     # Optional: Enable NCCL debugging for troubleshooting
#     os.environ['NCCL_DEBUG'] = 'INFO'
#     os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
#     os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker'

#     main()

# import torch
# import torch.distributed as dist
# from transformers import LlamaTokenizer, LlamaForCausalLM
# from datasets import load_dataset
# import os
# import json
# from tqdm import tqdm

# def setup_distributed():
#     """
#     Initialize the distributed environment.
#     """
#     try:
#         rank = int(os.environ['RANK'])
#         world_size = int(os.environ['WORLD_SIZE'])
#         local_rank = int(os.environ['LOCAL_RANK'])
#         dist_backend = 'nccl'

#         # Initialize the process group
#         dist.init_process_group(backend=dist_backend, rank=rank, world_size=world_size)
#         torch.cuda.set_device(local_rank)
#         device = torch.device(f"cuda:{local_rank}")
#         print(f"Rank {rank}: Distributed process group initialized.")
#         return rank, world_size, local_rank, device
#     except KeyError as e:
#         print(f"Rank {rank if 'rank' in locals() else 'Unknown'}: Environment variable {e} not set.")
#         raise
#     except Exception as e:
#         print(f"Rank {rank if 'rank' in locals() else 'Unknown'}: Failed to initialize distributed environment - {str(e)}")
#         raise

# def cleanup_distributed():
#     """
#     Clean up the distributed environment.
#     """
#     try:
#         dist.destroy_process_group()
#         print("Distributed process group destroyed.")
#     except Exception as e:
#         print(f"Error during distributed cleanup: {str(e)}")

# def shard_dataset(dataset, rank, world_size):
#     """
#     Shard the dataset for each process.
#     """
#     try:
#         total_samples = len(dataset)
#         per_gpu = total_samples // world_size
#         start = rank * per_gpu
#         # Ensure the last GPU gets any remaining samples
#         end = start + per_gpu if rank != world_size - 1 else total_samples
#         dataset_shard = dataset.select(range(start, end))
#         print(f"Rank {rank}: Sharded samples from index {start} to {end-1}. Shard size: {len(dataset_shard)}")
#         return dataset_shard
#     except Exception as e:
#         print(f"Rank {rank}: Failed to shard dataset - {str(e)}")
#         raise

# # def batch_loader(dataset, batch_size):
# #     """
# #     Generator to yield batches from the dataset.
# #     Each batch is a list of dictionaries representing individual samples.
# #     """
# #     try:
# #         for i in range(0, len(dataset), batch_size):
# #             batch = dataset[i : i + batch_size]
            
# #             # Each 'sample' in batch is already a dict
# #             batch_dicts = [sample for sample in batch]
            
# #             indices = list(range(i, min(i + batch_size, len(dataset))))
# #             yield batch_dicts, indices
# #     except Exception as e:
# #         print(f"Error in batch_loader: {str(e)}")
# #         raise

# def batch_loader(dataset, batch_size):
#     """
#     Generator to yield batches from the dataset.
#     Each batch is a list of dictionaries representing individual samples.
#     """
#     for i in range(0, len(dataset), batch_size):
#         # Select the batch range from the dataset
#         batch = dataset[i : i + batch_size]
        
#         # Convert the batch to a list of dictionaries directly from the dataset
#         batch_dicts = [{key: batch[key][j] for key in batch.keys()} for j in range(len(batch['selected_caption']))]

#         indices = list(range(i, min(i + batch_size, len(dataset))))
#         yield batch_dicts, indices

# def inference(rank, device, model_name, dataset_shard, batch_size, output_dir):
#     """
#     Inference function for each process.
#     """
#     try:
#         # Load model and tokenizer
#         model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
#         model.to(device)
#         model.eval()
#         print(f"Rank {rank}: Model loaded on {device}")

#         tokenizer = LlamaTokenizer.from_pretrained(model_name)
#         print(f"Rank {rank}: Tokenizer loaded")

#         # Shard the dataset
#         shard_size = len(dataset_shard)
#         print(f"Rank {rank}: Dataset sharded, size: {shard_size}")

#         # Sanity Check: Print the first sample
#         if shard_size > 0:
#             first_sample = dataset_shard[0]
#             print(f"Rank {rank}: First sample keys: {first_sample.keys()}")
#             print(f"Rank {rank}: First sample: {first_sample}")

#         results = []

#         # Calculate the number of batches
#         batch_count = (shard_size + batch_size - 1) // batch_size

#         for batch_idx, (batch, indices) in enumerate(tqdm(batch_loader(dataset_shard, batch_size), 
#                                                             total=batch_count, 
#                                                             desc=f"Rank {rank} Processing")):
#             print(f"Rank {rank}: Processing batch {batch_idx + 1}/{batch_count}")
#             for sample, idx in zip(batch, indices):
#                 try:
#                     selected_caption = sample['selected_caption']
#                 except KeyError:
#                     print(f"Rank {rank}: 'selected_caption' not found in sample {idx}. Skipping.")
#                     continue

#                 input_text = f"The following caption seems weird: '{selected_caption}'. Explain why it feels unusual."

#                 inputs = tokenizer(input_text, return_tensors="pt").to(device)

#                 with torch.no_grad():
#                     outputs = model.generate(inputs['input_ids'], max_new_tokens=50)

#                 generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
#                 results.append({
#                     "id": idx,
#                     "input_prompt": input_text,
#                     "caption": selected_caption,
#                     "generated_response": generated_text
#                 })

#         # Save results to a JSON file specific to this rank
#         output_file = os.path.join(output_dir, f"results_rank_{rank}.json")
#         with open(output_file, 'w') as f:
#             json.dump(results, f, indent=4)
        
#         print(f"Rank {rank}: Results written to {output_file}. Number of entries: {len(results)}")

#     except Exception as e:
#         print(f"Rank {rank}: Error during inference - {str(e)}")
#         raise

# def main():
#     """
#     Main function to perform distributed inference.
#     """
#     # Initialize distributed environment
#     try:
#         rank, world_size, local_rank, device = setup_distributed()
#     except Exception as e:
#         print(f"Rank {rank if 'rank' in locals() else 'Unknown'}: Initialization failed - {str(e)}")
#         return

#     # Each rank loads the dataset independently
#     dataset_name = 'nlphuji/whoops'
#     try:
#         dataset = load_dataset(dataset_name)['test']
#         print(f"Rank {rank}: Dataset loaded, size: {len(dataset)}")
#     except Exception as e:
#         print(f"Rank {rank}: Failed to load dataset - {str(e)}")
#         cleanup_distributed()
#         return

#     # Shard the dataset
#     try:
#         dataset_shard = shard_dataset(dataset, rank, world_size)
#         print(f"Rank {rank}: Dataset sharded, size: {len(dataset_shard)}")
#     except Exception as e:
#         print(f"Rank {rank}: Failed to shard dataset - {str(e)}")
#         cleanup_distributed()
#         return

#     # Define model and batch size
#     model_name = "meta-llama/Llama-2-7b-hf"
#     batch_size = 2

#     # Create output directory
#     output_dir = 'results'
#     try:
#         if rank == 0:
#             os.makedirs(output_dir, exist_ok=True)
#             print(f"Rank {rank}: Created output directory '{output_dir}'.")
#     except Exception as e:
#         print(f"Rank {rank}: Failed to create output directory - {str(e)}")
#         cleanup_distributed()
#         return

#     # Ensure all ranks wait until directory is created
#     try:
#         dist.barrier()
#         print(f"Rank {rank}: Passed the output directory barrier.")
#     except Exception as e:
#         print(f"Rank {rank}: Barrier synchronization failed - {str(e)}")
#         cleanup_distributed()
#         return

#     # Perform inference
#     try:
#         inference(rank, device, model_name, dataset_shard, batch_size, output_dir)
#     except Exception as e:
#         print(f"Rank {rank}: Inference failed - {str(e)}")
#         cleanup_distributed()
#         return

#     # Ensure all processes have finished inference before aggregating
#     try:
#         dist.barrier()
#         print(f"Rank {rank}: Passed the inference barrier.")
#     except Exception as e:
#         print(f"Rank {rank}: Barrier synchronization failed - {str(e)}")
#         cleanup_distributed()
#         return

#     # Only rank 0 will aggregate results
#     if rank == 0:
#         combined_results = []
#         for r in range(world_size):
#             output_file = os.path.join(output_dir, f"results_rank_{r}.json")
#             if os.path.exists(output_file):
#                 try:
#                     with open(output_file, 'r') as f:
#                         rank_results = json.load(f)
#                         combined_results.extend(rank_results)
#                     print(f"Rank {rank}: Loaded results from Rank {r}. Number of entries: {len(rank_results)}")
#                 except Exception as e:
#                     print(f"Rank {rank}: Failed to load results from Rank {r} - {str(e)}")
#             else:
#                 print(f"Rank {rank}: Result file {output_file} not found.")

#         # Write the combined results to a single JSON file
#         final_output_file = 'results_combined.json'
#         try:
#             with open(final_output_file, 'w') as f:
#                 json.dump(combined_results, f, indent=4)
#             print(f"Rank {rank}: Combined results written to {final_output_file}. Total entries: {len(combined_results)}")
#         except Exception as e:
#             print(f"Rank {rank}: Failed to write combined results - {str(e)}")

#     # Cleanup distributed environment after all operations
#     cleanup_distributed()

# if __name__ == "__main__":
#     # Optional: Enable NCCL debugging for troubleshooting
#     os.environ['NCCL_DEBUG'] = 'INFO'
#     os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
#     os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker'

#     main()

