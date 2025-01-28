# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import os
import sys

from llmfoundry.command_utils import train_from_yaml
from llmfoundry.command_utils.vllm_utils import create_vllm_engines

import multiprocessing
from concurrent.futures import ThreadPoolExecutor

def parallel_create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    enforce_eager: bool,
    pretrain: str,
    revision: str,
    seed: int,
    enable_prefix_caching: bool,
    max_model_len: int,
    use_multiprocessing: bool = True,
):
    """
    Calls create_vllm_engines in a multiprocessing or multithreading manner
    and returns the references of vllm_engines.
    """
    
    def worker(_):
        return create_vllm_engines(
            num_engines=1,  # Each worker creates one engine to distribute the workload
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
            pretrain=pretrain,
            revision=revision,
            seed=seed,
            enable_prefix_caching=enable_prefix_caching,
            max_model_len=max_model_len,
        )
    
    num_workers = num_engines
    
    if use_multiprocessing:
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(worker, range(num_workers))
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(worker, range(num_workers)))
    
    # Flatten results since each worker returns a list of engines
    vllm_engines = [engine for sublist in results for engine in sublist]
    
    return vllm_engines

if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]

    if os.getenv('NODE_RANK', None) == '0' and os.getenv('LOCAL_RANK', None) == '0':
        num_vllm_engines = 2
        tensor_parallel_size = 4
        vllm_model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        # vllm_engines = create_vllm_engines(
        #     num_engines=num_vllm_engines,
        #     tensor_parallel_size=tensor_parallel_size,
        #     enforce_eager=True,
        #     pretrain=vllm_model_name,
        #     revision=None,
        #     seed=1,
        #     enable_prefix_caching=False,
        #     max_model_len=4096,
        # )
        vllm_engines = parallel_create_vllm_engines(
            num_engines=num_vllm_engines,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=True,
            pretrain=vllm_model_name,
            revision=None,
            seed=1,
            enable_prefix_caching=False,
            max_model_len=4096,
            use_multiprocessing=True,
        )
        print ("vllm engines are: ", vllm_engines)

    train_from_yaml(yaml_path, args_list)
