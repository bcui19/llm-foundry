# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import os
import socket
import subprocess
import time

import ray
import torch
import torch.distributed as dist
from kubernetes import client, config
from transformers import (
    AutoModelForCausalLM,
)

from vllm_utils import create_vllm_engines, init_process_group
# from compose_rl.utils import print_debug_info
# I think there's an import here that messes w/ ray...
# import
# import transformer_engine.pytorch as te
from llmfoundry.command_utils import train_from_yaml


def get_gpu_info():
    try:
        # Run nvidia-smi command
        result = subprocess.run(['nvidia-smi'],
                                capture_output=True,
                                text=True,
                                check=True)

        # Print the output
        print(result.stdout)

        return result.stdout

    except subprocess.CalledProcessError as e:
        print(f'Error running nvidia-smi: {e}')
        return None
    except FileNotFoundError:
        print(
            'nvidia-smi not found. Please ensure NVIDIA drivers are installed correctly.',
        )
        return None


def broadcast_to_vllm(model, vllm_engines):
    # avoid OOM
    torch.cuda.empty_cache()
    count, num_params = 0, len(list(model.named_parameters()))
    refss = []
    for name, param in model.named_parameters():
        count += 1
        shape = param.shape
        refs = [
            engine.update_weight.remote(
                name,
                dtype=param.dtype,
                shape=shape,
                empty_cache=count == num_params,
            ) for engine in vllm_engines
        ]
        refss.extend(refs)
        torch.distributed.broadcast(
            param.data,
            0,
            group=model_update_group,
        )
    ray.get(refss)


if __name__ == "__main__":
    if os.getenv('NODE_RANK',
                 None) == '0' and os.getenv('LOCAL_RANK', None) == '0':
        # if os.getenv('')
        os.environ['NCCL_CUMEM_ENABLE'] = '0'
        os.environ['RAY_BACKEND_LOG_LEVEL'] = 'DEBUG'
        os.environ['RAY_DEBUG_LOGS'] = '1'
        print("NCCL CUM EM ENABLE is: ", os.getenv('NCCL_CUMEM_ENABLE', None))
        print(
            "Ray debug log level is: ",
            os.getenv('RAY_BACKEND_LOG_LEVEL', None)
        )
        print(
            "CUDA visible devices is: ",
            os.getenv('CUDA_VISIBLE_DEVICES', None)
        )

        print("in independent_ray_launcher.py")
        # print_debug_info()

        vllm_tensor_parallel_size = 2
        vllm_num_engines = 8
        vllm_sync_backend = 'nccl'
        model_name_or_path = 'allenai/Llama-3.1-Tulu-3-8B-DPO'
        model_name_or_path2 = 'allenai/Llama-3.1-Tulu-3-8B'

        vllm_engines = create_vllm_engines(
            num_engines=vllm_num_engines,
            tensor_parallel_size=vllm_tensor_parallel_size,
            enforce_eager=True,
            pretrain=model_name_or_path,
            revision=None,
            seed=1,
            enable_prefix_caching=False,
            max_model_len=4096,
        )

        print("number of vllm engines is: ", len(vllm_engines))

        master_address = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(('', 0))
            master_port = sock.getsockname()[1]

        world_size = vllm_num_engines * vllm_tensor_parallel_size + 1
        backend = vllm_sync_backend

        refs = [
            engine.init_process_group.remote(
                master_address,
                master_port,
                i * vllm_tensor_parallel_size + 1,
                world_size,
                'openrlhf',
                backend=backend,
            ) for i, engine in enumerate(vllm_engines)
        ]

        model_update_group = init_process_group(
            backend=backend,
            init_method=f'tcp://{master_address}:{master_port}',
            world_size=world_size,
            rank=0,
            group_name='openrlhf',
        )
        ray.get(refs)
        torch.set_default_device('cuda:0')
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path2,
            torch_dtype=torch.bfloat16,
        )
        model = model.to('cuda:0')
        # dist.barrier()
        get_gpu_info()
        torch.cuda.empty_cache()
        print('starting now')
        for i in range(10):
            start_time = time.time()

            broadcast_to_vllm(model, vllm_engines)
            print('at iter: ', i)
            print('took: ', time.time() - start_time, ' to broadcast')
        print('broadcasted model to vllm')
        for i in range(len(vllm_engines)):
            print('generating for vllm engine: ', i)
            print(
                ray.get(
                    vllm_engines[i].generate.remote('Hi there how are you?'),
                ),
            )

    print("Going to sleep for 3 minutes")
    time.sleep(180)
