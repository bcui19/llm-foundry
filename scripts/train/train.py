# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import os
import sys

from llmfoundry.command_utils import train_from_yaml
from llmfoundry.command_utils.vllm_utils import create_vllm_engines

if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]

    if os.getenv('NODE_RANK', None) == '0' and os.getenv('LOCAL_RANK', None) == '0':
        num_vllm_engines = 2
        tensor_parallel_size = 4
        vllm_model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        vllm_engines = create_vllm_engines(
            num_engines=num_vllm_engines,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=True,
            pretrain=vllm_model_name,
            revision=None,
            seed=1,
            enable_prefix_caching=False,
            max_model_len=4096,
        )
        print ("vllm engines are: ", vllm_engines)

    train_from_yaml(yaml_path, args_list)
