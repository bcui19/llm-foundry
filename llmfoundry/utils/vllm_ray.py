import ray
import torch


@ray.remote
class LLMRayActor:

    def __init__(self, *args, **kwargs):
        # pass 
        import vllm

        self.__version__ = vllm.__version__
        assert self.__version__ >= '0.4.1', 'OpenRLHF only supports vLLM >= 0.4.1'

        self.use_gpu_executor = kwargs['tensor_parallel_size'] == 1
        print ("cuda is available in actor: ", torch.cuda.is_available())
        print ("device counts is: ", torch.cuda.device_count())
        # print ("cuda visible devices in actor is: ", os.getenv('CUDA_VISIBLE_DEVICES', None))
        print('kwargs are: ', kwargs)

        # See https://github.com/vllm-project/vllm/blob/main/vllm/executor/gpu_executor.py
        if self.use_gpu_executor:

            vllm.worker.worker.Worker = WorkerWrap
        else:

            # RayGPUExecutor
            # See the patch https://github.com/vllm-project/vllm/commit/479d69fad0538f04cb22bf13e76ff91cfeb8a4e5
            # kwargs['worker_use_ray'] = True

            if vllm.__version__ > '0.4.1':
                RayWorkerWrapperPath = vllm.executor.ray_utils
            else:
                RayWorkerWrapperPath = vllm.engine.ray_utils

            if vllm.__version__ > '0.6.4.post1':
                # https://github.com/vllm-project/vllm/pull/10555
                kwargs['worker_cls'] = 'vllm_utils.WorkerWrap'
            else:
                RayWorkerWrapperPath = vllm.engine.ray_utils

                class RayWorkerWrapper(RayWorkerWrapperPath.RayWorkerWrapper):

                    def __init__(self, *args, **kwargs) -> None:
                        # kwargs["worker_module_name"] = "open_instruct.vllm_utils2"
                        kwargs['worker_module_name'] = 'vllm_utils'
                        kwargs['worker_class_name'] = 'vllm_utils.WorkerWrap'
                        super().__init__(*args, **kwargs)

                RayWorkerWrapperPath.RayWorkerWrapper = RayWorkerWrapper

        self.llm = vllm.LLM(*args, **kwargs)
        print('after vllm init')
        print('kwargs are: ', kwargs)

    def generate(self, *args, **kwargs):
        return self.llm.generate(*args, **kwargs)

    def init_process_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend,
    ):
        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.init_process_group(
                master_address,
                master_port,
                rank_offset,
                world_size,
                group_name,
                backend,
            )
        else:
            return self.llm.llm_engine.model_executor._run_workers(
                'init_process_group',
                master_address,
                master_port,
                rank_offset,
                world_size,
                group_name,
                backend,
            )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        self.stop_remote_worker_execution_loop()

        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.update_weight(
                name,
                dtype,
                shape,
                empty_cache,
            )
        else:
            return self.llm.llm_engine.model_executor._run_workers(
                'update_weight',
                name,
                dtype,
                shape,
                empty_cache,
            )

    def stop_remote_worker_execution_loop(self):
        # Fix error for using 2 communication group
        # https://github.com/vllm-project/vllm/commit/eb6d3c264d0cd8e44dec16bca7947fbe96415ce9#diff-e1ad69e38e033accddfa5480ec808c4740eb39244d1ef51cc3407e20dde8cfd4
        if self.__version__ > '0.4.2':
            self.llm.llm_engine.model_executor.stop_remote_worker_execution_loop(
            )
