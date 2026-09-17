# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""
Entry point for OPD (On-Policy Distillation) with a homologous external teacher.

Usage:
    python3 -m verl.trainer.opd_main config=path/to/safety_opd_config.yaml [overrides...]
"""

import json

import ray
from omegaconf import OmegaConf

from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.opd_fsdp_workers import OPDFSDPWorker
from ..workers.reward import AutoRewardManager
from .opd_config import OPDTrainConfig
from .opd_trainer import RayOPDTrainer, ResourcePoolManager, Role


@ray.remote(num_cpus=1)
class OPDRunner:
    """Ray driver for homologous OPD training."""

    def run(self, config: OPDTrainConfig):
        print("[OPD] Starting OPD Training Runner")
        print(json.dumps(config.to_dict(), indent=2))

        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRolloutRef: ray.remote(OPDFSDPWorker),
            Role.Critic: ray.remote(OPDFSDPWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRolloutRef: global_pool_id,
            Role.Critic: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        RemoteRewardManager = ray.remote(AutoRewardManager).options(num_cpus=config.worker.reward.num_cpus)
        reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
        val_reward_cfg = config.worker.reward
        if config.worker.val_reward is not None and config.worker.val_reward.reward_function:
            val_reward_cfg = config.worker.val_reward
        val_reward_fn = RemoteRewardManager.remote(val_reward_cfg, tokenizer)

        from .data_loader import create_dataloader

        train_dataloader, val_dataloader = create_dataloader(config.data, tokenizer, processor)

        trainer = RayOPDTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(OPDTrainConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    train_config = OmegaConf.merge(default_config, cli_args)
    train_config: OPDTrainConfig = OmegaConf.to_object(train_config)
    train_config.deep_post_init()

    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
            }
        }
        ray.init(runtime_env=runtime_env)

    runner = OPDRunner.remote()
    ray.get(runner.run.remote(train_config))

    if train_config.trainer.ray_timeline is not None:
        ray.timeline(filename=train_config.trainer.ray_timeline)


if __name__ == "__main__":
    main()
