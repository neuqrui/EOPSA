# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from contextlib import contextmanager
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from transformers.video_utils import VideoMetadata
from vllm import LLM, RequestOutput, SamplingParams

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image, process_video
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray, list], repeats: int) -> Union[torch.Tensor, np.ndarray, list]:
    # repeat the elements, supports tensor, numpy array and list
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    elif isinstance(value, np.ndarray):
        return np.repeat(value, repeats, axis=0)
    elif isinstance(value, list):
        out = []
        for v in value:
            out.extend([v] * repeats)
        return out
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[dict[int, float]]:
    # enforce vllm to not output vision special tokens (image/video placeholders)
    if processor is None:
        return None

    logit_bias = {}
    if hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        logit_bias[image_token_id] = -100
    if hasattr(processor, "video_token"):
        video_token_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)
        logit_bias[video_token_id] = -100

    return logit_bias if logit_bias else None


def _process_multi_modal_data(
    multi_modal_data: dict[str, Any],
    image_min_pixels: int,
    image_max_pixels: int,
    video_min_pixels: int,
    video_max_pixels: int,
    video_max_frames: int,
    video_fps: float,
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    """
    Process multi-modal data and return (mm_data, mm_kwargs).
    - For images: ({"image": [processed_images]}, None)
    - For videos: ({"video": [processed_videos]}, {"do_sample_frames": False})

    Handles both:
    - "video" key: tensors already processed in dataset stage
    - "videos" key: paths that need processing (backward compatibility)
    """
    images, videos = [], []
    mm_kwargs = None

    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, image_min_pixels, image_max_pixels))

    # Check for "video" key (tensors from dataset) first
    if "video" in multi_modal_data:
        # Video tensors already processed in dataset stage
        video_tensors = multi_modal_data["video"]
        video_metadatas = multi_modal_data.get("video_metadatas", None)

        # Combine tensors with metadata for vLLM
        if video_metadatas is not None and len(video_metadatas) == len(video_tensors):
            for tensor, metadata in zip(video_tensors, video_metadatas):
                videos.append((tensor, metadata))
        else:
            # Fallback: create minimal metadata
            for tensor in video_tensors:
                num_frames = tensor.shape[0] if hasattr(tensor, 'shape') else len(tensor)
                videos.append((tensor, {"fps": 2.0, "frames_indices": list(range(num_frames)), "total_num_frames": num_frames}))

        if len(videos) > 0:
            mm_kwargs = {"do_sample_frames": False, "do_resize": False}
    elif "videos" in multi_modal_data:
        # Backward compatibility: process video paths
        for video in multi_modal_data["videos"]:
            # process_video returns ((frames, metadata), fps) when return_fps=True
            # vLLM Qwen3-VL expects each video to be (video_array, metadata) tuple
            # Use the same min/max pixels as training to ensure consistency
            processed, _ = process_video(
                video,
                min_pixels=video_min_pixels,
                max_pixels=video_max_pixels,
                max_frames=video_max_frames,
                video_fps=video_fps,
                return_fps=True
            )
            # Handle the case where processed is (frames, metadata) tuple
            if isinstance(processed, tuple) and len(processed) == 2:
                frames, metadata = processed
                # Keep the (frames, metadata) tuple for vLLM Qwen3-VL
                videos.append((frames, metadata))
            else:
                # Fallback: if no metadata, create a minimal one
                videos.append((processed, {"fps": 2.0, "frames_indices": list(range(len(processed))), "total_num_frames": len(processed)}))

        if len(videos) > 0:
            # do_sample_frames=False: we already sampled frames in fetch_video
            # do_resize=False: we already resized in fetch_video
            mm_kwargs = {"do_sample_frames": False, "do_resize": False}

    if len(images) != 0:
        return {"image": images}, None

    if len(videos) != 0:
        return {"video": videos}, mm_kwargs

    return None, None


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.tokenizer = tokenizer  # 保存 tokenizer 用于 mix-policy 离线轨迹处理
        self.pad_token_id = tokenizer.pad_token_id
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)

        # Mix-policy 配置
        self.enable_mix_policy = getattr(config, 'enable_mix_policy', False)
        if self.enable_mix_policy and self.rank == 0:
            print(f"Mix-policy enabled: offline trajectories will replace last online generation when available.")
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        engine_kwargs = {}
        if processor is not None:  # only VLMs have processor
            engine_kwargs["disable_mm_preprocessor_cache"] = True
            if config.limit_images:
                engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images, "video": 1}

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="dummy",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        default_max_tokens = (
            config.train_response_length
            if config.train_response_length > 0
            else config.response_length
        )
        sampling_kwargs = {
            "max_tokens": default_max_tokens,
            "detokenize": False,
            "logit_bias": _get_logit_bias(processor),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_raw_prompt = non_tensor_batch.pop("raw_prompt", None)  # 提取原始文本prompt
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        batch_preprocessed_video = non_tensor_batch.pop("preprocessed_video", None)  # 不需要，已在Dataset阶段使用

        # Mix-policy: 获取预采集轨迹信息
        batch_has_offline = non_tensor_batch.pop("has_offline_trajectory", None)
        batch_offline_output = non_tensor_batch.pop("offline_output", None)

        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if batch_multi_modal_data is not None:
            vllm_inputs = []
            # 准备迭代器：如果有raw_prompt则同时迭代，否则只迭代ids
            if batch_raw_prompt is not None:
                iterator = zip(batch_raw_prompt_ids, batch_raw_prompt, batch_multi_modal_data)
            else:
                iterator = zip(batch_raw_prompt_ids, [None] * len(batch_raw_prompt_ids), batch_multi_modal_data)

            for raw_prompt_ids, raw_prompt, multi_modal_data in iterator:
                if "preprocessed_video_path" in multi_modal_data:
                    # 本地加载预处理的 .pt 文件
                    pt_path = multi_modal_data["preprocessed_video_path"]
                    preprocessed_data = torch.load(pt_path, map_location="cpu", weights_only=False)
                    video_tensors = [preprocessed_data["frames"]]
                    video_metadatas_raw = [preprocessed_data["metadata"]]

                    if raw_prompt is not None:
                        item = {"prompt": raw_prompt}
                    else:
                        item = {"prompt_token_ids": list(raw_prompt_ids)}

                    # Format for vLLM: list of (tensor, VideoMetadata) tuples
                    videos = []
                    for tensor, metadata in zip(video_tensors, video_metadatas_raw):
                        if isinstance(metadata, dict):
                            metadata_obj = VideoMetadata(
                                total_num_frames=metadata.get("total_num_frames", tensor.shape[0] if hasattr(tensor, 'shape') else len(tensor)),
                                fps=metadata.get("fps"),
                                frames_indices=metadata.get("frames_indices"),
                                video_backend=metadata.get("video_backend"),
                                width=metadata.get("width"),
                                height=metadata.get("height"),
                                duration=metadata.get("duration"),
                            )
                        else:
                            metadata_obj = metadata
                        videos.append((tensor, metadata_obj))

                    item["multi_modal_data"] = {"video": videos}
                    item["mm_processor_kwargs"] = {"do_sample_frames": False, "do_resize": False}
                elif "video" in multi_modal_data:
                    # Video tensors already processed in dataset stage
                    # Pass text prompt to vLLM (required for finding <video> placeholder)
                    if raw_prompt is not None:
                        item = {"prompt": raw_prompt}  # 使用文本prompt（包含<video>占位符）
                    else:
                        item = {"prompt_token_ids": list(raw_prompt_ids)}  # 后备方案

                    video_tensors = multi_modal_data["video"]
                    video_metadatas = multi_modal_data.get("video_metadatas", None)

                    # Format for vLLM: list of (tensor, metadata) tuples for Qwen3-VL
                    # Convert dict metadata to VideoMetadata object
                    if video_metadatas is not None and len(video_metadatas) == len(video_tensors):
                        videos = []
                        for tensor, metadata in zip(video_tensors, video_metadatas):
                            # Convert dict to VideoMetadata if needed
                            if isinstance(metadata, dict):
                                metadata_obj = VideoMetadata(
                                    total_num_frames=metadata.get("total_num_frames", tensor.shape[0] if hasattr(tensor, 'shape') else len(tensor)),
                                    fps=metadata.get("fps"),
                                    frames_indices=metadata.get("frames_indices"),
                                    video_backend=metadata.get("video_backend"),
                                    width=metadata.get("width"),
                                    height=metadata.get("height"),
                                    duration=metadata.get("duration"),
                                )
                            else:
                                metadata_obj = metadata
                            videos.append((tensor, metadata_obj))
                    else:
                        # Fallback: create minimal metadata
                        videos = []
                        for tensor in video_tensors:
                            num_frames = tensor.shape[0] if hasattr(tensor, 'shape') else len(tensor)
                            metadata_obj = VideoMetadata(
                                total_num_frames=num_frames,
                                fps=2.0,
                                frames_indices=list(range(num_frames)),
                            )
                            videos.append((tensor, metadata_obj))

                    item["multi_modal_data"] = {"video": videos}
                    item["mm_processor_kwargs"] = {"do_sample_frames": False, "do_resize": False}
                else:
                    # Backward compatibility: process video paths or images
                    # 对于多模态数据（图片/视频），使用文本prompt而不是token ids
                    # 这样vLLM才能找到<image>/<video>占位符位置
                    if raw_prompt is not None and ("images" in multi_modal_data or "videos" in multi_modal_data):
                        item = {"prompt": raw_prompt}  # 使用文本prompt（包含<image>/<video>占位符）
                    else:
                        item = {"prompt_token_ids": list(raw_prompt_ids)}  # 纯文本或后备方案

                    mm_data, mm_kwargs = _process_multi_modal_data(
                        multi_modal_data,
                        prompts.meta_info["image_min_pixels"],
                        prompts.meta_info["image_max_pixels"],
                        prompts.meta_info["video_min_pixels"],
                        prompts.meta_info["video_max_pixels"],
                        prompts.meta_info["video_max_frames"],
                        prompts.meta_info["video_fps"],
                    )
                    if mm_data is not None:
                        item["multi_modal_data"] = mm_data
                        # Inject mm_processor_kwargs for video processing (e.g., do_sample_frames)
                        if mm_kwargs is not None:
                            item["mm_processor_kwargs"] = mm_kwargs

                vllm_inputs.append(item)
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**prompts.meta_info):
            completions: list[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=self.use_tqdm
            )
            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            pad_length = prompts.meta_info.get("max_tokens")
            if pad_length is None:
                pad_length = (
                    self.config.train_response_length
                    if self.config.train_response_length > 0
                    else self.config.response_length
                )
            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=pad_length
            ).to(input_ids.device)

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)

            # Mix-policy: 用离线轨迹替换有预采集数据样本的最后一个在线生成
            # 对于没有离线轨迹的样本，保留所有 n 个在线生成
            if self.enable_mix_policy and batch_has_offline is not None and self.sampling_params.n > 1:
                response_ids = self._mix_offline_trajectories(
                    response_ids=response_ids,
                    batch_has_offline=batch_has_offline,
                    batch_offline_output=batch_offline_output,
                    n=self.sampling_params.n,
                )

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.ndim == 3:  # qwen2vl mrope: (batch_size, 4, seq_length)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if batch_multi_modal_data is not None:
            non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        else:
            non_tensor_batch = {}

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)

    def _mix_offline_trajectories(
        self,
        response_ids: torch.Tensor,
        batch_has_offline: np.ndarray,
        batch_offline_output: np.ndarray,
        n: int,
    ) -> torch.Tensor:
        """
        Mix-policy 核心逻辑: 用离线轨迹替换有预采集数据样本的最后一个在线生成。

        对于有离线轨迹的样本: 在线生成 n-1 个 + 1 个离线 = n 个总样本
        对于没有离线轨迹的样本: 在线生成 n 个 = n 个总样本

        处理逻辑与原始 rollout 一致:
        1. Tokenize 离线轨迹得到 token list
        2. 使用 VF.pad_2d_list_to_length 进行 padding (与在线生成的处理方式一致)
        3. 后续的 EOS/mask 处理由 VF.get_response_mask 统一完成

        Args:
            response_ids: 在线生成的 response ids, shape (batch_size * n, response_length)
            batch_has_offline: 每个原始样本是否有离线轨迹, shape (original_batch_size,)
            batch_offline_output: 每个原始样本的离线输出文本, shape (original_batch_size,)
            n: 每个样本的生成数量

        Returns:
            混合后的 response_ids, shape (batch_size * n, response_length)
        """
        device = response_ids.device
        response_length = response_ids.size(1)
        original_batch_size = len(batch_has_offline)

        # 遍历每个原始样本
        for i in range(original_batch_size):
            # 检查该样本是否有离线轨迹
            has_offline = batch_has_offline[i] if batch_has_offline is not None else False
            offline_output = batch_offline_output[i] if batch_offline_output is not None else ""

            if has_offline and offline_output and len(offline_output.strip()) > 0:
                # 1. Tokenize 离线轨迹 (与 vLLM 生成的 token_ids 格式一致)
                offline_token_ids = self.tokenizer.encode(
                    offline_output,
                    add_special_tokens=False,
                )

                # 2. 使用与原始代码相同的 padding 方式
                # VF.pad_2d_list_to_length 接受 2D list，这里包装成单样本的 2D list
                offline_tokens = VF.pad_2d_list_to_length(
                    [offline_token_ids],  # 包装成 2D list: [[token1, token2, ...]]
                    self.pad_token_id,
                    max_length=response_length,
                ).to(device)
                # offline_tokens shape: (1, response_length)

                # 3. 计算要替换的位置：第 i 个样本的最后一个生成 (即第 n-1 个)
                # response_ids 的布局是: [sample_0_gen_0, sample_0_gen_1, ..., sample_0_gen_{n-1},
                #                         sample_1_gen_0, sample_1_gen_1, ..., sample_1_gen_{n-1}, ...]
                replace_idx = i * n + (n - 1)  # 替换最后一个在线生成
                response_ids[replace_idx] = offline_tokens.squeeze(0)

        # 后续的 EOS/mask 处理由 generate_sequences 中的 VF.get_response_mask 统一完成
        return response_ids
