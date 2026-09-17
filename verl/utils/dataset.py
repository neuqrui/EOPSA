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

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from . import torch_functional as VF
from .prompt_template import build_prompt


def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def process_video(
    video: str,
    min_pixels: int = 4 * 32 * 32,
    max_pixels: int = 64 * 32 * 32,
    max_frames: int = 128,
    video_fps: float = 2.0,
    return_fps: bool = False,
) -> Union[list[ImageObject], tuple[list[ImageObject], list[float]]]:
    vision_info = {
        "video": video,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "max_frames": max_frames,
        "fps": video_fps,
    }
    return fetch_video(
        vision_info,
        image_patch_size=16,
        return_video_sample_fps=return_fps,
        return_video_metadata=return_fps,
    )


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        video_max_frames: int = 128,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        image_min_pixels: Optional[int] = None,
        image_max_pixels: Optional[int] = None,
        video_min_pixels: Optional[int] = None,
        video_max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
        use_preprocessed_videos: bool = True,
        preprocessed_video_dir: Optional[str] = None,
        trajectory_key: Optional[str] = None,
        apply_chat_template_kwargs: Optional[dict] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.apply_chat_template_kwargs = apply_chat_template_kwargs or {}
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.video_max_frames = video_max_frames
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels
        self.video_min_pixels = video_min_pixels
        self.video_max_pixels = video_max_pixels
        self.use_preprocessed_videos = use_preprocessed_videos
        self.preprocessed_video_dir = preprocessed_video_dir
        self.trajectory_key = trajectory_key

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )

    def _build_prompt_text(self, example: dict[str, Any]) -> str:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            # 传递完整的 example 字段给 Jinja 模板，支持按 problem_type 路由
            # 可用变量: problem, problem_type, data_type, options, data_source 等
            prompt_str = format_prompt.render(
                content=prompt_str,  # 兼容旧模板
                problem=prompt_str,  # 问题文本
                **{k: v for k, v in example.items() if k != self.prompt_key}  # 其他字段
            )
        else:
            # 只有在没有format_prompt时，才使用build_prompt添加任务特定指令
            # 注意：如果原始数据已经包含完整指令，build_prompt会导致指令重复
            prompt_str = build_prompt(prompt_str, example)
        return prompt_str

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_str = self._build_prompt_text(example)

        # Check if images exist and is a non-empty list/array
        # 支持 list 和 numpy.ndarray 两种类型
        images_data = example.get(self.image_key)
        has_images = (
            self.image_key in example
            and images_data is not None
            and hasattr(images_data, '__len__')
            and len(images_data) > 0
        )
        # Check if videos exist and is a non-empty list/array
        videos_data = example.get(self.video_key)
        has_videos = (
            self.video_key in example
            and videos_data is not None
            and hasattr(videos_data, '__len__')
            and len(videos_data) > 0
        )

        if has_images:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        elif has_videos:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        # Check if images exist and is a non-empty list/array
        # 支持 list 和 numpy.ndarray 两种类型
        images_data = example.get(self.image_key)
        has_images = (
            self.image_key in example
            and images_data is not None
            and hasattr(images_data, '__len__')
            and len(images_data) > 0
        )
        # Check if videos exist and is a non-empty list/array
        videos_data = example.get(self.video_key)
        has_videos = (
            self.video_key in example
            and videos_data is not None
            and hasattr(videos_data, '__len__')
            and len(videos_data) > 0
        )

        if has_images:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs)
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.image_min_pixels, self.image_max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif has_videos:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs)

            # 优先使用预处理的视频文件（与 __getitem__ 逻辑对齐）
            if self.use_preprocessed_videos and "preprocessed_video" in example and example["preprocessed_video"]:
                preprocessed_video_file = example["preprocessed_video"]
                if self.preprocessed_video_dir is not None:
                    preprocessed_video_path = os.path.join(self.preprocessed_video_dir, preprocessed_video_file)
                else:
                    preprocessed_video_path = preprocessed_video_file

                if os.path.exists(preprocessed_video_path):
                    preprocessed_data = torch.load(preprocessed_video_path, map_location="cpu", weights_only=False)
                    processed_videos = [preprocessed_data["frames"]]
                    video_metadatas = [preprocessed_data["metadata"]]
                    model_inputs = self.processor(
                        videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt",
                        video_metadata=video_metadatas,
                        do_resize=False,
                        do_sample_frames=False,
                    )
                    return model_inputs["input_ids"].size(-1) <= self.max_prompt_length

            # fallback: 实时解码原始视频
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_metadatas = []
            for video in videos:
                result = process_video(
                    video,
                    min_pixels=self.video_min_pixels if self.video_min_pixels else 4 * 32 * 32,
                    max_pixels=self.video_max_pixels if self.video_max_pixels else 64 * 32 * 32,
                    max_frames=self.video_max_frames,
                    video_fps=self.video_fps,
                    return_fps=True
                )
                if isinstance(result, tuple) and len(result) == 2:
                    video_data, _ = result  # Unpack (video_data, sample_fps)
                    if isinstance(video_data, tuple) and len(video_data) == 2:
                        frames, metadata = video_data
                        processed_videos.append(frames)
                        video_metadatas.append(metadata)
                    else:
                        processed_videos.append(video_data)
                        video_metadatas = None
                        break
                else:
                    processed_videos.append(result)
                    video_metadatas = None
                    break

            if video_metadatas is not None and len(video_metadatas) > 0:
                model_inputs = self.processor(
                    videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt",
                    video_metadata=video_metadatas,
                    do_resize=False,
                    do_sample_frames=False,
                )
            else:
                model_inputs = self.processor(
                    videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt",
                    do_sample_frames=False,
                )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, **self.apply_chat_template_kwargs)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        teacher_question = example[self.prompt_key]
        prompt_text = self._build_prompt_text(example)
        messages = self._build_messages(example)
        example.pop(self.prompt_key, None)

        # Check if images exist and is a non-empty list/array
        # 支持 list 和 numpy.ndarray 两种类型
        images_data = example.get(self.image_key)
        has_images = (
            self.image_key in example
            and images_data is not None
            and hasattr(images_data, '__len__')
            and len(images_data) > 0
        )
        # Check if videos exist and is a non-empty list/array
        videos_data = example.get(self.video_key)
        has_videos = (
            self.video_key in example
            and videos_data is not None
            and hasattr(videos_data, '__len__')
            and len(videos_data) > 0
        )

        if has_images:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs)
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.image_min_pixels, self.image_max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}
        elif has_videos:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs)
            videos = example.pop(self.video_key)

            # 保存原始视频路径（用于 fallback）
            original_video_paths = videos if isinstance(videos, list) else [videos]

            # 优先使用预处理的视频文件
            use_preprocessed_path = False
            preprocessed_video_path = None
            if self.use_preprocessed_videos and "preprocessed_video" in example and example["preprocessed_video"]:
                preprocessed_video_file = example.pop("preprocessed_video")
                # 构建完整路径
                if self.preprocessed_video_dir is not None:
                    preprocessed_video_path = os.path.join(self.preprocessed_video_dir, preprocessed_video_file)
                else:
                    preprocessed_video_path = preprocessed_video_file

                # 加载预处理的视频数据（仅用于 tokenize 确定 token 数量）
                if os.path.exists(preprocessed_video_path):
                    preprocessed_data = torch.load(preprocessed_video_path, map_location="cpu", weights_only=False)
                    processed_videos = [preprocessed_data["frames"]]
                    video_metadatas = [preprocessed_data["metadata"]]
                    video_fps_list = [preprocessed_data["sample_fps"]]
                    video_kwargs = {"do_sample_frames": False}
                    use_preprocessed_path = True
                else:
                    # 降级：预处理文件不存在，使用原始处理逻辑
                    print(f"Warning: Preprocessed video file not found: {preprocessed_video_path}, falling back to real-time processing")
                    preprocessed_video_path = None
                    if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):
                        videos = [os.path.join(self.image_dir, video) for video in videos]
                    processed_videos = []
                    video_fps_list = []
                    video_metadatas = []
                    video_kwargs = {"do_sample_frames": False}
                    for video in videos:
                        processed_video, video_fps = process_video(
                            video,
                            min_pixels=self.video_min_pixels if self.video_min_pixels else 4 * 32 * 32,
                            max_pixels=self.video_max_pixels if self.video_max_pixels else 64 * 32 * 32,
                            max_frames=self.video_max_frames,
                            video_fps=self.video_fps,
                            return_fps=True
                        )
                        processed_videos.append(processed_video)
                        video_fps_list.append(video_fps)
            else:
                # 原始处理逻辑：实时处理视频
                if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                    videos = [os.path.join(self.image_dir, video) for video in videos]

                processed_videos = [] if len(videos) != 0 else None  # text-only data
                video_fps_list = []
                video_kwargs = {"do_sample_frames": False}  # For Qwen3-VL
                for video in videos:
                    # Use video min/max pixels to match training stage
                    processed_video, video_fps = process_video(
                        video,
                        min_pixels=self.video_min_pixels if self.video_min_pixels else 4 * 32 * 32,
                        max_pixels=self.video_max_pixels if self.video_max_pixels else 64 * 32 * 32,
                        max_frames=self.video_max_frames,
                        video_fps=self.video_fps,
                        return_fps=True
                    )
                    processed_videos.append(processed_video)
                    video_fps_list.append(video_fps)

            # Handle video_metadata for Qwen3-VL
            if processed_videos is not None and len(processed_videos) > 0:
                # 检查 video_metadatas 是否已经在预处理加载阶段设置
                if 'video_metadatas' in locals() and video_metadatas is not None and len(video_metadatas) > 0:
                    # 预处理视频：直接使用 processed_videos 作为 frames
                    processed_video_frames = processed_videos
                else:
                    # 实时处理视频：process_video returns (frames, metadata) when return_fps=True
                    processed_video_frames = []
                    video_metadatas = []
                    for pv in processed_videos:
                        if isinstance(pv, tuple) and len(pv) == 2:
                            frames, metadata = pv
                            processed_video_frames.append(frames)
                            video_metadatas.append(metadata)
                        else:
                            processed_video_frames.append(pv)
                            video_metadatas = None
                            break

                if video_metadatas is not None and len(video_metadatas) > 0:
                    model_inputs = self.processor(
                        text=[prompt],
                        videos=processed_video_frames,
                        add_special_tokens=False,
                        video_metadata=video_metadatas,
                        return_tensors="pt",
                        do_resize=False,
                        **video_kwargs,
                    )
                else:
                    model_inputs = self.processor(
                        videos=processed_video_frames,
                        text=[prompt],
                        add_special_tokens=False,
                        return_tensors="pt",
                        **video_kwargs,
                    )
            else:
                model_inputs = self.processor(
                    videos=processed_videos,
                    text=[prompt],
                    add_special_tokens=False,
                    return_tensors="pt",
                )

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

            # 存储 multi_modal_data
            # 如果有预处理路径，只存路径字符串（~100B），Worker 端本地加载
            if use_preprocessed_path and preprocessed_video_path is not None:
                example["multi_modal_data"] = {
                    "preprocessed_video_path": preprocessed_video_path,
                }
            elif processed_video_frames is not None and len(processed_video_frames) > 0:
                # 原始模式：存储 PIL 图像列表
                multi_modal_data_content = {"video": processed_video_frames}
                if video_metadatas is not None:
                    multi_modal_data_content["video_metadatas"] = video_metadatas
                example["multi_modal_data"] = multi_modal_data_content
            else:
                example["multi_modal_data"] = {"videos": videos}
        else:
            # 纯文本样本（没有图片和视频）
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            # 为纯文本样本设置空字典，确保批次一致性（不能是 None，否则 vllm_rollout 会报错）
            example["multi_modal_data"] = {}

        # Clean up images/videos keys if they still exist
        if self.image_key in example:
            example.pop(self.image_key, None)
        if self.video_key in example:
            example.pop(self.video_key, None)

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["raw_prompt"] = prompt  # 保存原始文本prompt（用于vLLM预处理视频）
        example["prompt_text"] = prompt_text  # chat template 之前的文本，用于OPSD二次构造提示
        example["teacher_question"] = teacher_question  # 原始题目文本，用于Teacher专属模板
        example["ground_truth"] = example.pop(self.answer_key)

        # Mix-policy: 保留预采集轨迹信息
        # 数据格式: {"has_offline_trajectory": true, "offline_output": "<think>...</think><answer>...</answer>"}
        if "has_offline_trajectory" in example and example.get("has_offline_trajectory"):
            example["has_offline_trajectory"] = True
            example["offline_output"] = example.get("offline_output", "")
        else:
            example["has_offline_trajectory"] = False
            example["offline_output"] = ""

        # OPSD: 提取大模型轨迹（用于Navigator提示生成）
        if self.trajectory_key is not None:
            example["trajectory"] = example.pop(self.trajectory_key, "")
        else:
            example["trajectory"] = ""

        # 确保所有样本都有preprocessed_video字段（图片样本设为None）
        # 这是为了满足DataProto的batch一致性检查
        if "preprocessed_video" not in example:
            example["preprocessed_video"] = None

        return example
