# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""
OPD Trainer: on-policy distillation with an external homologous teacher.

Student and teacher share the SAME prompt token ids (no privileged hints).
Teacher scores the student's on-policy response; student matches via distill loss.

Reuses DataParallelOPSDActor.update_policy_opsd via loss_type="opsd" without
modifying OPSD source files.
"""

from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from .metrics import compute_timing_metrics, reduce_metrics
from .opd_config import OPDConfig, OPDTrainConfig
from .opsd_trainer import _concat_prompt_and_response
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role


class RayOPDTrainer(RayPPOTrainer):
    """Pure on-policy homologous distillation trainer."""

    def __init__(self, config: OPDTrainConfig, *args, **kwargs):
        original_n = config.worker.rollout.n
        if original_n == 1:
            config.worker.rollout.n = 2
        super().__init__(config, *args, **kwargs)
        config.worker.rollout.n = original_n

        self.opd_config: OPDConfig = self.config.opd
        # Wire worker flags consumed by OPDFSDPWorker / DataParallelOPSDActor.
        self.config.worker.actor.opsd_enabled = True
        self.config.worker.actor.freeze_teacher_model = True
        self.config.worker.actor.teacher_model_path = self.opd_config.teacher_model_path
        self.config.worker.actor.opd_assert_same_vocab = self.opd_config.assert_same_vocab

        print("[OPD] Mode: homologous on-policy distillation")
        print(f"[OPD]   student: {self.config.worker.actor.model.model_path}")
        print(f"[OPD]   teacher: {self.opd_config.teacher_model_path}")
        print(f"[OPD]   loss: {self.opd_config.distillation_loss_type} topk={self.opd_config.distillation_topk}")
        print("[OPD]   prompts: student == teacher (no privileged hint)")
        print(f"[OPD]   student_rollout_n: {self.opd_config.student_rollout_n}")

    def _get_student_max_response_length(self) -> int:
        data_cfg = self.config.data
        if not data_cfg.enable_dynamic_max_response_length:
            return int(data_cfg.max_response_length)
        step = self.global_step
        if step <= data_cfg.dynamic_max_response_length_phase1_end:
            return int(data_cfg.dynamic_max_response_length_phase1)
        if step <= data_cfg.dynamic_max_response_length_phase2_end:
            return int(data_cfg.dynamic_max_response_length_phase2)
        return int(data_cfg.dynamic_max_response_length_phase3)

    def _build_student_gen_batch(self, batch: DataProto) -> DataProto:
        student_gen = batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=[
                "raw_prompt_ids",
                "multi_modal_data",
                "has_offline_trajectory",
                "offline_output",
            ],
            meta_info_keys=[
                "image_min_pixels",
                "image_max_pixels",
                "video_min_pixels",
                "video_max_pixels",
                "video_fps",
                "video_max_frames",
            ],
        )
        if "raw_prompt" in batch.non_tensor_batch:
            student_gen.non_tensor_batch["raw_prompt"] = batch.non_tensor_batch["raw_prompt"].copy()
        return student_gen

    def _opd_meta_config(self) -> dict[str, Any]:
        """Meta passed to DataParallelOPSDActor.update_policy_opsd (on-policy only)."""
        return {
            "alpha": 1.0,
            "lambda_nav": 0.0,
            "distillation_loss_type": self.opd_config.distillation_loss_type,
            "opsd_kl_penalty": self.opd_config.opsd_kl_penalty,
            "distillation_topk": self.opd_config.distillation_topk,
            "distillation_topk_source": self.opd_config.distillation_topk_source,
            "distillation_add_tail": self.opd_config.distillation_add_tail,
            "distill_token_selection": self.opd_config.distill_token_selection,
            "distill_token_keep_ratio": self.opd_config.distill_token_keep_ratio,
            "dynamic_sample_hint": False,
            "teacher_rollout_n": 1,
            "outcome_ppo_coef": 0.0,
            "outcome_ppo_incorrect_only": False,
            "nav_clip_ratio_low": 0.2,
            "nav_clip_ratio_high": 0.3,
            "nav_clip_ratio_dual": 3.0,
            "self_distill_negative_off_policy": False,
            "self_distill_negative_off_policy_coef": 0.0,
            "self_distill_mask_prefix_tokens": 0,
            "overlap_topk": self.opd_config.overlap_topk,
            "enable_topk_overlap_metrics": self.opd_config.enable_topk_overlap_metrics,
            # Never sync external teacher from student weights.
            "stgca_teacher_sync_interval": 0,
            "global_step": self.global_step,
        }

    def _make_opd_batch(self, metrics: dict[str, Any]) -> tuple[DataProto, DataProto, dict[str, Any]]:
        timing_raw: dict[str, float] = {}

        with timer("data_load", timing_raw):
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

        meta_info = {
            "image_min_pixels": self.config.data.image_min_pixels,
            "image_max_pixels": self.config.data.image_max_pixels,
            "video_min_pixels": self.config.data.video_min_pixels,
            "video_max_pixels": self.config.data.video_max_pixels,
            "video_fps": self.config.data.video_fps,
            "video_max_frames": self.config.data.video_max_frames,
        }
        batch = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
        batch_size = len(batch.batch)
        batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(batch_size)], dtype=object
        )

        student_rollout_n = int(self.opd_config.student_rollout_n)
        student_max_response_length = self._get_student_max_response_length()
        metrics["data/student_max_response_length"] = float(student_max_response_length)

        student_gen_base = self._build_student_gen_batch(batch)

        with timer("student_rollout", timing_raw):
            student_gen = deepcopy(student_gen_base)
            student_gen.meta_info.update(
                {
                    "n": student_rollout_n,
                    "temperature": self.config.worker.rollout.temperature,
                    "top_p": self.config.worker.rollout.top_p,
                    "max_tokens": student_max_response_length,
                    "image_min_pixels": self.config.data.image_min_pixels,
                    "image_max_pixels": self.config.data.image_max_pixels,
                    "video_min_pixels": self.config.data.video_min_pixels,
                    "video_max_pixels": self.config.data.video_max_pixels,
                    "video_fps": self.config.data.video_fps,
                    "video_max_frames": self.config.data.video_max_frames,
                }
            )
            student_gen, student_pad_size = pad_dataproto_to_divisor(
                student_gen, self.actor_rollout_ref_wg.world_size
            )
            student_output = self.actor_rollout_ref_wg.generate_sequences(student_gen)
            student_output = unpad_dataproto(
                student_output, pad_size=student_pad_size * student_rollout_n
            )

        # Expand original batch non-tensors to match rollout.n.
        if student_rollout_n > 1:
            for key, value in list(batch.non_tensor_batch.items()):
                batch.non_tensor_batch[key] = np.repeat(value, student_rollout_n)

        num_samples = student_output.batch["responses"].shape[0]
        metrics["opd/num_student_samples"] = float(num_samples)

        # Same prompt for teacher and student: reuse full student sequences.
        on_student_input_ids = student_output.batch["input_ids"]
        on_student_attention_mask = student_output.batch["attention_mask"]
        on_student_position_ids = student_output.batch["position_ids"]
        on_student_responses = student_output.batch["responses"]
        on_student_response_mask = student_output.batch["response_mask"]

        # Explicit concat from student prompts for clarity / logging.
        prompt_ids = student_output.batch["prompts"]
        prompt_len = prompt_ids.shape[1]
        prompt_attention_mask = on_student_attention_mask[:, :prompt_len]
        prompt_position_ids = on_student_position_ids[..., :prompt_len]
        on_teacher_input_ids, on_teacher_attention_mask, on_teacher_position_ids = _concat_prompt_and_response(
            prompt_ids=prompt_ids,
            prompt_attention_mask=prompt_attention_mask,
            prompt_position_ids=prompt_position_ids,
            response_ids=on_student_responses,
            response_mask=on_student_response_mask,
        )

        # Minimal nav placeholders (lambda_nav=0 → skipped in actor).
        nav_resp_len = 1
        nav_tensors = {
            "nav_input_ids": torch.zeros(num_samples, 1, dtype=torch.long),
            "nav_attention_mask": torch.zeros(num_samples, 1, dtype=torch.long),
            "nav_position_ids": torch.zeros(num_samples, 1, dtype=torch.long),
            "nav_responses": torch.zeros(num_samples, 1, dtype=torch.long),
            "nav_response_mask": torch.zeros(num_samples, nav_resp_len),
            "nav_old_log_probs": torch.zeros(num_samples, nav_resp_len),
            "nav_advantages": torch.zeros(num_samples, nav_resp_len),
        }

        on_policy_tensors = {
            "on_student_input_ids": on_student_input_ids,
            "on_student_attention_mask": on_student_attention_mask,
            "on_student_position_ids": on_student_position_ids,
            "on_student_responses": on_student_responses,
            "on_student_response_mask": on_student_response_mask,
            "on_teacher_input_ids": on_teacher_input_ids,
            "on_teacher_attention_mask": on_teacher_attention_mask,
            "on_teacher_position_ids": on_teacher_position_ids,
            "on_teacher_responses": on_student_responses,
            "on_teacher_response_mask": on_student_response_mask,
            "on_student_distill_response_mask": on_student_response_mask,
        }

        all_tensors = {}
        all_tensors.update(nav_tensors)
        all_tensors.update(on_policy_tensors)

        original_mmd = batch.non_tensor_batch.get("multi_modal_data")
        original_uid = batch.non_tensor_batch.get("uid")
        if original_mmd is None:
            on_mmd = np.array([{}] * num_samples, dtype=object)
        else:
            on_mmd = original_mmd
        if original_uid is None:
            on_uid = np.array([f"opd_{i}" for i in range(num_samples)], dtype=object)
        else:
            on_uid = original_uid

        non_tensors = {
            "nav_multi_modal_data": np.array([{}] * num_samples, dtype=object),
            "nav_uid": np.array([f"__nav_{i}" for i in range(num_samples)], dtype=object),
            "on_multi_modal_data": on_mmd,
            "on_uid": on_uid,
            "off_multi_modal_data": np.array([{}] * num_samples, dtype=object),
            "off_uid": np.array([f"__off_{i}" for i in range(num_samples)], dtype=object),
        }

        combined_batch = DataProto.from_dict(
            tensors=all_tensors,
            non_tensors=non_tensors,
            meta_info={
                "temperature": self.config.worker.rollout.temperature,
                "loss_type": "opsd",
                "opsd_config": self._opd_meta_config(),
                "num_valid_off_policy": 0,
                "num_valid_on_policy": num_samples,
                "global_token_num": torch.sum(on_teacher_attention_mask, dim=-1).tolist(),
                "image_min_pixels": self.config.data.image_min_pixels,
                "image_max_pixels": self.config.data.image_max_pixels,
                "video_min_pixels": self.config.data.video_min_pixels,
                "video_max_pixels": self.config.data.video_max_pixels,
                "video_fps": self.config.data.video_fps,
                "video_max_frames": self.config.data.video_max_frames,
            },
        )

        metrics.update(compute_timing_metrics(batch=student_output, timing_raw=timing_raw))
        metrics["opd/same_prompt"] = 1.0
        print(
            f"[OPD] Batch ready: B={batch_size}, samples={num_samples}, "
            f"resp_len={student_max_response_length}"
        )
        return combined_batch, student_output, metrics

    def fit(self):
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="OPD Training", position=0)
        val_metrics = None

        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)

        while self.global_step < self.training_steps:
            self.global_step += 1
            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    combined_batch, original_batch, gen_metrics = self._make_opd_batch(metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()
                    metrics.update(gen_metrics)

                with timer("update_actor", timing_raw):
                    actor_output = self.actor_rollout_ref_wg.update_actor(combined_batch)

                actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                metrics.update(actor_metrics)

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            metrics.update(compute_timing_metrics(batch=original_batch, timing_raw=timing_raw))
            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        # Final validation only when validation is enabled in the schedule.
        if (
            self.val_reward_fn is not None
            and self.config.trainer.val_freq > 0
            and (
                val_metrics is None
                or self.global_step % self.config.trainer.val_freq != 0
            )
        ):
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        # Final checkpoint only when saving is enabled.
        if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
        print("[OPD] Training finished.")
