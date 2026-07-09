# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DreamZero embodiment transform for AgiBot World Alpha (G1 robot).

Dataset structure (LeRobot v0.3.3 format after conversion):
  - Videos : observation.images.top_head  (head front RGB)
             observation.images.hand_left (left wrist RGB)
             observation.images.hand_right (right wrist RGB)
  - State  : observation.state  shape=(20,)
               [left_arm×7, right_arm×7, left_effector, right_effector,
                head×2, waist×2]
  - Action : action  shape=(22,)
               [left_arm×7, right_arm×7, left_effector, right_effector,
                head×2, waist×2, robot_velocity×2]
  - Language: task  (tasks.jsonl → task field)

Camera layout (matches DreamZero-AgiBot training config):
  top row    : top_head  (full width)
  bottom-left: hand_left
  bottom-right: hand_right
"""

from typing import Any

import numpy as np
from groot.vla.data.dataset.lerobot import ModalityConfig
from groot.vla.data.transform.base import ComposedModalityTransform
from groot.vla.data.transform.concat import ConcatTransform
from groot.vla.data.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from groot.vla.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)

from rlinf.data.datasets.dreamzero.data_transforms.base import RolloutObsLayout
from rlinf.data.datasets.dreamzero.data_transforms.dream_transform import DreamTransform

# LeRobot video keys → DreamZero transform keys (short suffix maps automatically)
_VIDEO_KEYS = [
    "video.top_head",
    "video.hand_left",
    "video.hand_right",
]

# key 格式必须是 modality.name，其中 name 匹配 metadata.json modalities 里的 key
# metadata 生成时 --state-key observation.state → modalities.state.state
# metadata 生成时 --action-key action → modalities.action.actions
_STATE_KEYS = ["state.state"]
_ACTION_KEYS = ["action.actions"]

_VIDEO_BACKEND = "torchvision"

_TRAINING_PROMPT_PREFIX = "A multi-view video shows that a robot "
_MULTIVIEW_LAYOUT = (
    " The video is split into three views: The top view shows the camera view "
    "from the robot's head, the bottom-left view shows the camera view from the "
    "left hand camera, and the bottom-right view shows the camera view from "
    "the right hand camera. The robot "
)


class AgibotAlphaDataTransform:
    """Provides modality config and composed transform for agibot_alpha (G1 robot)."""

    TAG = "agibot_alpha"
    # Use projector id 26 — available in DreamZero-AgiBot action_loss_embodiment_ids
    DEFAULT_TAG_MAPPING = {"agibot_alpha": 26}
    DEFAULT_ACTION_HORIZON = 48

    ROLLOUT_OBS_LAYOUT = RolloutObsLayout(
        video_fields=(
            ("main_images", "video.top_head"),
            ("wrist_images", "video.hand_left"),
            ("extra_view_images", "video.hand_right"),
        ),
        state_fields=(
            ("states", "state.state"),
        ),
        binarize_gripper=False,
        fill_missing_video_keys=True,
    )

    @staticmethod
    def format_training_prompt(instruction: str) -> str:
        """Build multi-view layout prompt for AgiBot Alpha."""
        return _TRAINING_PROMPT_PREFIX + instruction + _MULTIVIEW_LAYOUT + instruction

    @staticmethod
    def concat_multiview_video(images: np.ndarray) -> np.ndarray:
        """Layout: top_head spans full top row; hand_left/right on bottom row.

        Input shape: (v=3, t, c, h, w)
        Output shape: (1, t, c, 2h, 2w)
        """
        v, t, c, h, w = images.shape
        if v < 3:
            raise ValueError(
                f"agibot_alpha expects 3 video views, got v={v} with shape {images.shape}"
            )
        top_head   = images[0]  # (t, c, h, w)
        hand_left  = images[1]
        hand_right = images[2]

        concat = np.zeros((1, t, c, 2 * h, 2 * w), dtype=images.dtype)
        # top row: head spans full width
        top_wide = np.repeat(top_head, 2, axis=-1)   # (t, c, h, 2w)
        concat[0, :, :, :h, :]  = top_wide
        # bottom row: hand_left left half, hand_right right half
        concat[0, :, :, h:, :w] = hand_left
        concat[0, :, :, h:, w:] = hand_right
        return concat

    @staticmethod
    def get_modality_config() -> dict[str, ModalityConfig]:
        """Return modality config dict for agibot_alpha.

        action_horizon=48 → delta_indices range(48), video range(49)=8*6+1.
        """
        return {
            "video": ModalityConfig(
                delta_indices=list(range(49)),
                eval_delta_indices=[0],
                modality_keys=list(_VIDEO_KEYS),
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=list(_STATE_KEYS),
            ),
            "action": ModalityConfig(
                delta_indices=list(range(48)),
                modality_keys=list(_ACTION_KEYS),
            ),
            "language": ModalityConfig(
                delta_indices=[0],
                modality_keys=["task"],
            ),
        }

    @staticmethod
    def get_transform(
        *,
        tokenizer_path: str,
        cfg: Any,
        embodiment_tag_mapping: dict[str, int],
        transform_on_gpu: bool = False,
    ) -> ComposedModalityTransform:
        """Build the full ComposedModalityTransform chain for agibot_alpha."""
        return AgibotAlphaDataTransform._build_composed_transform(
            tokenizer_path=tokenizer_path,
            state_horizon=int(cfg.get("state_horizon", 1)),
            action_horizon=int(
                cfg.get("action_horizon", AgibotAlphaDataTransform.DEFAULT_ACTION_HORIZON)
            ),
            max_state_dim=int(cfg.get("max_state_dim", 64)),
            max_action_dim=int(cfg.get("max_action_dim", 32)),
            max_length=int(cfg.get("max_seq_len", 512)),
            default_instruction=str(
                cfg.get("default_instruction", "Perform the default behavior.")
            ),
            language_dropout_prob=float(cfg.get("language_dropout_prob", 0.0)),
            always_use_default_instruction=bool(
                cfg.get("always_use_default_instruction", False)
            ),
            embodiment_tag_mapping=dict(embodiment_tag_mapping),
            transform_on_gpu=transform_on_gpu,
        )

    @staticmethod
    def _build_composed_transform(
        tokenizer_path: str,
        state_horizon: int,
        action_horizon: int,
        max_state_dim: int,
        max_action_dim: int,
        max_length: int,
        default_instruction: str,
        language_dropout_prob: float,
        always_use_default_instruction: bool,
        embodiment_tag_mapping: dict[str, int],
        transform_on_gpu: bool = False,
    ) -> ComposedModalityTransform:
        vk = list(_VIDEO_KEYS)
        state_k = list(_STATE_KEYS)
        action_k = list(_ACTION_KEYS)

        transforms: list[Any] = [
            VideoToTensor(
                apply_to=vk, backend=_VIDEO_BACKEND, output_on_cuda=transform_on_gpu
            ),
            VideoCrop(apply_to=vk, backend=_VIDEO_BACKEND, scale=0.95),
            VideoResize(
                apply_to=vk,
                backend=_VIDEO_BACKEND,
                height=176,
                width=320,
                interpolation="linear",
            ),
            VideoColorJitter(
                apply_to=vk,
                backend=_VIDEO_BACKEND,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=vk, backend=_VIDEO_BACKEND),
            StateActionToTensor(apply_to=state_k),
            StateActionTransform(
                apply_to=state_k,
                normalization_modes={
                    "state.state": "q99",
                },
            ),
            StateActionToTensor(apply_to=action_k),
            StateActionTransform(
                apply_to=action_k,
                normalization_modes={
                    "action.actions": "q99",
                },
            ),
            ConcatTransform(
                apply_to=[],
                video_concat_order=vk,
                state_concat_order=state_k,
                action_concat_order=action_k,
            ),
            DreamTransform(
                default_instruction=default_instruction,
                language_dropout_prob=language_dropout_prob,
                always_use_default_instruction=always_use_default_instruction,
                max_state_dim=max_state_dim,
                max_action_dim=max_action_dim,
                max_length=max_length,
                state_horizon=state_horizon,
                action_horizon=action_horizon,
                tokenizer_path=tokenizer_path,
                embodiment_tag_mapping=embodiment_tag_mapping,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)
