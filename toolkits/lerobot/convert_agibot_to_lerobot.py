"""
AgiBot World Alpha → LeRobot v0.3.3 格式转换脚本
用法：
    python convert_agibot_to_lerobot.py \
        --src_path /path/to/sample_dataset \
        --task_id 355 \
        --tgt_path /path/to/output
"""
import argparse
import json
from pathlib import Path

import av
import h5py
import numpy as np

FEATURES = {
    "observation.images.top_head": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.hand_left": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.hand_right": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (20,),
        "names": [
            "left_arm_j0", "left_arm_j1", "left_arm_j2", "left_arm_j3",
            "left_arm_j4", "left_arm_j5", "left_arm_j6",
            "right_arm_j0", "right_arm_j1", "right_arm_j2", "right_arm_j3",
            "right_arm_j4", "right_arm_j5", "right_arm_j6",
            "left_effector", "right_effector",
            "head_0", "head_1",
            "waist_0", "waist_1",
        ],
    },
    "action": {
        "dtype": "float32",
        "shape": (22,),
        "names": [
            "left_arm_j0", "left_arm_j1", "left_arm_j2", "left_arm_j3",
            "left_arm_j4", "left_arm_j5", "left_arm_j6",
            "right_arm_j0", "right_arm_j1", "right_arm_j2", "right_arm_j3",
            "right_arm_j4", "right_arm_j5", "right_arm_j6",
            "left_effector", "right_effector",
            "head_0", "head_1",
            "waist_0", "waist_1",
            "robot_vel_0", "robot_vel_1",
        ],
    },
}

VIDEO_MAP = {
    "observation.images.top_head": "head_color.mp4",
    "observation.images.hand_left": "hand_left_color.mp4",
    "observation.images.hand_right": "hand_right_color.mp4",
}


def get_task_instruction(task_json_path: Path) -> str:
    with open(task_json_path) as f:
        info = json.load(f)
    task_name = info[0]["task_name"]
    init_scene = info[0]["init_scene_text"]
    return f"{task_name}. {init_scene}"


def decode_video_frames(video_path: Path) -> list[np.ndarray]:
    """解码 mp4 为逐帧 numpy array，格式 (H, W, C) uint8"""
    frames = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def load_episode(src_path: Path, task_id: str, episode_id: int):
    ob_dir = src_path / "observations" / task_id / str(episode_id)
    proprio_file = src_path / "proprio_stats" / task_id / str(episode_id) / "proprio_stats.h5"

    with h5py.File(proprio_file) as f:
        state_joint    = np.array(f["state/joint/position"])       # (T, 14)
        state_effector = np.array(f["state/effector/position"])    # (T, 2)
        state_head     = np.array(f["state/head/position"])        # (T, 2)
        state_waist    = np.array(f["state/waist/position"])       # (T, 2)
        action_joint    = np.array(f["action/joint/position"])     # (T, 14)
        action_effector = np.array(f["action/effector/position"])  # (T, 2)
        action_head     = np.array(f["action/head/position"])      # (T, 2)
        action_waist    = np.array(f["action/waist/position"])     # (T, 2)
        action_velocity = np.array(f["action/robot/velocity"])     # (T, 2)

    states  = np.hstack([state_joint, state_effector, state_head, state_waist]).astype(np.float32)
    actions = np.hstack([action_joint, action_effector, action_head, action_waist, action_velocity]).astype(np.float32)

    print(f"    Decoding videos...")
    video_frames = {}
    for key, fname in VIDEO_MAP.items():
        vpath = ob_dir / "videos" / fname
        if not vpath.exists():
            raise FileNotFoundError(f"Video not found: {vpath}")
        video_frames[key] = decode_video_frames(vpath)
        print(f"    {key}: {len(video_frames[key])} frames")

    return states, actions, video_frames


def main(src_path: str, task_id: str, tgt_path: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    src = Path(src_path)
    task_json = src / "task_info" / f"task_{task_id}.json"
    instruction = get_task_instruction(task_json)
    print(f"Task instruction: {instruction}")

    repo_id = f"agibotworld/task_{task_id}"
    out_root = Path(tgt_path) / repo_id
    if out_root.exists():
        import shutil
        shutil.rmtree(out_root)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        features=FEATURES,
        root=out_root,
        robot_type="a2d",
        use_videos=True,
        image_writer_threads=4,
    )

    all_episodes = sorted(
        int(p.name) for p in (src / "observations" / task_id).iterdir() if p.is_dir()
    )
    print(f"Found {len(all_episodes)} episodes for task {task_id}")

    for ep_id in all_episodes:
        print(f"  Processing episode {ep_id}...")
        states, actions, video_frames = load_episode(src, task_id, ep_id)

        T = len(states)
        # 对齐帧数（视频解码帧数可能和 h5 帧数有 ±1 差异）
        for key, frames in video_frames.items():
            if len(frames) != T:
                print(f"    WARNING: {key} has {len(frames)} frames, h5 has {T}. Truncating to min.")
        T = min(T, *[len(f) for f in video_frames.values()])

        print(f"    Adding {T} frames...")
        for i in range(T):
            frame = {
                "observation.state": states[i],
                "action": actions[i],
                "observation.images.top_head": video_frames["observation.images.top_head"][i],
                "observation.images.hand_left": video_frames["observation.images.hand_left"][i],
                "observation.images.hand_right": video_frames["observation.images.hand_right"][i],
            }
            dataset.add_frame(frame, task=instruction)

        dataset.save_episode()
        print(f"  Episode {ep_id} saved.")

    print(f"Done! Saved to {dataset.root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", required=True)
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--tgt_path", required=True)
    args = parser.parse_args()
    main(args.src_path, args.task_id, args.tgt_path)


