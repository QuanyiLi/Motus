#!/usr/bin/env python3
"""
Evaluate StandaloneMotusPolicy in vla-align environments.
Similar to gwm_eval.py / lerobot_eval.py.
"""
import argparse
import json
import os
import shutil
import time

os.environ["SVT_LOG"] = "0"
import datasets
import torch
import numpy as np
import cv2

datasets.disable_progress_bar()

# Import vla_align components
from vla_align import PROJECT_ROOT
from vla_align.env.config import get_env_cfg, MAX_EPISODE_STEP_WORKSPACE
from vla_align.utils.env import build_endless_env
from vla_align.utils.rollout import rollout, calculate_averages
from vla_align.utils.lerobot import obs_state_key, obs_image_prefix

# Import StandaloneMotusPolicy
import sys
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from standalone_inference import StandaloneMotusPolicy

image_1_key = f"{obs_image_prefix}.image_1"
image_2_key = f"{obs_image_prefix}.image_2"
wrist_image_key = f"{obs_image_prefix}.wrist_image"


def parse_args():
    parser = argparse.ArgumentParser(description="Motus Evaluation")
    parser.add_argument("--ckpt_path", type=str, default="",
                        help="Path to Motus checkpoint mp_rank_00_model_states.pt")
    parser.add_argument("--wan_path", type=str, default="",
                        help="Path to Wan2.2-TI2V-5B")
    parser.add_argument("--vlm_path", type=str, default="",
                        help="Path to Qwen3-VL-2B-Instruct")
    parser.add_argument("--result_dir", type=str, default="./motus_eval_result",
                        help="Output directory for results")
    parser.add_argument("--start_subset", type=int, default=0,
                        help="Start config index (inclusive)")
    parser.add_argument("--end_subset", type=int, default=24,
                        help="End config index (exclusive)")
    parser.add_argument("--split", type=str, default="both",
                        choices=["train", "test", "both"],
                        help="Which split(s) to evaluate")
    parser.add_argument("--aggregate_only", action="store_true",
                        help="Skip rollouts, only compute final aggregated results")
    parser.add_argument("--num_env", type=int, default=12,
                        help="Number of parallel environments per rollout")
    parser.add_argument("--dataset_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "wise_dataset/no_noise_demo_1_round"),
                        help="Root dir for dataset to fetch precomputed T5 embeddings")
    parser.add_argument("--eval_rounds", type=int, default=1)
    parser.add_argument("--save_videos", action="store_true")
    return parser.parse_args()


def aggregate_results(result_dir):
    """Compute final aggregated results for train and test splits."""
    for split in ["train", "test"]:
        pattern = os.path.join(result_dir, f"*{split}*")
        final_results = calculate_averages(pattern)
        if final_results:
            out_path = os.path.join(result_dir, f"final_results_{split}.json")
            with open(out_path, "w") as f:
                json.dump(final_results, f, indent=2)
            print(f"Final results saved to {out_path}")
        else:
            print(f"No results found for split '{split}'")


def process_obs_images(obs):
    cam_high = obs[image_1_key].cpu().numpy() if torch.is_tensor(obs[image_1_key]) else obs[image_1_key]
    wrist = obs[wrist_image_key].cpu().numpy() if torch.is_tensor(obs[wrist_image_key]) else obs[wrist_image_key]
    
    B = cam_high.shape[0]
    has_second = image_2_key in obs
    if has_second:
        cam_right = obs[image_2_key].cpu().numpy() if torch.is_tensor(obs[image_2_key]) else obs[image_2_key]
    else:
        cam_right = np.zeros_like(wrist)

    concat_images = []
    for i in range(B):
        img_h = cam_high[i]
        img_l = wrist[i]
        img_r = cam_right[i]

        top_h, target_w, _ = img_h.shape
        split_w = target_w // 2
        right_w = target_w - split_w

        # Resize bottoms match target_w together
        img_l_resized = cv2.resize(img_l, (split_w, img_l.shape[0]))
        img_r_resized = cv2.resize(img_r, (right_w, img_r.shape[0]))
        bottom = np.concatenate([img_l_resized, img_r_resized], axis=1)

        concat = np.concatenate([img_h, bottom], axis=0)
        concat_images.append(concat)

    return np.stack(concat_images, axis=0)


def main():
    args = parse_args()

    if args.aggregate_only:
        aggregate_results(args.result_dir)
        return

    os.makedirs(args.result_dir, exist_ok=True)

    # Initialize policy
    policy = StandaloneMotusPolicy(
        checkpoint_path=args.ckpt_path,
        wan_path=args.wan_path,
        vlm_path=args.vlm_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        execute_steps=20,
        stat_path=os.path.join(CURRENT_DIR, "utils", "stat.json")
    )

    splits = ["train", "test"] if args.split == "both" else [args.split]

    print(f"Evaluating subsets {args.start_subset} to {args.end_subset - 1} "
          f"on splits: {splits}")

    for split in splits:
        for i in range(args.start_subset, args.end_subset):
            cfg_name = f"config_{i}"
            subset_result_dir = os.path.join(args.result_dir, f"config_{i}_{split}")

            metrics_file = os.path.join(subset_result_dir, "episode_metrics.json")
            if os.path.exists(metrics_file):
                print(f"Skipping {cfg_name} ({split}) \u2014 already evaluated")
                continue

            if os.path.exists(subset_result_dir):
                shutil.rmtree(subset_result_dir)
            os.makedirs(subset_result_dir)

            # Look up T5 embedding for this task (assuming episode_000000.pt has it)
            # T5 embeddings are identical per task config.
            t5_path = os.path.join(
                args.dataset_root, f"{cfg_name}_{split}", 
                "lerobot_data", "t5_embedding", "episode_000000.pt"
            )
            if not os.path.exists(t5_path):
                # Fallback. It might not have test split if we are using train split generated T5
                t5_path = os.path.join(
                    args.dataset_root, f"{cfg_name}_train", 
                    "lerobot_data", "t5_embedding", "episode_000000.pt"
                )
            
            if os.path.exists(t5_path):
                policy.set_t5_embedding(t5_path)
                print(f"Loaded T5 embedding from {t5_path}")
            else:
                print(f"WARNING: No T5 embedding found at {t5_path}. Motus might fail to replan.")

            # Reset policy queues at the start of each new config evaluation
            policy.reset()

            # Build environment
            scene_cfg = dict(
                robot_init_qpos_noise=0.0,
                cube_size_noise=0.0,
                cfg_name=cfg_name,
                mode=split,
            )
            env_cfg = get_env_cfg(
                num_env=args.num_env,
                max_steps=MAX_EPISODE_STEP_WORKSPACE,
                obs_mode="rgb+segmentation",
                scene_cfg_to_overwrite=scene_cfg,
            )
            envs = build_endless_env(env_cfg, record_video=False, data_record_dir="test")

            def wrapped_policy(obs):
                # obs is a dictionary from rollout
                batched_images = process_obs_images(obs)
                state = obs[obs_state_key].cpu().numpy() if torch.is_tensor(obs[obs_state_key]) else obs[obs_state_key]

                # Run inference
                action_to_take = policy.act(batched_images, state)
                # Ensure action matches torch tensor expected by the environment
                action_to_take = torch.from_numpy(action_to_take).float().to("cuda" if torch.cuda.is_available() else "cpu")
                return action_to_take, action_to_take, dict(use_expert_action=0)

            print("\n" + "=" * 60)
            print(f"Starting Rollout for {cfg_name} ({split})")
            print("=" * 60)

            start_time = time.perf_counter()
            with torch.no_grad():
                performance = rollout(
                    envs,
                    wrapped_policy,
                    round_to_collect=args.eval_rounds,
                    demo_saving_dir=subset_result_dir,
                    debug_mode=True,
                    indices_to_save=[] if not args.save_videos else None,
                )
            elapsed = time.perf_counter() - start_time

            print("\n" + "=" * 60)
            print(f"Performance for {cfg_name} ({split}) \u2014 {elapsed:.1f}s")
            print("=" * 60)
            for key, v in performance.items():
                print(f"  {key}: {v}")

            envs.unwrapped.close()

    if args.start_subset == 0 and args.end_subset == 24:
        print("\nAll subsets evaluated. Running aggregation...")
        aggregate_results(args.result_dir)

    print("Done.")

if __name__ == "__main__":
    main()
