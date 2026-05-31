"""Evaluate PI0.5 Policy on HSR MuJoCo Environment.

Loads a pretrained PI0.5 policy from HuggingFace and evaluates it in the
HSR MuJoCo simulation. Records episodes as MP4 video files.

Usage:
    python run.py --model_id paulprt/pi0.5-hsr --task "pick up the object"
"""

import os
import argparse
import numpy as np
import torch

from env import HSREnv
from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors
from peft import PeftModel


def load_policy(model_id: str, device: torch.device):
    """Load PI0.5 policy and its pre/post processors from HuggingFace.

    Args:
    - model_id: HuggingFace model identifier.
    - device: Torch device to load the model onto.

    Returns:
        tuple: (policy, preprocessor, postprocessor)
    """
    policy = PI05Policy.from_pretrained("lerobot/pi05_base").to(device).eval()
    policy = PeftModel.from_pretrained(policy, model_id)

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # Print input features so we can verify key names
    print(f"Policy input features: {list(policy.config.input_features.keys())}")
    print(f"Policy output features: {list(policy.config.output_features.keys())}")

    return policy, preprocess, postprocess


def build_observation(env, task: str):
    """Build the observation dict expected by the PI0.5 policy.

    Constructs a raw batch dict with:
    - observation.image.head: (C, H, W) float32 tensor
    - observation.image.hand: (C, H, W) float32 tensor
    - observation.state: (8,) float32 tensor
    - task: natural language instruction string

    Args:
    - env: HSREnv instance.
    - task: Task instruction string.

    Returns:
        dict: Raw observation batch (without batch dimension).
    """
    # Get images from both cameras: (H, W, 3) uint8 -> (3, H, W) float32 [0, 1]
    head_image = env.get_head_image()
    hand_image = env.get_hand_image()

    head_tensor = torch.from_numpy(head_image).permute(2, 0, 1).float() / 255.0
    hand_tensor = torch.from_numpy(hand_image).permute(2, 0, 1).float() / 255.0

    # Get state: (8,) float32
    state = env.get_state()
    state_tensor = torch.from_numpy(state).float()

    raw_batch = {
        "observation.image.head": head_tensor,
        "observation.image.hand": hand_tensor,
        "observation.state": state_tensor,
        "task": task,
    }

    return raw_batch


def run_evaluation(args):
    """Run the full evaluation loop.

    Args:
    - args: Parsed command-line arguments.
    """
    # Allow pytorch to use expandable memory segments
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Select device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load policy
    print(f"Loading policy from {args.model_id}...")
    policy, preprocess, postprocess = load_policy(args.model_id, device)
    print("Policy loaded successfully.")

    # Create environment
    env_args = {
        'logging': args.logging,
        'render': args.render,
    }
    env = HSREnv(env_args)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Episode loop
    for ep in range(args.num_episodes):
        print(f"\n=== Episode {ep} ===")
        env.reset()

        for step in range(args.num_steps):
            # Build observation
            raw_batch = build_observation(env, args.task)

            # Preprocess (adds batch dim, normalizes, moves to device)
            batch = preprocess(raw_batch)

            # Predict action
            with torch.inference_mode():
                action = policy.select_action(batch)

            # Postprocess (denormalize)
            action = postprocess(action)

            # Extract the 11-dim action and move to numpy
            action_np = action[0, :11].cpu().numpy().astype(np.float32)

            if step % 50 == 0:
                print(f"  Step {step}: action={action_np}")

            # Step environment
            env.step(action_np)

        # Save episode video
        env.save_episode_video(args.output_dir, ep, fps=30)
        print(f"Episode {ep} saved to {args.output_dir}/")

    env.close()
    print(f"\nEvaluation complete. {args.num_episodes} episodes saved to {args.output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PI0.5 policy on HSR MuJoCo environment"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="paulprt/pi0.5-hsr",
        help="HuggingFace model ID for the PI0.5 policy",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="pick up the object",
        help="Natural language task instruction for the policy",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes to run",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=300,
        help="Number of steps per episode (at 30Hz, 300 steps = 10 seconds)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/videos",
        help="Directory to save episode MP4 videos",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable MuJoCo viewer rendering",
    )
    parser.add_argument(
        "--logging",
        action="store_true",
        help="Enable action/state logging to file",
    )

    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()