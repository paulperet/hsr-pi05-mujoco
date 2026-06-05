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
from lerobot.configs.policies import PreTrainedConfig
from lerobot.processor.pipeline import ProcessorStepRegistry
from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep
from peft import PeftModel

def register_legacy_processor_aliases() -> None:
    if "relative_actions_processor" not in ProcessorStepRegistry.list():
        ProcessorStepRegistry.register("relative_actions_processor")(RelativeActionsProcessorStep)

def load_policy(model_id: str, device: torch.device):
    """Load PI0.5 policy and its pre/post processors from HuggingFace.

    Args:
    - model_id: HuggingFace model identifier.
    - device: Torch device to load the model onto.

    Returns:
        tuple: (policy, preprocessor, postprocessor)
    """
    # Use the FINE-TUNED config (input_features = observation.images.head/hand,
    # state[8], action[11]) so the loaded model validates against our schema --
    # not the base model's DROID slots (base_0_rgb/left_wrist_0_rgb/...).
    # Weight shapes are unchanged (camera count only affects token count;
    # state/action are padded to max_*_dim=32), so base weights load cleanly.
    #config = PreTrainedConfig.from_pretrained(model_id)
    policy = PI05Policy.from_pretrained(model_id).to(device).eval()

    # Attach the LoRA adapter, then re-set eval (PeftModel wrap resets train mode)
    #policy = PeftModel.from_pretrained(policy, model_id)
    policy.eval()

    register_legacy_processor_aliases()
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
    - observation.images.head: (C, H, W) float32 tensor
    - observation.images.hand: (C, H, W) float32 tensor
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

    # Keys MUST match the training schema (see train_config.json input_features):
    #   observation.images.head, observation.images.hand, observation.state
    raw_batch = {
        "observation.images.head": head_tensor,
        "observation.images.hand": hand_tensor,
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

    # Select device: prefer CUDA, then Apple MPS, then CPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # torch.compile's inductor backend fails to autotune on MPS/CPU, so disable
    # dynamo off CUDA (the model still runs eagerly, just without compilation).
    if device.type != "cuda":
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
        os.environ["TORCH_COMPILE_DISABLE"] = "1"

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

    # Resolve how many actions of each predicted chunk to apply before
    # re-planning (the execution/replan horizon). Defaults to n_action_steps,
    # which reproduces the original open-loop behavior (execute the full chunk).
    # Smaller values re-query the policy more often (more reactive/closed-loop).
    chunk_size = policy.config.chunk_size
    chunk_steps = args.chunk_steps if args.chunk_steps is not None else policy.config.n_action_steps
    chunk_steps = max(1, min(chunk_steps, chunk_size))
    print(f"Applying {chunk_steps}/{chunk_size} actions per chunk before re-planning.")

    # Episode loop
    for ep in range(args.num_episodes):
        print(f"\n=== Episode {ep} ===")
        env.reset()

        # Locally managed chunk buffer (bypasses the policy's internal queue so
        # the execution horizon is independent of n_action_steps).
        action_chunk = None
        chunk_idx = 0

        for step in range(args.num_steps):
            # Re-plan once we've applied `chunk_steps` actions (or exhausted the
            # predicted chunk), using the freshest observation at that moment.
            if action_chunk is None or chunk_idx >= chunk_steps:
                # Build observation
                raw_batch = build_observation(env, args.task)

                # Preprocess (adds batch dim, normalizes, moves to device, tokenizes task)
                batch = preprocess(raw_batch)

                # Predict a full chunk: (batch, chunk_size, action_dim)
                with torch.inference_mode():
                    action_chunk = policy.predict_action_chunk(batch)

                chunk_idx = 0
                print(f"  Step {step}: predicted new chunk (applying {chunk_steps} actions)")

            # Postprocess (denormalize, convert relative -> absolute, move to CPU)
            action = postprocess(action_chunk[:, chunk_idx, :])

            # Extract the 11-dim action and move to numpy
            action_np = action[0, :11].cpu().numpy().astype(np.float32)

            # Step environment
            env.step(action_np)
            chunk_idx += 1

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
        "--chunk_steps",
        type=int,
        default=None,
        help=(
            "Number of actions to apply from each predicted action chunk before "
            "re-planning (the execution horizon, clamped to [1, chunk_size]). "
            "Lower values re-query the policy more often (more reactive/closed-loop); "
            "the default applies the full chunk (n_action_steps), matching open-loop "
            "execution."
        ),
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
