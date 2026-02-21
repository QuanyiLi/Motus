#!/usr/bin/env python3
# Evaluate single-batch training loss for Motus

import os
import sys
import argparse
import logging
from pathlib import Path

# Set CUDA memory management environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

# Add project root and standalone directory to path
sys.path.append(str(Path(__file__).parent.parent))
standalone_dir = Path(__file__).parent.parent / "inference" / "standalone"
if str(standalone_dir) not in sys.path:
    sys.path.append(str(standalone_dir))

from models.motus import Motus, MotusConfig
from data.dataset import create_dataset, collate_fn
from standalone_inference import StandaloneMotusPolicy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> OmegaConf:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    config.common.action_chunk_size = config.common.num_video_frames * config.common.video_action_freq_ratio
    return config

def main():
    parser = argparse.ArgumentParser(description="Evaluate loss on a single batch")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, default="/home/quanyi/motus_original_wan/mp_rank_00_model_states.pt",
                        help="Path to the model states .pt file (e.g., mp_rank_00_model_states.pt)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"],
                        help="Which dataset split to pull the batch from")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    logger.info("Initializing StandaloneMotusPolicy...")
    # Instantiate StandaloneMotusPolicy to match exact inference loading structures
    policy = StandaloneMotusPolicy(
        checkpoint_path=args.checkpoint,
        wan_path=config.model.wan.checkpoint_path,
        vlm_path=config.model.vlm.checkpoint_path,
        device=str(device),
        state_dim=config.common.state_dim,
        action_dim=config.common.action_dim,
        video_height=config.common.video_height,
        video_width=config.common.video_width,
        execute_steps=20,
        stat_path=None # Optimization: Skip normalization config checks for loss evaluation
    )
    model = policy.model
    
    logger.info(f"Creating {args.split} dataloader...")
    dataset = create_dataset(config, val=(args.split == "val"))
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True, # Shuffle to get a random batch
        num_workers=config.system.num_workers,
        pin_memory=config.system.pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )
    
    logger.info("Fetching a single batch...")
    data_iter = iter(dataloader)
    batch = None
    while batch is None:
        try:
            batch = next(data_iter)
        except StopIteration:
            logger.error("Dataloader ran out of batches immediately!")
            return
            
    logger.info("Moving batch to device...")
    first_frame = batch['first_frame'].to(device, dtype=dtype)          # [B, C, H, W]
    video_frames = batch['video_frames'].to(device, dtype=dtype)        # [B, num_video_frames, C, H, W]
    
    language_embeddings = batch['language_embedding']
    if language_embeddings is not None:
        language_embeddings = language_embeddings.to(device, dtype=dtype)
        
    state = batch.get('initial_state', None)
    if state is not None:
        state = state.to(device, dtype=dtype)      # [B, state_dim]
        
    actions = batch['action_sequence'].to(device, dtype=dtype)  # [B, action_chunk_size, action_dim]
    
    vlm_inputs = batch['vlm_inputs']
    if vlm_inputs is not None:
        vlm_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in vlm_inputs.items()}
                     
    logger.info("Running forward pass (calculating loss) via policy.model...")
    with torch.no_grad():
        loss_dict = model.training_step(
            first_frame=first_frame,
            video_frames=video_frames,
            state=state,
            actions=actions,
            language_embeddings=language_embeddings,
            vlm_inputs=vlm_inputs,
            return_dict=True
        )
        
    logger.info("=" * 60)
    logger.info("LOSS CALCULATION RESULTS")
    logger.info("=" * 60)
    for k, v in loss_dict.items():
        val = v.item() if torch.is_tensor(v) else v
        logger.info(f"{k}: {val:.6f}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
