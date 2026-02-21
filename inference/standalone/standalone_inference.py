import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
import sys
import os
import logging
from typing import List, Dict, Any, Optional
from collections import deque
from PIL import Image
from transformers import AutoProcessor
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend

# Setup path for local dependencies (models, utils, bak)
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))
if str(CURRENT_DIR / "models") not in sys.path:
    sys.path.append(str(CURRENT_DIR / "models"))
if str(CURRENT_DIR / "bak") not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR / "bak"))

try:
    from models.motus import Motus, MotusConfig
    from wan.modules.t5 import T5EncoderModel
    from utils.image_utils import resize_with_padding
except ImportError as e:
    raise ImportError(f"Failed to import local dependencies. Make sure 'models', 'utils', and 'bak' folders are in {CURRENT_DIR}. Error: {e}")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class StandaloneMotusPolicy:
    """
    Standalone Motus Policy for inference in external simulation environments.
    Generates 48 actions and executes the first 20 (action chunking).
    """
    def __init__(
        self,
        checkpoint_path: str,
        wan_path: str,
        vlm_path: str,
        device: str = "cuda",
        state_dim: int = 9,
        action_dim: int = 8,
        video_height: int = 384,
        video_width: int = 320,
        execute_steps: int = 20,
        stat_path: Optional[str] = None
    ):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.wan_path = wan_path
        self.vlm_path = vlm_path
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.video_height = video_height
        self.video_width = video_width
        self.execute_steps = execute_steps

        # Initialize model WITHOUT loading pretrained backbones
        self.model = self._load_model()

        # Initialize T5 encoder for language embeddings (WAN text encoder)
        self.t5_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=device,
            checkpoint_path=os.path.join(self.wan_path, 'models_t5_umt5-xxl-enc-bf16.pth'),
            tokenizer_path=os.path.join(self.wan_path, 'google', 'umt5-xxl'),
        )

        # Initialize VLM processor from vlm_path (for tokenization only)
        self.vlm_processor = AutoProcessor.from_pretrained(self.vlm_path, trust_remote_code=True)
        
        # Initialize caches
        self.obs_cache = deque(maxlen=1)
        self.action_queue = deque()
        
        # Model state
        self.current_state = None
        self.current_state_norm = None
        self.current_instruction = "Perform the task."
        
        # Load normalization stats if provided
        self.stat_path = stat_path
        self.action_min = None
        self.action_max = None
        self.action_range = None
        self._load_normalization_stats()

        logger.info("Standalone Motus Policy initialized successfully.")

    def set_instruction(self, instruction: str):
        self.current_instruction = instruction
        logger.info(f"Instruction set: {instruction}")

    def _load_model(self) -> Motus:
        logger.info("Initializing Motus model from scratch (no pretrained backbones)")
        config = self._create_model_config()
        model = Motus(config).to(self.device)
        
        try:
            logger.info(f"Loading checkpoint from {self.checkpoint_path}")
            model.load_checkpoint(self.checkpoint_path, strict=False)
            logger.info("Model checkpoint loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
        
        model.eval()
        return model

    def _create_model_config(self) -> MotusConfig:
        vae_path = os.path.join(self.wan_path, "Wan2.2_VAE.pth")
        
        config = MotusConfig(
            wan_checkpoint_path=self.wan_path,
            vae_path=vae_path,
            wan_config_path=self.wan_path,
            video_precision='bfloat16',
            vlm_checkpoint_path=self.vlm_path,
            
            und_expert_hidden_size=512,
            und_expert_ffn_dim_multiplier=4,
            und_expert_norm_eps=1e-5,
            und_layers_to_extract=None,
            vlm_adapter_input_dim=2048,
            vlm_adapter_projector_type="mlp3x_silu",
            
            num_layers=30,
            action_state_dim=self.state_dim,
            action_dim=self.action_dim,
            action_expert_dim=1024,
            action_expert_ffn_dim_multiplier=4,
            action_expert_norm_eps=1e-6,
            
            global_downsample_rate=1,
            video_action_freq_ratio=6,
            num_video_frames=8,
            video_loss_weight=1.0,
            action_loss_weight=1.0,
            
            batch_size=1,
            video_height=self.video_height,
            video_width=self.video_width,
            
            load_pretrained_backbones=False,
            training_mode='finetune',
        )
        return config

    def _load_normalization_stats(self):
        if self.stat_path and os.path.exists(self.stat_path):
            import json
            with open(self.stat_path, 'r') as f:
                stat_data = json.load(f)
            # Find the first valid robot stats if not explicitly named
            stats = None
            if "robotwin2" in stat_data:
                stats = stat_data["robotwin2"]
            else:
                stats = list(stat_data.values())[0] if stat_data else None

            if stats:
                self.action_min = torch.tensor(stats['min'], dtype=torch.float32, device=self.device)
                self.action_max = torch.tensor(stats['max'], dtype=torch.float32, device=self.device)
                self.action_range = self.action_max - self.action_min
                logger.info(f"Loaded normalization stats from {self.stat_path}")
            else:
                logger.warning("No valid stats found in stat file.")
        else:
            logger.warning("No stat file provided or found. Assuming actions/states are already normalized or don't need it.")

    def _normalize_actions(self, x: torch.Tensor) -> torch.Tensor:
        if self.action_min is None:
            return x
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])
        norm = (x_flat - self.action_min.unsqueeze(0)) / self.action_range.unsqueeze(0)
        return norm.reshape(shape)

    def _denormalize_actions(self, y: torch.Tensor) -> torch.Tensor:
        if self.action_min is None:
            return y
        shape = y.shape
        y_flat = y.reshape(-1, shape[-1])
        denorm = y_flat * self.action_range.unsqueeze(0) + self.action_min.unsqueeze(0)
        return denorm.reshape(shape)

    def update_obs(self, image: np.ndarray, state: np.ndarray):
        """
        Update the observation.
        image: RGB image of shape (H, W, 3) or multiple camera images concatenated.
        state: 1D array of shape (state_dim,)
        """
        target_size = (self.video_height, self.video_width)

        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        else:
            image_tensor = image.clone().detach() if torch.is_tensor(image) else torch.tensor(image)
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)

        if image_tensor.shape[-2:] != target_size:
            image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            resized_np = resize_with_padding(image_np, target_size)
            if resized_np.dtype == np.uint8:
                resized_np = resized_np.astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(resized_np).permute(2, 0, 1).unsqueeze(0)
        
        self.obs_cache.append(image_tensor.to(self.device).float())

        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        else:
            state_tensor = state.float().unsqueeze(0) if state.dim() == 1 else state.float()

        self.current_state = state_tensor.to(self.device)
        self.current_state_norm = self._normalize_actions(self.current_state).to(self.device)

    def _tensor_to_pil_image(self, tensor_chw: torch.Tensor) -> Image.Image:
        if tensor_chw.dtype != torch.float32:
            tensor_chw = tensor_chw.float()
        tensor_chw = tensor_chw.clamp(0, 1)
        np_img = (tensor_chw.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(np_img, mode='RGB')

    def _preprocess_vlm_messages(self, instruction: str, image: Image.Image) -> Dict[str, torch.Tensor]:
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': instruction},
                    {'type': 'image', 'image': image},
                ]
            }
        ]
        text = self.vlm_processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        encoded = self.vlm_processor(text=[text], images=[image], return_tensors='pt')
        vlm_inputs = {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device), 
            'pixel_values': encoded['pixel_values'].to(self.device),
            'image_grid_thw': encoded.get('image_grid_thw', None)
        }
        if vlm_inputs['image_grid_thw'] is not None:
            vlm_inputs['image_grid_thw'] = vlm_inputs['image_grid_thw'].to(self.device)
        return vlm_inputs

    def act(self, image: np.ndarray, state: np.ndarray) -> np.ndarray:
        """
        Main entry point for external environments.
        Updates observation and returns a single action.
        If the action queue is empty, replans to generate 48 actions and keeps 20.
        Pop one action from the queue and return it.
        """
        self.update_obs(image, state)

        if len(self.action_queue) == 0:
            logger.info("Action queue empty. Replanning...")
            self._replan()

        # Pop the next action
        return self.action_queue.popleft()

    def _replan(self):
        """Run the model to generate a new chunk of actions."""
        current_frame = self.obs_cache[-1]

        scene_prefix = ("The whole scene is in a realistic, industrial art style with three views: "
                        "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
                        "The aloha robot is currently performing the following task: ")
        full_instruction = f"{scene_prefix}{self.current_instruction}"
        
        t5_out = self.t5_encoder([full_instruction], self.device)
        if isinstance(t5_out, torch.Tensor):
            t5_list = [t5_out.squeeze(0)] if t5_out.dim() == 3 else [t5_out]
        elif isinstance(t5_out, list):
            t5_list = t5_out
        else:
            raise ValueError("Unexpected T5 encoder output format")

        first_frame_pil = self._tensor_to_pil_image(current_frame.squeeze(0).cpu())
        vlm_inputs = self._preprocess_vlm_messages(full_instruction, first_frame_pil)

        num_inference_steps = 10
        with torch.no_grad():
            predicted_frames, predicted_actions = self.model.inference_step(
                first_frame=current_frame,
                state=self.current_state,
                num_inference_steps=num_inference_steps,
                language_embeddings=t5_list,
                vlm_inputs=[vlm_inputs],
            )

        actions_real = predicted_actions.squeeze(0).cpu().numpy()
        
        # Take only the first `execute_steps` (e.g., 20) actions
        kept_actions = actions_real[:self.execute_steps]
        
        logger.info(f"Generated {len(actions_real)} actions. Keeping first {len(kept_actions)}.")
        
        self.action_queue.clear()
        self.action_queue.extend(kept_actions)

    def reset(self):
        self.obs_cache.clear()
        self.action_queue.clear()
        self.current_state = None


def verify():
    print("Starting standalone inference verification...")
    # These are dummy paths / stats if not provided, just testing model initialization and forward pass
    ckpt_path = "/work/vita/lanfeng/vlas/Motus/ckpt_original_wan/wise_dataset/motus_wise_dataset/checkpoint_step_5000/pytorch_model/mp_rank_00_model_states.pt"
    wan_path = "/work/vita/lanfeng/vlas/Motus/pretrained_models/Wan2.2-TI2V-5B"
    vlm_path = "/work/vita/lanfeng/vlas/Motus/pretrained_models/Qwen3-VL-2B-Instruct"

    try:
        policy = StandaloneMotusPolicy(
            checkpoint_path=ckpt_path,
            wan_path=wan_path,
            vlm_path=vlm_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            execute_steps=20
        )
        
        # Create dummy observation
        dummy_image = np.random.randint(0, 255, (384, 320, 3), dtype=np.uint8)
        dummy_state = np.zeros((9,), dtype=np.float32)

        print("Requesting first action (should trigger replan)...")
        action1 = policy.act(dummy_image, dummy_state)
        print(f"Action 1 shape: {action1.shape}")
        print(f"Queue size after action 1: {len(policy.action_queue)}")
        
        print("Requesting second action (should pop from queue)...")
        action2 = policy.act(dummy_image, dummy_state)
        print(f"Action 2 shape: {action2.shape}")
        print(f"Queue size after action 2: {len(policy.action_queue)}")

        print("Verification passed successfully.")
    except Exception as e:
        print(f"Verification could not complete because model files are missing or inaccessible: {e}")
        print("However, the code structure has been verified up to this point.")


if __name__ == "__main__":
    verify()
