
<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <h1>Motus: A Unified Latent Action World Model</h1>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://motus-robotics.github.io/motus"><img alt="Homepage"
    src="https://img.shields.io/badge/Motus-Homepage-4287f5?logo=readme&logoColor=white"/></a>
  <a href="https://huggingface.co/motus-robotics"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-motus--robotics-ffc107?color=ffc107&logoColor=white"/></a>
  <a href="https://arxiv.org/abs/2512.13030"><img alt="arXiv"
    src="https://img.shields.io/badge/arXiv-2512.13030-b31b1b?logo=arxiv&logoColor=white"/></a>
  <br>
  <a href="https://motus-robotics.github.io/assets/motus/png/feishu.jpg"><img alt="Feishu"
    src="https://img.shields.io/badge/Feishu-Motus-blue?logo=lark&logoColor=white"/></a>
  <a href="https://motus-robotics.github.io/assets/motus/png/wechat.jpg"><img alt="WeChat"
    src="https://img.shields.io/badge/WeChat-Motus-green?logo=wechat&logoColor=white"/></a>
  <a href="LICENSE"><img alt="License"
    src="https://img.shields.io/badge/License-Apache--2.0-f5de53?logo=apache&color=f5de53"/></a>
</div>

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Updates](#updates)
- [Requirements](#requirements)
- [Installation](#installation)
- [Model Checkpoints](#model-checkpoints)
- [Data Format](#data-format)
  - [1. RoboTwin 2.0 (Simulation)](#1-robotwin-20-simulation)
  - [2. Real-World Robot Data (AC-One, Aloha-Agilex-2)](#2-real-world-robot-data-ac-one-aloha-agilex-2)
  - [3. Latent Action Pretraining (Stage 2)](#3-latent-action-pretraining-stage-2)
- [Running Inference](#running-inference)
  - [1. RoboTwin 2.0 Simulation](#1-robotwin-20-simulation-1)
  - [2. Real-World Inference (No Environment)](#2-real-world-inference-no-environment)
- [Training](#training)
  - [1. Fine-Tuning from Pretrained Checkpoint (Stage 3)](#1-fine-tuning-from-pretrained-checkpoint-stage-3)
  - [2. Resume Training](#2-resume-training)
  - [3. Training from Scratch](#3-training-from-scratch)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## Overview

**Motus** is a **unified latent action world model** that leverages existing pretrained models and rich, sharable motion information. Motus introduces a **Mixture-of-Transformers (MoT)** architecture to integrate three experts (understanding, action, and video generation) and adopts a **UniDiffuser-style scheduler** to enable flexible switching between different modeling modes (World Models, Vision-Language-Action Models, Inverse Dynamics Models, Video Generation Models, and Video-Action Joint Prediction Models). Motus further leverages **optical flow** to learn **latent actions** and adopts a **three-phase training pipeline** and **six-layer data pyramid**, thereby extracting pixel-level "delta action" and enabling large-scale action pretraining.

| Component | Base Model | Parameters |
|-----------|------------|------------|
| **VGM (Video Generation Model)** | [Wan2.2-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) | ~5.00B |
| **VLM (Vision-Language Model)** | [Qwen3-VL-2B](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | ~2.13B |
| **Action Expert** | - | ~641.5M |
| **Understanding Expert** | - | ~253.5M |
| **Total** | - | **~8B** |

**Key Results (RoboTwin 2.0 Simulation.** With 50 clean and 500 randomized data entries per task, we merge the data from all 50 tasks for multi-task training.):
- **87.02%** average success rate (+15% over X-VLA, +45% over π₀.₅)

## Updates

- [2025-12] Initial release of Motus with pretrained checkpoints and training code.

## Requirements

| Mode | VRAM | Recommended GPU |
|------|------|-----------------|
| Inference (with pre-encoded T5) | ~ 24 GB | RTX 5090 |
| Inference (without pre-encoded T5) | ~ 41 GB | A100 (40GB) / A100 (80GB) / H100 / B200 |
| Training | > 80 GB | A100 (80GB) / H100 / B200 |

## Installation

```bash
# Clone the repository
git clone https://github.com/thu-ml/Motus.git
cd Motus

# Create conda environment
conda create -n motus python=3.10 -y
conda activate motus

# Install other dependencies
pip install -r requirements.txt

pip install flash-attn --no-build-isolation
```

## Model Checkpoints

We provide multiple checkpoints for different use cases:

| Model | Use Case | Description | Checkpoint Path |
|-------|----------|-------------|-----------------|
| **Motus_Wan2_2_5B_pretrain** | Pretrain / VGM Backbone | Stage 1 VGM pretrained checkpoint | [`motus-robotics/Motus_Wan2_2_5B_pretrain`](https://huggingface.co/motus-robotics/Motus_Wan2_2_5B_pretrain) |
| **Motus** | Fine-Tuning | Stage 2 latent action pretrained checkpoint | [`motus-robotics/Motus`](https://huggingface.co/motus-robotics/Motus) |
| **Motus_robotwin2** | Inference / Fine-Tuning | Stage 3 RoboTwin2 fine-tuned checkpoint | [`motus-robotics/Motus_robotwin2`](https://huggingface.co/motus-robotics/Motus_robotwin2) |

**Download checkpoints:**
```bash
# Create pretrained models directory
mkdir -p pretrained_models

# Download Motus checkpoints
huggingface-cli download motus-robotics/Motus_Wan2_2_5B_pretrain --local-dir ./pretrained_models/Motus_Wan2_2_5B_pretrain
huggingface-cli download motus-robotics/Motus --local-dir ./pretrained_models/Motus
huggingface-cli download motus-robotics/Motus_robotwin2 --local-dir ./pretrained_models/Motus_robotwin2

# Download foundation models
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --local-dir ./pretrained_models/Qwen3-VL-2B-Instruct
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./pretrained_models/Wan2.2-TI2V-5B
```

**Update config paths** in your embodiment-specific config file (e.g., `configs/robotwin.yaml`, `configs/ac_one.yaml`, or other embodiment configs):
```yaml
model:
  wan:
    checkpoint_path: "./pretrained_models/Motus_Wan2_2_5B_pretrain"
    config_path: "./pretrained_models/Motus_Wan2_2_5B_pretrain"
    vae_path: "./pretrained_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
  vlm:
    checkpoint_path: "./pretrained_models/Qwen3-VL-2B-Instruct"
    config_path: "./pretrained_models/Qwen3-VL-2B-Instruct"
```

## Data Format

Motus supports three types of datasets. Each dataset type has its own directory structure:

### 1. RoboTwin 2.0 (Simulation)

```
/path/to/robotwin2/
├── clean/
│   ├── task_name/
│   │   ├── qpos/           # Robot joint positions (.pt)
│   │   │   ├── 0.pt
│   │   │   └── ...
│   │   ├── videos/         # MP4 video files
│   │   │   ├── 0.mp4
│   │   │   └── ...
│   │   └── umt5_wan/       # Pre-encoded T5 language embeddings (.pt)
│   │       ├── 0.pt
│   │       └── ...
│   └── ...
└── randomized/
    └── ... (same structure)
```

### 2. Real-World Robot Data (AC-One, Aloha-Agilex-2)

```
/path/to/ac_one/
├── task_category/
│   ├── task_variant/
│   │   ├── videos/
│   │   │   ├── 0.mp4
│   │   │   └── ...
│   │   ├── qpos/           # Robot joint positions (.pt)
│   │   │   ├── 0.pt
│   │   │   └── ...
│   │   └── instructions/   # Language instructions
│   │       ├── task.txt    # Text instruction
│   │       └── task.pt     # Pre-encoded T5 embedding
│   └── ...
└── ...
```

### 3. Latent Action Pretraining (Stage 2)

```
/path/to/latent_action_data/
├── videos/                 # MP4 video files
│   ├── episode_0.mp4
│   └── ...
├── umt5_wan/              # Pre-encoded T5 language embeddings
│   ├── episode_0.pt
│   └── ...
└── latent_action_dim14/   # Latent action labels (from optical flow)
    ├── episode_0.pt
    └── ...
```

**Configure dataset in YAML:**
```yaml
dataset:
  type: robotwin           # Options: robotwin, ac_one, aloha_agilex_2, latent_action
  dataset_dir: /path/to/dataset
  data_mode: both          # For robotwin: clean, randomized, or both
  task_mode: multi         # single or multi
```

## Running Inference

### 1. RoboTwin 2.0 Simulation

For evaluation on [RoboTwin 2.0](https://robotwin-platform.github.io/) benchmark:

```bash
cd inference/robotwin/Motus

# Single task evaluation
bash eval.sh <task_name>

# Multi-task batch evaluation
bash auto_eval.sh
```

### 2. Real-World Inference (No Environment)

We provide a minimal inference script that runs Motus on a single image without any robot environment:

**With pre-encoded T5 embeddings (recommended, ~24GB VRAM):**
```bash
# Step 1: Encode instruction to T5 embeddings (do this once)
python inference/real_world/Motus/encode_t5_instruction.py \
  --instruction "pick up the cube and place it on the right" \
  --output t5_embed.pt \
  --wan_path /path/to/pretrained_models

# Step 2: Run inference with pre-encoded embeddings
python inference/real_world/Motus/inference_example.py \
  --model_config inference/real_world/Motus/utils/robotwin.yml \
  --ckpt_dir ./pretrained_models/Motus_robotwin2 \
  --wan_path /path/to/pretrained_models \
  --image /path/to/input_frame.png \
  --instruction "pick up the cube and place it on the right" \
  --t5_embeds t5_embed.pt \
  --output result.png
```

**Without pre-encoded T5 (encode on-the-fly, ~41GB VRAM):**
```bash
python inference/real_world/Motus/inference_example.py \
  --model_config inference/real_world/Motus/utils/robotwin.yml \
  --ckpt_dir ./pretrained_models/Motus_robotwin2 \
  --wan_path /path/to/pretrained_models \
  --image /path/to/input_frame.png \
  --instruction "pick up the cube and place it on the right" \
  --use_t5 \
  --output result.png
```

**Output:**
- `result.png`: Grid of condition frame + predicted future frames
- Console: Predicted action chunk with shape `(action_chunk_size, action_dim)`

## Training

Motus follows a **three-stage training pipeline**:

| Stage | Data | Training |
|-------|------|----------|
| **Pretrained Foundation Models** | Level 1: Web Data | VGM and VLM |
| **Stage 1 (VGM Training)** | Level 2: Egocentric Human Videos<br>Level 3: Synthetic Data<br>Level 5: Multi-Robot Task Trajectory | Only VGM |
| **Stage 2 (Motus Pretraining)** | Level 2: Egocentric Human Videos<br>Level 3: Synthetic Data<br>Level 4: Task-agnostic Data<br>Level 5: Multi-Robot Task Trajectory | Motus (all 3 experts, with **latent actions**) |
| **Stage 3 (Motus SFT)** | Level 6: Target-Robot Task Trajectory | Motus (all 3 experts, with actions) |

The six-layer data pyramid is shown in the figure here:

<img width="615" height="455" alt="image" src="https://github.com/user-attachments/assets/b1389887-2f6b-4e82-87f9-08f0525301b5" />

### 1. Fine-Tuning from Pretrained Checkpoint (Stage 3)

To fine-tune Motus on your own robot data:

**Step 1:** Set the pretrain checkpoint path in your config (e.g., `configs/robotwin.yaml`):
```yaml
finetune:
  checkpoint_path: ./pretrained_models/Motus  # Stage 2 pretrained checkpoint
```

**Step 2:** Run training using one of the following methods:

**Option A: Using shell script with tmux (recommended for long-running jobs)**

SSH into your machine and start a tmux session for background training:
```bash
# Start a new tmux session
tmux new -s motus_train

# Set environment variables and run training
export CONFIG_FILE="configs/robotwin.yaml"
export RUN_NAME="motus_finetune"
export GPU_IDS="0,1,2,3,4,5,6,7"
bash scripts/run_local.sh

# Detach from tmux: Ctrl+B, then D
# Re-attach later: tmux attach -t motus_train
```

**Option B: Using SLURM**
```bash
# Single node with SLURM
sbatch scripts/slurm_single_node.sh

# Multi-node with SLURM
sbatch scripts/slurm_multi_node.sh
```

**Option C: Direct command (multi-GPU with torchrun + DeepSpeed)**
```bash
torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  --master_port=29500 \
  train/train.py \
  --deepspeed configs/zero1.json \
  --config configs/robotwin.yaml \
  --run_name motus_finetune \
  --report_to tensorboard
```

To provide a better understanding, here is an explanation of key parameters in [`scripts/run_local.sh`](scripts/run_local.sh):

```bash
# Environment setup
export CONFIG_FILE="configs/robotwin.yaml"   # Config file path
export RUN_NAME="robotwin"                   # Experiment name for logging
export GPU_IDS="0,1,2,3,4,5,6,7"            # GPUs to use (comma-separated)

# The script automatically:
# - Validates GPU availability
# - Sets NCCL environment variables for multi-GPU
# - Chooses single-GPU or DeepSpeed based on GPU count
# - Logs to logs/local_${RUN_NAME}_*.log
```

### 2. Resume Training

To resume from an interrupted checkpoint:

**Step 1:** Set the resume checkpoint path in your config:
```yaml
resume:
  checkpoint_path: ./checkpoints/motus_finetune/checkpoint_step_10000
```

**Step 2:** Run training (same commands as above):
```bash
bash scripts/run_local.sh
```

> **Note:** When resuming or fine-tuning, WAN and VLM pretrained weights are **not reloaded** (only VAE is needed). This prevents overwriting fine-tuned weights.

### 3. Training from Scratch

To train Motus from scratch (load WAN + VLM pretrained weights):

**Step 1:** Ensure `resume.checkpoint_path` and `finetune.checkpoint_path` are both `null` in your config:
```yaml
resume:
  checkpoint_path: null
finetune:
  checkpoint_path: null
```

**Step 2:** Run training:
```bash
bash scripts/run_local.sh
```

This will load:
- Wan2.2-5B pretrained weights from `model.wan.checkpoint_path`
- Qwen3-VL pretrained weights from `model.vlm.checkpoint_path`

## Troubleshooting

We will collect common issues and their solutions here. If you encounter an issue, please check here first. If you can't find a solution, please file an issue on the repo (see [here](CONTRIBUTING.md) for guidelines).

| Issue | Likely Cause | Solution |
|-------|--------------|----------|
| OOM during inference (~41GB) | T5 encoder loaded at runtime | Use pre-encoded T5 embeddings (`--t5_embeds`) to reduce to ~24GB |
| Poor action predictions | Checkpoint mismatch | Ensure using correct config for your checkpoint |
| Slow training | No flash-attn | Install flash-attention: `pip install flash-attn --no-build-isolation` |
| WAN/VLM weights not loading | Resume/finetune mode | Set both `resume.checkpoint_path` and `finetune.checkpoint_path` to `null` |
| NCCL timeout | Network issues | Check NCCL environment variables in scripts |

## Citation
If you find our work helpful, please cite us:

```bibtex
@misc{bi2025motusunifiedlatentaction,
      title={Motus: A Unified Latent Action World Model}, 
      author={Hongzhe Bi and Hengkai Tan and Shenghao Xie and Zeyuan Wang and Shuhe Huang and Haitian Liu and Ruowen Zhao and Yao Feng and Chendong Xiang and Yinze Rong and Hongyan Zhao and Hanyu Liu and Zhizhong Su and Lei Ma and Hang Su and Jun Zhu},
      year={2025},
      eprint={2512.13030},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.13030}, 
}
```

Thank you!

