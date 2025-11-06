#!/bin/bash
#SBATCH --job-name=smolvla_resume
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=88
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --partition=hopper-prod
#SBATCH --output=/fsx/dana_aubakirova/vla/logs/smolvla_resume_%j.out
#SBATCH --error=/fsx/dana_aubakirova/vla/logs/smolvla_resume_%j.err
#SBATCH --exclusive

# Create logs directory if it doesn't exist
mkdir -p /fsx/dana_aubakirova/vla/logs

# Activate conda environment
source /fsx/dana_aubakirova/miniconda/etc/profile.d/conda.sh
conda activate vlab

# Add VLAb source to Python path
export PYTHONPATH="/fsx/dana_aubakirova/vla/VLAb/src:$PYTHONPATH"

# CUDA environment configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True,garbage_collection_threshold:0.8
export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_DEBUG=WARN
export CUDA_LAUNCH_BLOCKING=0
export ACCELERATE_USE_FSDP=false
export ACCELERATE_USE_DEEPSPEED=false
export HF_ACCELERATE_DEVICE_MAP=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Change to working directory
cd /fsx/dana_aubakirova/vla

# RESUME CONFIGURATION - Update these paths to your checkpoint
export RESUME_FROM_CHECKPOINT="/fsx/dana_aubakirova/vla/outputs/your_training_run"
export OUTPUT_DIR="$RESUME_FROM_CHECKPOINT"

# Dataset configuration - Use ALL datasets from all_datasets_relative.txt
export REPO_IDS=$(cat /fsx/dana_aubakirova/vla/VLAb/examples/all_datasets_relative.txt)
# For testing with a small subset, use:
#export REPO_IDS="AndrejOrsula/lerobot_double_ball_stacking_random,koenvanwijk/orange50-variation-2"

# Model configuration
export VLM_REPO_ID=HuggingFaceTB/SmolVLM2-500M-Video-Instruct
export STEPS=200000
export BATCH_SIZE=4  # Per GPU
export EVAL_FREQ=10000
export NUM_WORKERS=0  # Memory optimization
export SAVE_FREQ=10000

# Policy configuration
export POLICY=smolvla2
export USE_AMP=false
export OPTIMIZER_LR=5e-4
export PEFT_METHOD=lora
export LOAD_VLM_WEIGHTS=true
export MAX_ACTION_DIM=32
export MAX_STATE_DIM=32

# Dataset config
export USE_IMAGENET_STATS=false
export ENABLE_IMG_TRANSFORM=true
export MAX_NUM_IMAGES=2
export MAX_IMAGE_DIM=256
export TRAIN_ON_ALL_FEATURES=true
export FEATURES_VERSION=2

# Advanced optimizations
export FPS_MIN=30
export FPS_MAX=30
export GRADIENT_ACCUMULATION_STEPS=1
export PRECISION=no
export DROP_LAST=true

# VLM parameters
export VLM_LAYERS=16
export EXPERT_WIDTH_MULTIPLIER=0.75
export CAUSAL_ACTION_ATTENTION=true
export SELF_ATTN_EVERY_N_LAYERS=2
export ATTENTION_MODE=cross_attn
export LORA_R=32
export LORA_TARGET_MODULES=q_proj,v_proj
export PREFIX_LENGTH=0

# Learning rate schedule
export DECAY_LR=1e-6
export DECAY_STEPS=50000
export LR_VLM=1e-4

# Cache configuration
export HF_LEROBOT_HOME="/fsx/dana_aubakirova/vla"
export HF_HOME="/fsx/dana_aubakirova/vla/.cache/huggingface"
export HF_HUB_CACHE="/fsx/dana_aubakirova/vla/.cache/huggingface"
export TRANSFORMERS_CACHE="/fsx/dana_aubakirova/vla/.cache/huggingface"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

# Optional: Clean HuggingFace cache before training (set to true to enable)
# This can help resolve issues with corrupted or outdated cached datasets
CLEAN_CACHE=false
if [ "$CLEAN_CACHE" = true ]; then
    echo "🧹 Cleaning HuggingFace cache..."
    # Clean datasets cache (removes cached dataset files)
    rm -rf "$HF_HOME/datasets" 2>/dev/null || true
    # Clean hub cache (removes cached model/dataset downloads)
    rm -rf "$HF_HUB_CACHE/hub" 2>/dev/null || true
    # Clean transformers cache (removes cached model files)
    rm -rf "$TRANSFORMERS_CACHE" 2>/dev/null || true
    echo "✅ Cache cleaning completed"
    echo "   Note: Datasets will be re-downloaded on first use"
else
    echo "ℹ️  Cache cleaning disabled (set CLEAN_CACHE=true to enable)"
fi

# WandB configuration - RESUME MODE
export WANDB_PROJECT="smolvla2-training"
export WANDB_RUN_ID=""  # Set this to your existing run ID to resume
export WANDB_NOTES="VLAb training RESUMED from checkpoint"
export WANDB_MODE="online"
export WANDB_RESUME="must"  # Resume existing WandB run

# Print configuration info
echo "🚀 =============================================="
echo "🚀 VLAb RESUME TRAINING CONFIGURATION"
echo "🚀 =============================================="
echo "🔄 RESUMING FROM CHECKPOINT: $OUTPUT_DIR"
echo "📊 Datasets: ALL datasets from all_datasets_relative.txt"
echo "📁 Output directory: $OUTPUT_DIR"
echo "📊 Number of datasets: $(echo "$REPO_IDS" | tr ',' '\n' | wc -l)"
echo "🎯 Policy: $POLICY"
echo "🔧 Batch size per GPU: $BATCH_SIZE (GLOBAL: $((BATCH_SIZE * 8)))"
echo "📈 Training steps: $STEPS"
echo "💾 Save frequency: $SAVE_FREQ"
echo "🔬 Evaluation frequency: $EVAL_FREQ"
echo "⚡ AMP enabled: $USE_AMP"
echo "📚 Learning rate: $OPTIMIZER_LR"
echo "🎓 VLM Learning rate: $LR_VLM"
echo "📷 Max images: $MAX_NUM_IMAGES"
echo "🖼️  Image dimension: $MAX_IMAGE_DIM"
echo "👥 Data workers: $NUM_WORKERS"
echo "🧠 VLM layers: $VLM_LAYERS"
echo "🎯 LORA rank: $LORA_R"
echo "🖥️  GPUs: 8"
echo "📊 Wandb project: $WANDB_PROJECT"
echo "🚀 =============================================="

# Check GPU availability
echo "🖥️  GPU Information:"
nvidia-smi --list-gpus

# Run distributed training with resume
accelerate launch --config_file /fsx/dana_aubakirova/vla/VLAb/accelerate_configs/multi_gpu.yaml \
    VLAb/src/lerobot/scripts/train.py \
    --resume=true \
    --config_path="$RESUME_FROM_CHECKPOINT/checkpoints/last/pretrained_model/train_config.json" \
    --policy.type=$POLICY \
    --dataset.repo_id="$REPO_IDS" \
    --dataset.root="/fsx/dana_aubakirova/vla" \
    --dataset.use_imagenet_stats=$USE_IMAGENET_STATS \
    --dataset.image_transforms.enable=$ENABLE_IMG_TRANSFORM \
    --dataset.train_on_all_features=$TRAIN_ON_ALL_FEATURES \
    --dataset.features_version=$FEATURES_VERSION \
    --policy.max_action_dim=$MAX_ACTION_DIM \
    --policy.max_state_dim=$MAX_STATE_DIM \
    --output_dir=$OUTPUT_DIR \
    --batch_size=$BATCH_SIZE \
    --steps=$STEPS \
    --eval_freq=$EVAL_FREQ \
    --save_freq=$SAVE_FREQ \
    --policy.use_amp=$USE_AMP \
    --policy.optimizer_lr=$OPTIMIZER_LR \
    --policy.optimizer_lr_vlm=$LR_VLM \
    --policy.scheduler_decay_lr=$DECAY_LR \
    --policy.scheduler_decay_steps=$DECAY_STEPS \
    --policy.peft_method=$PEFT_METHOD \
    --policy.peft_config.r=$LORA_R \
    --policy.peft_config.target_modules=$LORA_TARGET_MODULES \
    --policy.load_vlm_weights=$LOAD_VLM_WEIGHTS \
    --policy.repo_id=$VLM_REPO_ID \
    --policy.push_to_hub=false \
    --dataset.max_num_images=$MAX_NUM_IMAGES \
    --dataset.max_image_dim=$MAX_IMAGE_DIM \
    --dataset.video_backend=pyav \
    --num_workers=$NUM_WORKERS \
    --wandb.enable=true \
    --wandb.project=$WANDB_PROJECT \
    --wandb.run_id=$WANDB_RUN_ID \
    --wandb.notes="$WANDB_NOTES" \
    --dataset.min_fps=$FPS_MIN \
    --dataset.max_fps=$FPS_MAX \
    --policy.num_vlm_layers=$VLM_LAYERS \
    --policy.expert_width_multiplier=$EXPERT_WIDTH_MULTIPLIER \
    --policy.causal_action_attention_mask=$CAUSAL_ACTION_ATTENTION \
    --policy.self_attn_every_n_layers=$SELF_ATTN_EVERY_N_LAYERS \
    --policy.attention_mode=$ATTENTION_MODE \
    --policy.prefix_length=$PREFIX_LENGTH

echo "✅ VLAb resume training completed! Check results in: $OUTPUT_DIR"
echo "📊 View training progress at: https://wandb.ai"
echo "🔄 RESUME TRAINING SUMMARY:"
echo "   • RESUMED from checkpoint: $OUTPUT_DIR/checkpoints/last"
echo "   • All model weights, optimizer state, and scheduler state restored"
echo "   • Training will continue to step $STEPS"
echo ""
echo "🚀 Key optimizations:"
echo "   • 8 GPUs with global batch size $((BATCH_SIZE * 8))"
echo "   • Memory-optimized data loading: $NUM_WORKERS workers"
echo "   • Stable training without mixed precision"
echo "   • Optimized NCCL settings for 8-GPU communication"
