set -x

export NCCL_DEBUG=WARN 
export WANDB_API_KEY=''
# 极端内存管理
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:8
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

PROJECT_NAME='SimpleVLA-RL'
EXPERIMENT_NAME='test1_lr5e-6_bs8_node1_lora_extreme_low' 
SFT_MODEL_PATH="/home/hhjiang/Documents/SimpleVLA-RL/simplevla-models/Haozhan72/Openvla-oft-SFT-libero10-traj1"
CKPT_PATH="/home/hhjiang/Documents/SimpleVLA-RL/ckpt"
DATASET_NAME="libero_10"
VLA_NAME="openvla-oft"
NUM_GPUS=1
NUM_NODES=1 
ALIGN_PATH="/home/hhjiang/Documents/SimpleVLA-RL/align.json"

echo "Starting training with EXTREME LOW LoRA configuration..."
echo "Using rank=2 LoRA for maximum memory savings"

# 清理GPU内存
python3 -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"

HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    data.task_suite_name=$DATASET_NAME \
    data.num_trials_per_task=10 \
    data.n_samples=2 \
    data.filter_accuracy=True \
    data.accuracy_lower_bound=0.1 \
    data.accuracy_upper_bound=0.9 \
    data.oversample_factor=1 \
    data.train_batch_size=8 \
    data.val_batch_size=16 \
    data.max_prompt_length=64 \
    data.max_response_length=32 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.model.vla=$VLA_NAME \
    actor_rollout_ref.model.action_token_len=7 \
    actor_rollout_ref.model.action_chunks_len=8 \
    actor_rollout_ref.model.lora_rank=2 \
    actor_rollout_ref.model.lora_alpha=4 \
    actor_rollout_ref.model.target_modules="all-linear" \
    actor_rollout_ref.actor.optim.lr=2e-5 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.num_images_in_input=1 \
    actor_rollout_ref.actor.traj_mini_batch_size=2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.entropy_coeff=0. \
    actor_rollout_ref.rollout.num_images_in_input=1 \
    actor_rollout_ref.rollout.val_micro_batch_size=1 \
    actor_rollout_ref.rollout.temperature=1.6 \
    actor_rollout_ref.rollout.experiment_name=$EXPERIMENT_NAME \
    actor_rollout_ref.rollout.micro_batch_size=1 \
    actor_rollout_ref.rollout.unnorm_key=$DATASET_NAME \
    actor_rollout_ref.rollout.model_family=openvla \
    actor_rollout_ref.rollout.task_suite_name=$DATASET_NAME \
    actor_rollout_ref.rollout.num_steps_wait=10 \
    actor_rollout_ref.rollout.pretrained_checkpoint=$SFT_MODEL_PATH \
    actor_rollout_ref.rollout.center_crop=True \
    actor_rollout_ref.rollout.max_prompt_length=64 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +critic.lora_rank=2 \
    +critic.lora_alpha=4 \
    +critic.target_modules="all-linear" \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.logger=['console'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.total_epochs=20 \
    trainer.val_only=False \
    algorithm.adv_estimator=grpo \
    algorithm.adv_params.verifier_gamma=1.0 \
    algorithm.adv_params.reward_model_gamma=1.0 \
    trainer.runtime_env=$ALIGN_PATH \
    trainer.wandb_mode=offline \
    trainer.val_before_train=False \



