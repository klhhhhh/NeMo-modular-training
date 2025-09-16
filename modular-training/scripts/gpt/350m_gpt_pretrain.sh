#!/bin/bash

source /pscratch/sd/k/klhhhhh/envs/nemo/bin/activate
source /global/homes/k/klhhhhh/NeMo-modular-training/modular-training/scripts/gpt/export_package.sh

export WANDB_API_KEY=54c49dff7abb6ed19894a8aaec8b305d316f0072

torchrun \
    --nnodes=8 \
    --nproc_per_node=4 \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --rdzv_id=gpt_350m \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    /global/homes/k/klhhhhh/NeMo-modular-training/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
    --config-path=/global/homes/k/klhhhhh/NeMo-modular-training/examples/nlp/language_modeling/conf \
    --config-name=megatron_gpt_config \
    trainer.devices=4 \
    trainer.num_nodes=8 \
    trainer.max_epochs=null \
    trainer.max_steps=100000 \
    trainer.val_check_interval=1000 \
    trainer.log_every_n_steps=25 \
    trainer.limit_val_batches=50 \
    trainer.limit_test_batches=50 \
    trainer.accumulate_grad_batches=1 \
    trainer.precision=bf16 \
    model.transformer_engine=True \
    model.megatron_amp_O2=False \
    model.micro_batch_size=8 \
    model.global_batch_size=256 \
    model.tensor_model_parallel_size=1 \
    model.pipeline_model_parallel_size=1 \
    model.max_position_embeddings=1024 \
    model.encoder_seq_length=1024 \
    model.hidden_size=1024 \
    model.ffn_hidden_size=4096 \
    model.num_layers=24 \
    model.num_attention_heads=16 \
    model.init_method_std=0.021 \
    model.hidden_dropout=0.1 \
    model.layernorm_epsilon=1e-5 \
    model.tokenizer.vocab_file=/pscratch/sd/k/klhhhhh/dataset/gpt2-datasets/gpt2-vocab.json \
    model.tokenizer.merge_file=/pscratch/sd/k/klhhhhh/dataset/gpt2-datasets/gpt2-merges.txt \
    model.data.data_prefix=[1,/pscratch/sd/k/klhhhhh/openwebtext_data/my-gpt2-oepnwebtext_text_document] \
    model.data.num_workers=4 \
    model.data.seq_length=1024 \
    model.data.splits_string=\'980,10,10\' \
    model.optim.name=fused_adam \
    model.optim.lr=3e-4 \
    model.optim.betas=[0.9,0.95] \
    model.optim.weight_decay=0.1 \
    model.optim.sched.name=CosineAnnealing \
    model.optim.sched.warmup_steps=2000 \
    model.optim.sched.constant_steps=80000 \
    model.optim.sched.min_lr=6e-5 \
    exp_manager.exp_dir=/pscratch/sd/k/klhhhhh/run_gpt_350m_compare \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.project="project_train_350m" \
    exp_manager.wandb_logger_kwargs.name="run_compare_$SLURM_JOB_ID" \
    exp_manager.resume_if_exists=True \
    exp_manager.resume_ignore_no_checkpoint=True \
    exp_manager.create_checkpoint_callback=True \
    exp_manager.checkpoint_callback_params.dirpath=/pscratch/sd/k/klhhhhh/checkpoints/nemo/gpt_350m \
    exp_manager.checkpoint_callback_params.monitor=val_loss \
    exp_manager.checkpoint_callback_params.save_top_k=5 \
    exp_manager.checkpoint_callback_params.mode=min \
    exp_manager.checkpoint_callback_params.always_save_nemo=True