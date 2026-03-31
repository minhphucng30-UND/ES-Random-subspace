#!/bin/bash
#SBATCH -c 8 # request two cores 
#SBATCH -p hpg-b200,hpg-turin,hpg-milan,hpg-default
#SBATCH -o log/KFAC-Qwen.out
#SBATCH -e log/error-KFAC-Qwen.out
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=KFAC-Qwen
#SBATCH --ntasks-per-node=1
#SBATCH -G b200:1

source ~/.bashrc
conda activate peft_rl
module load gcc/14.2.0
nvidia-smi

  # --model_name runs/es_eggroll_test/checkpoint_step_500 \
  
pop_sizes=(8 16 32)

for pop_size in "${pop_sizes[@]}";do
torchrun --nproc_per_node=1  run_es_eggroll_sft.py \
  --data_path data/LIMO/train.parquet \
  --model_name models/Qwen3-1.7B \
  --lora_rank 1 \
  --filter_rank 4 \
  --population_size ${pop_size} \
  --micro_batch_size 2 \
  --num_micro_batches 8 \
  --sigma 5e-4 \
  --alpha 5e-4 \
  --output_dir runs/es_eggroll_rando_proj_pop${pop_size}_rank4

done