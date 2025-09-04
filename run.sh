#!/bin/bash
#SBATCH --chdir=/home/zjhzjh/workdir/Modify_MonoDGP  # 设置工作目录
#SBATCH --job-name=wSCAB     # 任务名称
#SBATCH --partition=gpu-a40     # 请求的分区
#SBATCH --nodes=1                         # 请求一个节点
#SBATCH --ntasks=1                        # 请求一个任务
#SBATCH --cpus-per-task=20                # 每个任务使用一个 CPU
#SBATCH --gres=gpu:1
#SBATCH --error=logs/%j_quota.log                # 错误输出文件
#SBATCH --output=logs/%j_loss.log                # 输出文件

source /home/zjhzjh/miniconda3/etc/profile.d/conda.sh
conda activate monodgp

# ./train.sh configs/monodgp_scab.yaml
./test.sh configs/monodgp_scab.yaml

# python test.py