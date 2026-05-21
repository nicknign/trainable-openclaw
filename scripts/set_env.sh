#!/usr/bin/env bash
# ==============================================================================
# trainable-openclaw 3090 Linux 环境安装脚本
#
# 用法:
#   chmod +x setup_env.sh
#   ./setup_env.sh
#
# 自定义配置（环境变量覆盖）:
#   CUDA_VER=12.6  VENV_DIR=./venv  ./setup_env.sh
# ==============================================================================
set -x

# Fix: conda libs need newer libstdc++ than system provides
export LD_LIBRARY_PATH=/data/anaconda3/lib:$LD_LIBRARY_PATH

cd /data/wangye/trainable-openclaw
pip install -r requirements.txt
cd /data/wangye/trainable-openclaw/verl-main-0516 && pip install .


