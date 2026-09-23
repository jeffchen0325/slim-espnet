# slim-espnet

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)

**slim-espnet** 是一个基于 [ESPnet](https://espnet.github.io/espnet/) 的轻量化语音处理框架。

本项目目标是在保持 ESPnet 核心能力的基础上，移除历史包袱和不必要依赖，提供一个更加简洁、现代化、易维护的语音 AI 开发环境。

> 本项目基于 ESPnet（Apache License 2.0）。
>
> 完整第三方许可声明请参阅 `Third-Party Notices`。

---

## 简介

本项目从官方 ESPnet 仓库中提取并重新组织以下模块：
- `egs3`
- `espnet2`
- `espnet3`
- `tools`

并进行独立打包，减少对旧版本代码和复杂依赖链的依赖。

主要面向以下任务：
- 自动语音识别（ASR）
- 语音合成（TTS）
- 语音增强（Speech Enhancement）
- 语音转换（Speech Conversion）
- 其他语音生成相关任务

---

## 主要特点

- 🚀 **轻量化**
  - 移除历史兼容代码
  - 减少复杂依赖

- 🔥 **现代 PyTorch 生态**
  - 基于 PyTorch Lightning 管理训练流程
  - 支持现代 GPU 环境

- ⚙️ **统一配置系统**
  - 使用 YAML + OmegaConf + Hydra 管理配置
  - 支持：
    - Trainer
    - Callback
    - Language Model
    - Latent Diffusion Model
    - 其他组件配置

- 🧩 **模块化设计**
  - 数据处理
  - 模型定义
  - 训练流程
  - 推理流程
  - 工具脚本

均保持独立结构。

- 📦 **简化模型与前处理依赖**

正在集成：
- `espnet_model_zoo`
- `espnet-tts-frontend`

---

## 项目结构

目标目录结构：
```text
slim-espnet/
├── egs3/              # 各任务示例
├── espnet2/           # 核心实现
├── espnet3/           # 核心实现
├── tasks/             # 标准训练与推理流程
├── tools/             # 工具脚本
└── README.md
```

---

# 安装

## 1. Windows 环境（可选）

推荐使用：
- Windows 11
- WSL2
- Ubuntu 22.04


管理员身份打开 PowerShell：

启用虚拟机平台：
```powershell
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

启用 WSL：
```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
```

重启系统后：
```powershell
wsl --install
```

查看可用发行版：
```powershell
wsl --list --online
```

安装 Ubuntu：
```powershell
wsl --install -d Ubuntu-22.04 --location D:\WSL\Ubuntu-22.04
```

---

## WSL 网络配置（推荐）

在 Windows 用户目录：
```
C:\Users\<用户名>\.wslconfig
```

创建配置：
```ini
[wsl2]

networkingMode=mirrored
dnsTunneling=true
```

关闭并重新启动 WSL：
```powershell
wsl --shutdown
```

---

# Ubuntu 环境准备

## 安装系统依赖

推荐安装：
```bash
sudo apt update
sudo apt install -y ffmpeg cmake sox flac
```

验证：
```bash
cmake --version
sox --version
flac --version
```

---

# 安装 Miniconda

下载：
```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

安装：
```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

重新加载环境：
```bash
source ~/.bashrc
```

---

# 创建 Python 环境

创建环境：
```bash
conda create -n espnet python=3.12 -y
```

激活：
```bash
conda activate espnet
```

安装 uv：
```bash
conda install -c conda-forge uv -y
```

---

# 安装 PyTorch

根据 GPU 环境选择对应版本。

例如：

CUDA 12.8：
```bash
uv pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

> 注意：
>
> 不同 GPU 架构需要匹配对应 CUDA 版本。
>
> 例如部分新架构 GPU 需要 CUDA 12.8 或更新版本支持。

---

# 安装 slim-espnet

## 克隆仓库

```bash
cd ~
git clone https://github.com/jeffchen0325/slim-espnet.git
```

进入目录：
```bash
cd slim-espnet
```

安装：
```bash
uv pip install -e ".[all]"
```

---

# 验证安装

查看安装信息：
```bash
uv pip show slim-espnet
```

或者：
```bash
cd tools
python3 check_install.py
```

---

# 快速开始

以下示例运行一个基础 ASR 实验。

## 安装 ASR 依赖

```bash
bash ~/slim-espnet/tools/installers/install_warp-transducer.sh
```

## 运行测试任务

```bash
cd ~/slim-espnet/egs3/test/asr
python3 run.py dry_run=True
```

---

# 开发计划

- [x] 完成 ESPnet 模型生态简化
- [x] 集成轻量模型 Zoo
- [ ] 简化 TTS Frontend
- [ ] 提供更多端到端示例
- [ ] 支持更轻量化部署流程

---

# License

本项目基于：

- ESPnet
- Apache License 2.0

详细许可信息请参阅：

```
Third-Party Notices
```

---

# Acknowledgements

感谢以下项目：
- ESPnet
- PyTorch
- PyTorch Lightning
- Hydra
- OmegaConf
- YAML
