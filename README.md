slim-espnet

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](...)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c.svg)](...)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](...)

本项目基于 [ESPnet](https://espnet.github.io/espnet/)，旨在提供一个更轻量、更现代化的语音处理库。

本项目基于 ESPnet (Apache 2.0)。完整第三方许可声明请参阅 Third-Party Notices。

## 简介

本项目从官方 ESPnet 仓库中提取了 `egs3`、`espnet3` 和 `tools` 模块，并进行了独立打包。它移除了对旧版本代码的依赖，专注于提供简洁、高效的语音识别（ASR）、语音合成（TTS）等任务的训练和推理流程。

本项目以 PyTorch Lightning 为核心，提供独立的数据处理封装。

本项目采用 YAML + OmegaConf + Hydra 进行配置（包括但不限于 LM、LDM、Trainer、Callbacks 等）。

本项目命令行工具基于 OmegaConf 实现。

本项目正在集成 `espnet_model_zoo` 与 `espnet-tts-frontend` 的轻量化依赖，以进一步简化模型下载与前端处理流程。

本项目目标文件结构如下：
```text
slim-espnet/
├── egs/            # 各任务示例
├── espnet/         # 核心实现
├── tasks/          # 标准训练与推理流程
├── tools/          # 工具脚本
├── egs3/           # 兼容目录（计划移除）
├── espnet2/        # 兼容目录（计划移除）
├── espnet3/        # 兼容目录（计划移除）
└── README.md
```

## 安装

1.  （可选）如果是 windows 系统，安装 WSL2 + Ubuntu-22.04

    管理员身份打开 PowerShell，启用虚拟机平台和 WSL 功能：
    ```bash
    $ dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
    $ dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
    ```
    安装成功后重启电脑，再次以管理员身份打开 PS，输入：
    ```bash
    $ wsl --install						    # 安装最新版本 WSL
    $ wsl --list --online					# 列出可用的发行版版本
    $ wsl --install -d Ubuntu-22.04 --location D:\WSL\Ubuntu-22.04		# 下载安装注册启动（需要较新的 WSL 版本支持 --location。）
    ```
    在 Windows 用户目录（C:\Users\<用户名>）下创建 .wslconfig 文件，添加网络配置（让 WSL 也可以用 windows 的代理）：
    ```bash
    [wsl2]
    networkingMode=mirrored
    dnsTunneling=true
    ```
    关掉 WSL 再重启即生效

2.  Ubuntu 环境安装

    (可选)建议在 Ubuntu 环境安装 ffmpeg cmake sox flac
    ```bash
    $ sudo apt update 
    $ sudo apt install -y ffmpeg cmake sox flac
    $ cmake --version && sox --version && flac --version
    ```
    安装 miniconda
    ```bash
    $ cd ~ 
    $ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    $ bash Miniconda3-latest-Linux-x86_64.sh
    $ source ~/.bashrc
    ```
3.  conda 激活虚拟环境

    创建虚拟环境并激活
    ```bash
    $ conda create -n espnet python=3.12 -y
    $ conda activate espnet
    $ conda install -c conda-forge uv -y    # 安装 uv 用于后续包安装
    ```
    安装 PyTorch + CUDA（请根据 GPU 选择对应版本，例如 RTX 5060 Ti 至少需要 CUDA 12.8（SM120））
    ```bash
    $ uv pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
    ```
4.  安装 slim-espnet

    克隆仓库：
    ```bash
    $ cd ~
    $ git clone https://github.com/jeffchen0325/slim-espnet.git
    ```
    安装 slim-espnet
    ```bash
    $ cd <slim-espnet root>
    $ uv pip install -e .[all]    
    ```
    验证安装
    ```bash
    $ uv pip show slim-espnet
    ```
    或
    ```bash
    $ cd <slim-espnet root>/tools
    $ python3 check_install.py
    ```

## 🚀 快速开始

以下是一个简单的示例，展示如何运行一个基础的 ASR 实验：
```bash
$ bash ~/slim-espnet/tools/installers/install_warp-transducer.sh    # ASR模型依赖
$ cd ~/slim-espnet/egs/mini_an4/asr
$ python3 run.py dry_run=True
```
