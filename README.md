slimEspnets

独立的 ESPnet 框架。本项目基于 [ESPnet](https://espnet.github.io/espnet/)，旨在提供一个更轻量、更现代化的语音处理库。

## 简介

本项目从官方 ESPnet 仓库中提取了 `egs3`、`espnet3` 和 `tools` 模块，并进行了独立打包。它移除了对旧版本代码的依赖，专注于提供简洁、高效的语音识别（ASR）、语音合成（TTS）等任务的训练和推理流程。

本项目以 pytorch lighning 为核心，外部封装独立的数据处理。

本项目采用 yaml + omegaConf + hydra 进行配置（包括但不限于 LM， LDM， Trainer， Callbacks 等）。

本项目命令行工具使用 omegaConf。

本项目准备内嵌 espnet_model_zoo 和 espnet-tts-frontend 简洁包依赖（ongoing）。

本项目目标文件结构如下：

Espnet

├── egs/			# 各种数据集及相关模式的实战用例
├── egs3/			    # 原备份（将来删除）
├── espnet2/			# 原备份（将来删除）
├── espnet3/			# 原备份（将来删除）
├── espnet/			# 所有模式的底层实现
├── tasks/			# 每种模式实战的操作流程
├── tool/			# 所有模式的底层实现
└── tools/			    # 原备份（将来删除）

## 安装

1.  （可选）如果是windows系统，安装 WSL2 + Ubuntu-22.04

    管理员身份打开PowerShell，启用虚拟机平台和WSL功能：
    ```bash
    $ dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
    $ dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
    ```
    安装成功后重启电脑，再次以管理员身份打开PS，输入：
    ```bash
    $ wsl --install						    # 安装最新版本WSL
    $ wsl --list --online					# 列出可用的发行版版本
    $ wsl --install -d Ubuntu-22.04 --location D:\WSL\Ubuntu-22.04		# 下载安装注册启动
    ```
    在 Windows 用户目录（C:\Users\<用户名>）下创建 .wslconfig 文件，添加网络配置（让WSL也可以用windows的代理）：
    ```bash
    [wsl2]
    networkingMode=mirrored
    dnsTunneling=true
    autoProxy=true
    ```
    关掉wsl再重启即生效

2.  Ubuntu环境安装

    (可选)建议在Ubuntu环境安装ffmpeg cmake sox flac
    ```bash
    $ sudo apt update 
    $ sudo apt install -y ffmpeg cmake sox flac
    $ cmake --version && sox --version && flac –version
    ```
    安装miniconda
    ```bash
    $ cd ~ 
    $ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    $ bash Miniconda3-latest-Linux-x86_64.sh
    $ source ~/.bashrc
    ```
3.  conda激活虚拟环境

    创建虚拟环境并激活
    ```bash
    $ conda create -n espnet python=3.10
    $ conda activate espnet
    $ conda install -c conda-forge uv -y    # 安装 uv 用于后续包安装
    ```
    安装torch+cuda（选取合适的pytorch+cuda版本， 例如5060ti至少cuda12.8支持sm120）
    ```bash
    $ uv pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
    ```
4.  安装slimESPnet

    克隆仓库：
    ```bash
    $ cd ~
    $ git clone https://github.com/jeffchen0325/slimESPnet.git
    ```
    安装slimESPnet
    ```bash
    $ cd <slimESPnet-root>
    $ uv pip install -e .[all]    
    ```
    检查slimESPnet版本
    ```bash
    $ uv pip show slimESPnet
    ```
    或
    ```bash
    $ cd <slimESPnet-root>/tool
    $ python3 check_install.py
    ```

## 🚀 快速开始

以下是一个简单的示例，展示如何运行一个基础的 ASR 实验：
```bashs
$ bash ~/slimEspnet/tools/installers/install_warp-transducer.sh    # ASR模型依赖
$ cd espnet/egs/mini_an4/asr
$ python3 run.py dry_run=True
```
