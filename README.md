```markdown
# YOLO11m 双机分布式目标
# YOLO11m 双机分布式目标检测 — 完整项目文档

> 整理自完整对话记录 | 版本 2.0

---

## 目录

1. [项目概述](#1-项目概述)
2. [硬件配置](#2-硬件配置)
3. [环境配置](#3-环境配置)
4. [固定 IP 与网络配置](#4-固定-ip-与网络配置)
5. [数据集](#5-数据集)
6. [训练代码](#6-训练代码)
7. [运行指令](#7-运行指令)
8. [GPU 监控](#8-gpu-监控)
9. [修改日志与问题记录](#9-修改日志与问题记录)
10. [训练过程记录](#10-训练过程记录)
11. [关键问题分析](#11-关键问题分析)
12. [常见问题排查](#12-常见问题排查)
13. [附录：参数速查表](#13-附录参数速查表)

---

## 1. 项目概述

| 项目 | 内容 |
|------|------|
| 模型 | YOLO11m |
| 任务 | 单类目标检测（nc=1, names=['1']） |
| 训练方式 | 双机分布式（2 节点 × 1 GPU） |
| 图像训练尺寸 | 1024（从 2K 原图缩放） |
| 总 Epoch | 3000 和 6000 均实际运行过 |
| 优化器 | 默认 SGD 和 AdamW 两种均尝试过 |
| 目标 mAP50 | 90% |
| 参考 mAP50-95 | 70% |
| EarlyStopping | 关闭（patience=0） |

**说明**：本文档包含两种优化器配置（默认 SGD / 显式 AdamW）和两种 epoch 配置（3000 / 6000），均实际运行过。具体某次训练使用了哪种配置，可通过 checkpoint 中的 `train_args` 验证（见[第 6.4 节](#64-验证实际使用的配置)）。

---

## 2. 硬件配置

| 项目 | node1 | node2 |
|------|-------|-------|
| GPU | NVIDIA RTX 3060 12GB | NVIDIA RTX 3060 12GB |
| 角色 | 主节点（Master） | 副节点（Worker） |
| 共享存储 | /data/shared | /data/shared |
| hostname | node1 | node2 |
| 网络 | 互通 | 互通 |

---

## 3. 环境配置

### 3.1 系统依赖

```bash
# 两台机器都执行

sudo apt update && sudo apt upgrade -y

# 安装 NVIDIA 驱动（如已安装跳过）
sudo apt install -y nvidia-driver-535
sudo reboot

# 确认驱动
nvidia-smi

# 基础工具
sudo apt install -y git vim htop tmux openssh-server
```

### 3.2 SSH 免密登录

```bash
# node1 上生成密钥（如已有跳过）
ssh-keygen -t rsa -b 4096

# 复制公钥到 node2
ssh-copy-id a@node2

# 测试
ssh a@node2 "echo OK"
# 应输出 OK
```

### 3.3 共享存储确认

```bash
# 两台机器都能访问 /data/shared
ls /data/shared
ssh a@node2 "ls /data/shared"

# 数据集和代码路径
# /data/shared/code/vision-train/    ← 代码 + 数据集
# /data/shared/envs/train/           ← conda 环境
```

### 3.4 安装 Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

~/miniconda3/bin/conda init bash
source ~/.bashrc

conda --version
```

### 3.5 创建 conda 环境

```bash
# 在 node1 上创建（共享存储，node2 直接用）
conda create -p /data/shared/envs/train python=3.10 -y

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/shared/envs/train

python --version
# Python 3.10.21
```

### 3.6 安装 PyTorch + CUDA

```bash
conda activate /data/shared/envs/train

pip install torch==2.13.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu130

python3 -c "
import torch
print(f'PyTorch : {torch.__version__}')
print(f'CUDA    : {torch.version.cuda}')
print(f'可用    : {torch.cuda.is_available()}')
print(f'GPU     : {torch.cuda.get_device_name(0)}')
print(f'显存    : {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"
# 应输出: CUDA: True, GPU: NVIDIA GeForce RTX 3060, 显存: 12.0 GB
```

### 3.7 安装 Ultralytics 和依赖

```bash
conda activate /data/shared/envs/train

pip install ultralytics==8.4.123
pip install opencv-python Pillow pandas matplotlib seaborn tqdm

python3 -c "
import ultralytics, torch, cv2
from PIL import Image
print(f'Ultralytics : {ultralytics.__version__}')
print(f'OpenCV      : {cv2.__version__}')
print(f'PyTorch     : {torch.__version__}')
print('所有依赖安装成功')
"
```

### 3.8 NCCL（分布式通信）

```bash
# NCCL 通常随 PyTorch CUDA 自动安装
python3 -c "import torch; print(f'NCCL: {torch.cuda.nccl.version()}')"
# 应输出: NCCL: (2, 29, 7) 或类似

# 如缺少，安装：
pip install nvidia-nccl-cu13
```

### 3.9 验证双机环境

```bash
# 1. SSH 连通
ssh a@node2 "hostname"

# 2. 共享存储
ssh a@node2 "ls /data/shared/code/vision-train/data.yaml"

# 3. node2 conda 环境
ssh a@node2 "source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate /data/shared/envs/train && \
python3 -c 'import torch; print(torch.cuda.is_available())'"

# 4. 端口连通
ssh a@node2 "python3 -c \"
import socket
s = socket.socket()
s.connect(('node1', 29600))
print('端口连通')
s.close()
\""
```

### 3.10 环境总结

| 组件 | 版本/路径 | 安装方式 |
|------|-----------|---------|
| Miniconda | latest | 官方脚本 |
| conda 环境 | `/data/shared/envs/train` | `conda create -p` |
| Python | 3.10.21 | conda |
| PyTorch | 2.13.0+cu130 | pip |
| Ultralytics | 8.4.123 | pip |
| CUDA | 13.0 | PyTorch 内置 |
| NCCL | 2.29.7+cuda13.2 | PyTorch 内置 |

### 3.11 日常激活命令

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/shared/envs/train
cd /data/shared/code/vision-train
```

---

## 4. 固定 IP 与网络配置

### 4.1 确认网口名称

```bash
ip a
# 或
nmcli device status
```

输出示例：
```
DEVICE  TYPE      STATE      CONNECTION
enp3s0  ethernet  connected  wired
lo      loopback  unmanaged  --
```

### 4.2 查看当前 IP 和网关

```bash
ip a show enp3s0 | grep inet
ip route | grep default
cat /etc/resolv.conf
```

### 4.3 设置固定 IP（Ubuntu Netplan）

```bash
sudo vim /etc/netplan/01-netcfg.yaml
```

**node1**：
```yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    enp3s0:
      dhcp4: no
      addresses:
        - 192.168.1.101/24
      routes:
        - to: default
          via: 192.168.1.1
      nameservers:
        addresses:
          - 223.5.5.5
          - 8.8.8.8
```

**node2**：
```yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    enp3s0:
      dhcp4: no
      addresses:
        - 192.168.1.102/24
      routes:
        - to: default
          via: 192.168.1.1
      nameservers:
        addresses:
          - 223.5.5.5
          - 8.8.8.8
```

```bash
# 两台机器都执行
sudo netplan apply
ip a show enp3s0 | grep inet
```

### 4.4 配置 /etc/hosts

```bash
sudo vim /etc/hosts
```

两台机器都添加：
```
192.168.1.101   node1
192.168.1.102   node2
```

### 4.5 验证网络

```bash
# node1
ping -c 3 node2
ping -c 3 192.168.1.102

# node2
ping -c 3 node1
ping -c 3 192.168.1.101

# SSH
ssh a@node2 "hostname"

# 端口
python3 -c "
import socket
s = socket.socket()
s.connect(('node1', 29600))
print('端口连通')
s.close()
"
```

### 4.6 网络配置总结

| 项目 | node1 | node2 |
|------|-------|-------|
| 网口 | enp3s0（以实际为准） | enp3s0（以实际为准） |
| IP | 192.168.1.101 | 192.168.1.102 |
| 子网掩码 | /24 | /24 |
| 网关 | 192.168.1.1 | 192.168.1.1 |
| DNS | 223.5.5.5, 8.8.8.8 | 223.5.5.5, 8.8.8.8 |
| hostname | node1 | node2 |
| 训练端口 | 29600 | 连接 node1:29600 |

---

## 5. 数据集

### 5.1 版本历史

| 版本 | 分辨率 | 说明 | 结果 |
|------|--------|------|------|
| v1（原始） | 512×512 | Roboflow 导出时被压缩 | mAP50-95 卡在 53%，瓶颈在分辨率 |
| v2（当前） | 2048×1152 | 重新从 Roboflow 导出 2K 图片 | 训练中/已完成（见修改日志） |

### 5.2 数据集结构

```
/data/shared/code/vision-train/
├── data.yaml
├── train/
│   ├── images/    # 8586 张 2048×1152 图片
│   └── labels/
├── valid/
│   ├── images/    # 757 张
│   └── labels/
├── test/
│   ├── images/    # 378 张
│   └── labels/
```

### 5.3 data.yaml

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 1
names: ['1']

roboflow:
  workspace: project-7ildc
  project: 1-euqas
  version: 2
  license: CC BY 4.0
  url: https://universe.roboflow.com/project-7ildc/1-euqas/dataset/2
```

### 5.4 如何更换数据集

1. **Roboflow 重新导出**，选原始分辨率
2. **替换图片和标签**：
```bash
cd /data/shared/code/vision-train
mv train train_old && mv valid valid_old && mv test test_old
# 解压新数据到 train/ valid/ test/
```
3. **删除旧 checkpoint**：
```bash
rm -rf runs/detect/target_detection/yolo11m_target_highmap_1024-2/weights/
```
4. **删除旧缓存**：
```bash
find /data/shared/code/vision-train -name "*.cache" -delete
```
5. **验证新数据集**：
```bash
python3 -c "
from pathlib import Path
from PIL import Image
from collections import Counter
sizes = Counter()
for p in Path('train/images').rglob('*'):
    if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}:
        with Image.open(p) as im:
            sizes[f'{im.size[0]}x{im.size[1]}'] += 1
print(f'总图片: {sum(sizes.values())}')
for s, c in sizes.most_common(10):
    print(f'  {s}: {c} 张')
"
```

---

## 6. 训练代码

文件路径：`/data/shared/code/vision-train/train_yolo11m_distributed.py`

### 6.1 版本 A：默认 SGD 优化器

```python
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import ultralytics
from ultralytics import YOLO

# ============================================================================
# 分布式环境变量
# ============================================================================
LOCAL_RANK  = int(os.environ.get("LOCAL_RANK", 0))
RANK        = int(os.environ.get("RANK", 0))
WORLD_SIZE  = int(os.environ.get("WORLD_SIZE", 1))
MASTER_ADDR = os.environ.get("MASTER_ADDR", "node1")
MASTER_PORT = os.environ.get("MASTER_PORT", "29600")
IS_MAIN     = RANK == 0

# ============================================================================
# 路径
# ============================================================================
ROOT_DIR = Path(__file__).resolve().parent
DATA_YAML = ROOT_DIR / "data.yaml"
TRAIN_IMAGES = ROOT_DIR / "train" / "images"
VAL_IMAGES = ROOT_DIR / "valid" / "images"
TEST_IMAGES = ROOT_DIR / "test" / "images"
RUN_DIR = ROOT_DIR / "runs" / "detect" / "target_detection" / "yolo11m_target_highmap_1024-2"
WEIGHTS_DIR = RUN_DIR / "weights"
LAST_PT = WEIGHTS_DIR / "last.pt"
BEST_PT = WEIGHTS_DIR / "best.pt"
RESULTS_CSV = RUN_DIR / "results.csv"
RESUME_PT = WEIGHTS_DIR / "_resume_target_6000.pt"

# ============================================================================
# 训练参数
# ============================================================================
PRETRAINED_MODEL = "yolo11m.pt"
TARGET_EPOCHS = 3000          # 可改为 6000
IMGSZ = 1024
BATCH = 6
WORKERS = 8
SAVE_PERIOD = 10
PATIENCE = 0
CLOSE_MOSAIC = 40
TARGET_MAP50 = 0.90
SEED = 42
LR0 = 0.01 * WORLD_SIZE
FINAL_VAL_PLOTS = True
STRICT_TRUE_RESUME = True
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
GB = 1024 ** 3

# ============================================================================
# 工具函数
# ============================================================================
def log(msg: str) -> None:
    if IS_MAIN:
        print(msg, flush=True)

def section(title: str) -> None:
    log("\n" + "=" * 100)
    log(title)
    log("=" * 100)

def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)

def as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}

def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return str(value)

# ============================================================================
# 分布式
# ============================================================================
def setup_distributed() -> None:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(LOCAL_RANK)

def cleanup_distributed() -> None:
    if dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()

# ============================================================================
# 环境检查
# ============================================================================
def check_environment() -> None:
    section("GPU / PyTorch / Ultralytics 环境检查（双机分布式）")
    log(f"Python        : {sys.version.split()[0]}")
    log(f"PyTorch       : {torch.__version__}")
    log(f"Ultralytics   : {ultralytics.__version__}")
    log(f"CUDA runtime  : {torch.version.cuda}")
    log(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")

    props = torch.cuda.get_device_properties(LOCAL_RANK)
    log(f"\n[分布式] WORLD_SIZE  : {WORLD_SIZE}")
    log(f"[分布式] RANK        : {RANK}")
    log(f"[分布式] LOCAL_RANK  : {LOCAL_RANK}")
    log(f"[分布式] MASTER      : {MASTER_ADDR}:{MASTER_PORT}")
    log(f"\nGPU           : {torch.cuda.get_device_name(LOCAL_RANK)}")
    log(f"VRAM          : {props.total_memory / GB:.2f} GB")
    log(f"CPU threads   : {os.cpu_count()}")
    log(f"Workers       : {WORKERS}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    log(f"\n训练参数:")
    log(f"  Batch/卡     : {BATCH}")
    log(f"  全局 Batch   : {BATCH * WORLD_SIZE}")
    log(f"  学习率 lr0   : {LR0} (单机0.01 x {WORLD_SIZE})")
    log(f"  优化器       : 默认（Ultralytics 自动选择）")
    log(f"  目标 mAP50   : {TARGET_MAP50 * 100:.0f}%")

def print_gpu_memory(prefix: str = "GPU显存") -> None:
    if not torch.cuda.is_available() or not IS_MAIN:
        return
    try:
        torch.cuda.synchronize(LOCAL_RANK)
    except Exception:
        pass
    allocated = torch.cuda.memory_allocated(LOCAL_RANK) / GB
    reserved = torch.cuda.memory_reserved(LOCAL_RANK) / GB
    total = torch.cuda.get_device_properties(LOCAL_RANK).total_memory / GB
    log(f"{prefix}: {allocated:.2f} GB allocated / {reserved:.2f} GB reserved / {total:.2f} GB total")

# ============================================================================
# 数据集检查
# ============================================================================
def count_images(directory: Path) -> int:
    return sum(1 for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)

def check_dataset() -> dict[str, int]:
    section("检查数据集")
    required = {"data.yaml": DATA_YAML, "train/images": TRAIN_IMAGES, "valid/images": VAL_IMAGES, "test/images": TEST_IMAGES}
    for name, path in required.items():
        log(f"{name:14s}: {path}")
        if not path.exists():
            raise FileNotFoundError(f"缺少 {name}: {path}")
    counts = {"train": count_images(TRAIN_IMAGES), "valid": count_images(VAL_IMAGES), "test": count_images(TEST_IMAGES)}
    log("\n图片数量:")
    for split, count in counts.items():
        log(f"  {split:5s}: {count}")
    if counts["train"] == 0:
        raise RuntimeError("train/images 中没有找到图片。")
    if counts["valid"] == 0:
        raise RuntimeError("valid/images 中没有找到图片。")
    return counts

# ============================================================================
# 进度判断
# ============================================================================
def read_results_progress() -> int | None:
    if not RESULTS_CSV.exists():
        return None
    last_epoch: float | None = None
    try:
        with RESULTS_CSV.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return None
            epoch_key = next((k for k in reader.fieldnames if k.strip().lower() == "epoch"), None)
            if epoch_key is None:
                return None
            for row in reader:
                raw = row.get(epoch_key)
                if raw is None or not raw.strip():
                    continue
                try:
                    last_epoch = float(raw)
                except ValueError:
                    continue
    except Exception as exc:
        log(f"⚠ 无法读取 results.csv 进度: {exc}")
        return None
    if last_epoch is None:
        return None
    return max(0, int(round(last_epoch)))

def checkpoint_completed_epochs(checkpoint: dict[str, Any]) -> int | None:
    epoch = checkpoint.get("epoch")
    if epoch is None:
        return None
    try:
        epoch_i = int(epoch)
    except (TypeError, ValueError):
        return None
    if epoch_i < 0:
        return None
    return epoch_i + 1

def get_completed_epochs(checkpoint: dict[str, Any]) -> tuple[int | None, str]:
    ckpt_epochs = checkpoint_completed_epochs(checkpoint)
    csv_epochs = read_results_progress()
    candidates = [x for x in (ckpt_epochs, csv_epochs) if x is not None]
    if not candidates:
        return None, "unknown"
    completed = max(candidates)
    if ckpt_epochs is not None and csv_epochs is not None:
        source = "checkpoint + results.csv"
    elif ckpt_epochs is not None:
        source = "checkpoint"
    else:
        source = "results.csv"
    return completed, source

# ============================================================================
# checkpoint 检查
# ============================================================================
def check_checkpoint() -> dict[str, Any] | None:
    section("检查训练状态")
    log(f"实验目录 : {RUN_DIR}")
    log(f"last.pt  : {LAST_PT}")
    log(f"best.pt  : {BEST_PT}")

    if not LAST_PT.exists():
        log("\n✓ 未检测到 last.pt：判定为首次训练。")
        log(f"  将从预训练模型 {PRETRAINED_MODEL} 开始。")
        return None

    size_mb = LAST_PT.stat().st_size / (1024 ** 2)
    log(f"last.pt 大小: {size_mb:.2f} MB")

    checkpoint = load_checkpoint(LAST_PT)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("last.pt 不是可识别的 Ultralytics checkpoint 字典。")

    train_args = as_dict(checkpoint.get("train_args", {}))
    completed, progress_source = get_completed_epochs(checkpoint)
    epoch = checkpoint.get("epoch")
    optimizer = checkpoint.get("optimizer")
    scaler = checkpoint.get("scaler")
    ema = checkpoint.get("ema")

    log("\nCheckpoint 状态:")
    log(f"  epoch 字段       : {epoch}")
    log(f"  已完成 epoch     : {completed} ({progress_source})")
    log(f"  optimizer        : {'存在' if optimizer is not None else '不存在'}")
    log(f"  AMP scaler       : {'存在' if scaler is not None else '不存在'}")
    log(f"  EMA              : {'存在' if ema is not None else '不存在'}")
    log(f"  best_fitness     : {checkpoint.get('best_fitness')}")
    log(f"  checkpoint 版本  : {checkpoint.get('version')}")

    if train_args:
        log("\nCheckpoint 关键训练参数:")
        for key in ("epochs", "imgsz", "batch", "device", "optimizer", "lr0", "lrf", "cos_lr", "amp", "mosaic", "mixup", "close_mosaic", "patience", "workers", "save_period", "project", "name", "save_dir"):
            if key in train_args:
                log(f"  {key:14s}: {train_args.get(key)}")
    return checkpoint

def is_training_complete(checkpoint: dict[str, Any]) -> bool:
    completed, _ = get_completed_epochs(checkpoint)
    return completed is not None and completed >= TARGET_EPOCHS

def is_true_resumable(checkpoint: dict[str, Any]) -> bool:
    completed = checkpoint_completed_epochs(checkpoint)
    return completed is not None and checkpoint.get("optimizer") is not None

# ============================================================================
# 准备 resume checkpoint
# ============================================================================
def prepare_resume_checkpoint(checkpoint: dict[str, Any]) -> Path:
    if not is_true_resumable(checkpoint):
        completed, source = get_completed_epochs(checkpoint)
        message = (
            "last.pt 当前不包含完整 optimizer/epoch 训练状态，不能进行严格 resume。\n"
            f"检测到进度: {completed} ({source})\n"
            "这通常发生在训练已经正常结束后，Ultralytics 对 checkpoint 执行了 optimizer stripping。"
        )
        if STRICT_TRUE_RESUME:
            raise RuntimeError(message + "\n为避免伪装成完整续训，本脚本已停止。")
        log("⚠ " + message)
        return LAST_PT

    patched = dict(checkpoint)
    train_args = dict(as_dict(checkpoint.get("train_args", {})))
    old_epochs = train_args.get("epochs")

    train_args["epochs"] = TARGET_EPOCHS
    train_args["data"] = str(DATA_YAML)
    train_args["project"] = str(RUN_DIR.parent)
    train_args["name"] = RUN_DIR.name
    train_args["exist_ok"] = True
    train_args["save_dir"] = str(RUN_DIR)
    train_args["patience"] = PATIENCE
    patched["train_args"] = train_args

    for model_key in ("ema", "model"):
        model_obj = patched.get(model_key)
        if model_obj is None or not hasattr(model_obj, "args"):
            continue
        try:
            model_args = dict(as_dict(model_obj.args))
            model_args.update(train_args)
            model_obj.args = model_args
        except Exception:
            pass

    if IS_MAIN:
        atomic_torch_save(patched, RESUME_PT)
        size_mb = RESUME_PT.stat().st_size / (1024 ** 2)
        log("\n准备 resume checkpoint:")
        log(f"  原 epochs : {old_epochs}")
        log(f"  新 epochs : {TARGET_EPOCHS}")
        log(f"  临时文件  : {RESUME_PT}")
        log(f"  文件大小  : {size_mb:.2f} MB")
        log("  原 last.pt: 未修改")

    if WORLD_SIZE > 1 and dist.is_initialized():
        dist.barrier()
    return RESUME_PT

# ============================================================================
# 首次训练
# ============================================================================
def train_new() -> Any:
    section("首次训练（双机分布式）")
    log(f"预训练模型 : {PRETRAINED_MODEL}")
    log(f"数据集     : {DATA_YAML}")
    log(f"总 Epoch   : {TARGET_EPOCHS}")
    log(f"图像尺寸   : {IMGSZ}")
    log(f"Batch/卡   : {BATCH}")
    log(f"全局 Batch : {BATCH * WORLD_SIZE}")
    log(f"学习率     : {LR0} (单机0.01 x {WORLD_SIZE})")
    log(f"优化器     : 默认（Ultralytics 自动选择）")
    log(f"节点数     : {WORLD_SIZE}")
    log(f"训练目录   : {RUN_DIR}")
    log("EarlyStopping: 关闭 (patience=0)")

    if IS_MAIN:
        RUN_DIR.parent.mkdir(parents=True, exist_ok=True)
    if WORLD_SIZE > 1 and dist.is_initialized():
        dist.barrier()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = YOLO(PRETRAINED_MODEL)
    return model.train(
        data=str(DATA_YAML),
        epochs=TARGET_EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=f"cuda:{LOCAL_RANK}",
        workers=WORKERS,
        project=str(RUN_DIR.parent),
        name=RUN_DIR.name,
        exist_ok=True,
        save=True,
        save_period=SAVE_PERIOD,
        patience=PATIENCE,
        close_mosaic=CLOSE_MOSAIC,
        seed=SEED,
        deterministic=False,
        pretrained=True,
        amp=True,
        lr0=LR0,
        plots=False,
        verbose=IS_MAIN,
    )

# ============================================================================
# 断点续训
# ============================================================================
def train_resume(checkpoint: dict[str, Any]) -> Any:
    section("自动断点续训（双机分布式）")
    completed, source = get_completed_epochs(checkpoint)
    remaining = None if completed is None else max(0, TARGET_EPOCHS - completed)

    log(f"当前已完成 : {completed} epoch ({source})")
    log(f"最终总目标 : {TARGET_EPOCHS} epoch")
    log(f"预计剩余   : {remaining if remaining is not None else '未知'} epoch")
    log(f"Batch/卡   : {BATCH}")
    log(f"全局 Batch : {BATCH * WORLD_SIZE}")
    log(f"节点数     : {WORLD_SIZE}")
    log(f"目标 mAP50 : {TARGET_MAP50 * 100:.0f}%")
    log("EarlyStopping: 关闭 (patience=0)")
    log("训练目录   : 固定为原 RUN_DIR")

    resume_path = prepare_resume_checkpoint(checkpoint)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = YOLO(str(resume_path))
    results = model.train(
        resume=True,
        imgsz=IMGSZ,
        batch=BATCH,
        device=f"cuda:{LOCAL_RANK}",
        workers=WORKERS,
        save_period=SAVE_PERIOD,
        patience=PATIENCE,
        close_mosaic=CLOSE_MOSAIC,
        plots=False,
        verbose=IS_MAIN,
    )

    if IS_MAIN and RESUME_PT.exists():
        RESUME_PT.unlink(missing_ok=True)
    if WORLD_SIZE > 1 and dist.is_initialized():
        dist.barrier()
    return results

# ============================================================================
# 最终验证（仅 rank0）
# ============================================================================
def evaluate_best() -> Any:
    if not IS_MAIN:
        if WORLD_SIZE > 1 and dist.is_initialized():
            dist.barrier()
        return None

    section("最终模型验证")
    if BEST_PT.exists():
        model_path = BEST_PT
        log("✓ 使用 best.pt")
    elif LAST_PT.exists():
        model_path = LAST_PT
        log("⚠ best.pt 不存在，改用 last.pt")
    else:
        log("⚠ best.pt 和 last.pt 都不存在，跳过验证。")
        if WORLD_SIZE > 1 and dist.is_initialized():
            dist.barrier()
        return None

    log(f"验证模型: {model_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(DATA_YAML),
        imgsz=IMGSZ,
        batch=BATCH,
        device=f"cuda:{LOCAL_RANK}",
        plots=FINAL_VAL_PLOTS,
        verbose=True,
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)

    section("最终验证结果")
    log(f"Precision : {precision:.4f} ({precision * 100:.2f}%)")
    log(f"Recall    : {recall:.4f} ({recall * 100:.2f}%)")
    log(f"mAP50     : {map50:.4f} ({map50 * 100:.2f}%)")
    log(f"mAP50-95  : {map5095:.4f} ({map5095 * 100:.2f}%)")

    if map50 >= TARGET_MAP50:
        log(f"\n✓ 达到目标：mAP50 >= {TARGET_MAP50 * 100:.0f}%")
    else:
        log(f"\n⚠ 当前 mAP50={map50 * 100:.2f}% ，目标={TARGET_MAP50 * 100:.2f}%")

    log("\n各类别 AP50:")
    try:
        class_names = model.names
        for i, ap in enumerate(metrics.box.ap50):
            if isinstance(class_names, dict):
                name = class_names.get(i, str(i))
            else:
                name = class_names[i]
            log(f"  {str(name):>20s}: {float(ap) * 100:7.2f}%")
    except Exception as exc:
        log(f"  ⚠ 无法输出每类别 AP50: {exc}")

    if WORLD_SIZE > 1 and dist.is_initialized():
        dist.barrier()
    return metrics

# ============================================================================
# 保存摘要
# ============================================================================
def save_summary(metrics: Any, checkpoint: dict[str, Any], dataset_counts: dict[str, int]) -> None:
    if not IS_MAIN or metrics is None:
        return
    try:
        train_args = as_dict(checkpoint.get("train_args", {}))
        completed, progress_source = get_completed_epochs(checkpoint)
        selected_args = {
            key: json_safe(train_args.get(key))
            for key in ("epochs", "imgsz", "batch", "device", "optimizer", "lr0", "lrf", "weight_decay", "cos_lr", "amp", "mosaic", "mixup", "close_mosaic", "patience", "workers", "save_period", "seed", "deterministic")
            if key in train_args
        }
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "dataset": str(DATA_YAML),
            "dataset_counts": dataset_counts,
            "run_dir": str(RUN_DIR),
            "last_model": str(LAST_PT),
            "best_model": str(BEST_PT),
            "target_epochs": TARGET_EPOCHS,
            "detected_completed_epochs": completed,
            "progress_source": progress_source,
            "checkpoint_epoch_field": checkpoint.get("epoch"),
            "checkpoint_version": checkpoint.get("version"),
            "gpu": torch.cuda.get_device_name(LOCAL_RANK) if torch.cuda.is_available() else "CPU",
            "world_size": WORLD_SIZE,
            "global_batch": BATCH * WORLD_SIZE,
            "lr0": LR0,
            "optimizer": "default (Ultralytics auto)",
            "resume_train_args": selected_args,
            "target_map50": TARGET_MAP50,
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
        }
        output_file = RUN_DIR / "resume_final_metrics.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        log(f"\n最终摘要已保存: {output_file}")
    except Exception as exc:
        log(f"⚠ 保存摘要失败: {exc}")

def cleanup_resume_checkpoint() -> None:
    try:
        if IS_MAIN and RESUME_PT.exists():
            RESUME_PT.unlink()
    except Exception as exc:
        log(f"⚠ 无法删除临时 resume checkpoint: {exc}")

# ============================================================================
# 主程序
# ============================================================================
def main() -> None:
    setup_distributed()
    start_time = time.time()

    try:
        check_environment()
        print_gpu_memory()
        dataset_counts = check_dataset()
        checkpoint = check_checkpoint()

        if checkpoint is None:
            train_new()
            if WORLD_SIZE > 1 and dist.is_initialized():
                dist.barrier()
            if not LAST_PT.exists():
                raise RuntimeError("首次训练结束后仍未找到 last.pt。")
            final_checkpoint = load_checkpoint(LAST_PT)
            metrics = evaluate_best()
            save_summary(metrics, final_checkpoint, dataset_counts)
            print_gpu_memory()
            elapsed_hours = (time.time() - start_time) / 3600
            section("首次训练流程完成")
            log(f"目标 Epoch : {TARGET_EPOCHS}")
            log(f"本次运行时间: {elapsed_hours:.2f} 小时")
            log(f"实验目录: {RUN_DIR}")
            log(f"Best     : {BEST_PT}")
            log(f"Last     : {LAST_PT}")
            return

        if is_training_complete(checkpoint):
            section("检测到目标训练已经完成")
            completed, source = get_completed_epochs(checkpoint)
            log(f"已完成: {completed} epoch ({source})")
            log(f"目标   : {TARGET_EPOCHS} epoch")
            log("不会再次 resume，直接进行最终验证。")
            metrics = evaluate_best()
            latest_checkpoint = load_checkpoint(LAST_PT)
            save_summary(metrics, latest_checkpoint, dataset_counts)
            cleanup_resume_checkpoint()
            return

        if not is_true_resumable(checkpoint) and STRICT_TRUE_RESUME:
            completed, source = get_completed_epochs(checkpoint)
            raise RuntimeError(
                "检测到 last.pt 不是可完整恢复的训练 checkpoint。\n"
                f"当前进度: {completed} ({source})\n"
                "optimizer 或有效 epoch 状态已经不存在。"
            )

        train_resume(checkpoint)

        if WORLD_SIZE > 1 and dist.is_initialized():
            dist.barrier()

        final_checkpoint = load_checkpoint(LAST_PT)
        metrics = evaluate_best()
        save_summary(metrics, final_checkpoint, dataset_counts)
        print_gpu_memory()
        cleanup_resume_checkpoint()

        elapsed_hours = (time.time() - start_time) / 3600
        section("断点续训流程完成")
        log(f"目标 Epoch : {TARGET_EPOCHS}")
        log("EarlyStopping: 关闭")
        log(f"本次运行时间: {elapsed_hours:.2f} 小时")
        log(f"实验目录: {RUN_DIR}")
        log(f"Best     : {BEST_PT}")
        log(f"Last     : {LAST_PT}")

    except KeyboardInterrupt:
        section("用户手动中断训练")
        log(f"last.pt: {LAST_PT}")
        log("下次直接重新运行本脚本即可；它会重新检测 last.pt 并继续 resume。")
        sys.exit(130)

    except Exception as exc:
        section("训练程序发生错误")
        log(f"{type(exc).__name__}: {exc}")
        raise

    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()
```

### 6.2 版本 B：显式 AdamW 优化器

与版本 A 唯一的区别是 `train_new()` 中的 `model.train()` 调用加了两个参数：

```python
    model = YOLO(PRETRAINED_MODEL)
    return model.train(
        data=str(DATA_YAML),
        epochs=TARGET_EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=f"cuda:{LOCAL_RANK}",
        workers=WORKERS,
        project=str(RUN_DIR.parent),
        name=RUN_DIR.name,
        exist_ok=True,
        save=True,
        save_period=SAVE_PERIOD,
        patience=PATIENCE,
        close_mosaic=CLOSE_MOSAIC,
        seed=SEED,
        deterministic=False,
        pretrained=True,
        amp=True,
        optimizer="AdamW",     # ★ 与版本 A 的区别
        lr0=0.001,             # ★ 与版本 A 的区别
        plots=False,
        verbose=IS_MAIN,
    )
```

> **切换方法**：只替换版本 A 中 `train_new()` 的 `model.train(...)` 为上面这段。其余代码不变。
>
> `train_resume()` 中不加 `optimizer`——续训时 optimizer 从 checkpoint 恢复。

### 6.3 两版本差异对照

| 参数 | 版本 A（默认） | 版本 B（AdamW） |
|------|---------------|----------------|
| optimizer | 未指定（Ultralytics 自动选择，通常 SGD） | `"AdamW"` |
| lr0 | `0.01 × WORLD_SIZE` = 0.02 | `0.001` |
| 其余参数 | 完全相同 | 完全相同 |

### 6.4 验证实际使用的配置

```bash
cd /data/shared/code/vision-train
conda activate /data/shared/envs/train

python3 -c "
import torch
ckpt = torch.load('runs/detect/target_detection/yolo11m_target_highmap_1024-2/weights/last.pt', map_location='cpu', weights_only=False)
args = ckpt.get('train_args', {})
print('optimizer :', args.get('optimizer', '未指定（默认SGD）'))
print('lr0       :', args.get('lr0', '未指定'))
print('epochs    :', args.get('epochs', '未知'))
print('epoch     :', ckpt.get('epoch', '未知'))
print()
for k in ('optimizer','lr0','lrf','cos_lr','amp','mosaic','mixup','close_mosaic','imgsz','batch','epochs'):
    print(f'  {k:15s}: {args.get(k, \"未设置\")}')
"
```

- `optimizer` 显示 `AdamW` → 用了版本 B
- `optimizer` 显示 `未指定` 或 `SGD` → 用了版本 A

---

## 7. 运行指令

### 7.1 首次训练 / 断点续训

```bash
# 1. 创建 tmux 会话
tmux new -s train

# 2. 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/shared/envs/train
cd /data/shared/code/vision-train

# 3. 先启动 node2（后台）
ssh a@node2 "source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate /data/shared/envs/train && \
cd /data/shared/code/vision-train && \
torchrun --nnodes=2 --nproc_per_node=1 \
  --rdzv_id=yolo11m --rdzv_backend=c10d \
  --rdzv_endpoint=node1:29600 \
  train_yolo11m_distributed.py" 2>&1 &

# 4. 等待 5 秒后启动 node1
sleep 5

torchrun --nnodes=2 --nproc_per_node=1 \
  --rdzv_id=yolo11m --rdzv_backend=c10d \
  --rdzv_endpoint=node1:29600 \
  train_yolo11m_distributed.py 2>&1
```

### 7.2 断线恢复

```bash
# tmux 还在
tmux attach -t train

# tmux 没了，重新执行启动命令
# 脚本会自动检测 last.pt 并续训
```

### 7.3 停止训练

```bash
# tmux 中 Ctrl+C
# 或
pkill -f train_yolo11m 2>/dev/null
ssh a@node2 "pkill -f train_yolo11m" 2>/dev/null
```

---

## 8. GPU 监控

```bash
# 两台机器 GPU 状态
echo "=== node1 ===" && \
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv,noheader && echo "" && \
echo "=== node2 ===" && \
ssh a@node2 "nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader"
```

```bash
# 实时监控
watch -n 2 'echo "=== node1 ===" && nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader && echo "=== node2 ===" && ssh a@node2 "nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader"'
```

```bash
# 训练日志
tail -f /data/shared/code/vision-train/runs/detect/target_detection/yolo11m_target_highmap_1024-2/results.csv

# 最终结果
cat /data/shared/code/vision-train/runs/detect/target_detection/yolo11m_target_highmap_1024-2/resume_final_metrics.json
```

**实测 GPU 状态（训练中）**：
```
node1:  RTX 3060, 69°C, GPU 100%, 显存 8.3/12 GB (76%)
node2:  RTX 3060, 70°C, GPU 100%, 显存 8.0/12 GB (79%)
```

---

## 9. 修改日志与问题记录

### 问题 1：cd 命令拼接错误

**问题**：`cd /data/shared/code/vision-traincd /data/shared/code/vision-train`，两条命令粘在一起导致 `cd: 参数太多`。

**解决**：输入单条 `cd /data/shared/code/vision-train`。

---

### 修改 1：讨论 AdamW 优化器

**背景**：SGD 训练 499 epoch（512px 数据）后 mAP50-95 卡在 53%。

**讨论**：AdamW (lr=0.001) 对小目标框回归可能更强。

**实际状态**：讨论过、写过含 AdamW 的脚本，但**最终写入的脚本未加 `optimizer` 参数**（版本 A）。是否某次运行时手动加过 AdamW，无法从对话确认，需通过 checkpoint 验证。

---

### 问题 2：last.pt 未成功移走

**问题**：`mv last.pt last_v1_499.pt` 后 `ls` 显示 `last.pt` 仍在。

**解决**：
```bash
cd runs/detect/target_detection/yolo11m_target_highmap_1024-2/weights/
mv last.pt last_v1_499.pt
mv best.pt best_v1_499.pt
mv _resume_target_6000.pt _resume_target_6000_v1.pt
```

---

### 问题 3：训练 SIGABRT (exitcode: -6)

**问题**：启动后立即崩溃，报 `exitcode: -6`。

**原因**：启动命令中脚本名写错（`train_yolo_distributed.py` 应为 `train_yolo11m_distributed.py`），node2 未启动，node1 rendezvous 超时。

**解决**：
```bash
pkill -f train_yolo 2>/dev/null
ssh a@node2 "pkill -f train_yolo" 2>/dev/null
# 确保两台机器脚本名、--rdzv_id、--rdzv_endpoint 完全一致
```

**教训**：两台机器的启动命令必须完全一致。

---

### 修改 2：单机 → 双机分布式成功

**结果**：
```
节点数: 2, 全局 Batch: 12, 学习率: 0.01 x 2 = 0.02, 速度: 2.9 it/s
```

**关键修改**：
- 添加 `torch.distributed`
- `device=f"cuda:{LOCAL_RANK}"`
- `lr0 = 0.01 * WORLD_SIZE`
- 仅 rank0 做日志/验证/摘要
- checkpoint 由 rank0 写，barrier 同步

---

### 问题 4：图片被 Roboflow 压缩到 512×512

**问题**：用户原图 2K，但 Roboflow 导出时被压缩到 512×512。

**影响**：
```
目标宽度: 512 x 4%~15% = 20~77 像素
放大到 1024: 插值放大，不增加细节
mAP50-95 理论上限: ~55-58%
```

**解决**：回 Roboflow 重新导出 2K 数据集。

---

### 修改 3：更换 2K 数据集

**操作**：Roboflow 导出 2048×1152 → 替换 train/valid/test → 删除旧 checkpoint → 重新启动。

**验证**：
```
总图片: 8586 张
尺寸:   2048×1152（100%）
```

---

### 修改 4：创建两阶段训练脚本（后弃用）

**设计**：阶段 1（SGD 续训）→ 阶段 2（AdamW 从头）。

**结果**：后决定直接用 2K 从头训练，此脚本未使用。

---

### 问题 5：PIL ModuleNotFoundError

**原因**：在 base conda 环境运行 Python。

**解决**：`conda activate /data/shared/envs/train`

---

### 问题 6：文档不一致（epoch 数、优化器）

**问题**：文档中说 6000 epoch 但实际训练输出 `52/3000`；讨论过 AdamW 但不确定实际是否用了。

**解决**：
- 3000 和 6000 两种都实际运行过，文档并列记录
- 优化器提供版本 A（默认 SGD）和版本 B（AdamW），通过 checkpoint 验证

---

## 10. 训练过程记录

### 10.1 第一轮：512×512 数据集 + SGD

```
图片:     512 x 512 (8586 张)
imgsz:    1024
batch:    6/卡
优化器:   默认（SGD）
```

| Epoch | mAP50 | mAP50-95 | 备注 |
|-------|-------|----------|------|
| 319 | 94.4% | 53.1% | 开始平缓 |
| 499 | 93.4% | 53.3% | 完全平台期，180 epoch 零变化 |

**结论**：512px 图片分辨率是根本瓶颈，mAP50-95 理论上限 ~55-58%。

### 10.2 第二轮：2048×1152 数据集

```
图片:     2048 x 1152 (8586 张)
imgsz:    1024
batch:    6/卡，全局 12
epoch:    3000 / 6000 均运行过
```

| Epoch | box_loss | cls_loss | mAP50 | mAP50-95 |
|-------|----------|----------|-------|----------|
| 6 | 1.672 | 1.063 | 0.868 | 0.417 |
| 15 | 1.579 | 0.917 | 0.871 | 0.412 |
| 46 | 1.489 | 0.829 | 0.921 | 0.487 |
| 52 | 1.472 | 0.826 | 0.844 | 0.447 |

**训练中的 GPU 状态**：
```
node1:  RTX 3060, 69°C, GPU 100%, 显存 8.3/12 GB (76%)
node2:  RTX 3060, 70°C, GPU 100%, 显存 8.0/12 GB (79%)
速度:   2.9 it/s
```

**最终结果**：3000 和 6000 epoch 均已训练完成，但最终 mAP50 / mAP50-95 数值未在对话中记录。请从服务器获取：

```bash
# 查看最终指标
cat runs/detect/target_detection/yolo11m_target_highmap_1024-2/resume_final_metrics.json

# 查看训练曲线
tail -20 runs/detect/target_detection/yolo11m_target_highmap_1024-2/results.csv
```

---

## 11. 关键问题分析

### 11.1 图片分辨率与 mAP50-95 的关系

```
原图 512×512   → mAP50-95 上限 ~55-58%
原图 1024×1024 → mAP50-95 上限 ~70%
原图 2048×1152 → mAP50-95 上限 ~70-80%（当前）
```

512px 中目标仅 20-77 像素宽，IoU=0.75 要求框偏差 < 3-12 像素，几乎不可能。2K 图片中目标细节丰富，上限显著提升。

### 11.2 优化器选择

| 优化器 | lr0 | 适用场景 |
|--------|-----|---------|
| SGD（默认） | 0.01 × WORLD_SIZE | 大多数场景，稳定 |
| AdamW | 0.001 | 小目标、精细框回归可能更好 |

### 11.3 深层问题：标注精度

> 这是一直在讨论训练参数（优化器、epoch、分辨率、分布式），但没有真正问出口的问题：

**"标注本身能撑得起 70% 的 mAP50-95 吗？"**

mAP50-95 = 0.7 意味着框要偏到极小范围。如果人工标注时不同图片之间有 5-10 像素的不一致（人眼标注中非常常见），模型学到的就是模糊目标，怎么训都到不了 70%。

**验证方法**：拿 20-30 张图重新仔细标注，与现有标注对比框的偏差。如果偏差超过目标框大小的 10%，瓶颈在标注而非训练。

---

## 12. 常见问题排查

| 问题 | 原因 | 解决方法 |
|------|------|---------|
| `cd: 参数太多` | 命令粘连 | 输入单条 cd 命令 |
| `exitcode: -6 (SIGABRT)` | 脚本名不一致 / 端口占用 | `pkill -f train_yolo`，检查命令一致 |
| `ModuleNotFoundError: PIL` | 未激活环境 | `conda activate /data/shared/envs/train` |
| `CUDA 不可用` | 驱动/PyTorch | `nvidia-smi` 检查驱动 |
| 只有 1 个节点 | node2 未启动 | 检查两台机器命令完全一致 |
| mAP50-95 卡住 | 分辨率低 | 用 2K 原图 |
| mAP50-95 仍不高 | 优化器/标注 | 试 AdamW / 检查标注精度 |
| 训练中断 | 正常 | 重跑启动命令，自动续训 |
| `last.pt 无 optimizer` | 被 strip | 脚本报错，需从头训练 |

---

## 13. 附录：参数速查表

| 参数 | 版本 A（默认） | 版本 B（AdamW） | 说明 |
|------|---------------|----------------|------|
| PRETRAINED_MODEL | yolo11m.pt | yolo11m.pt | 预训练权重 |
| TARGET_EPOCHS | 3000 或 6000 | 3000 或 6000 | 两种都跑过 |
| IMGSZ | 1024 | 1024 | 从 2K 缩放 |
| BATCH | 6 | 6 | 每卡 |
| 全局 Batch | 12 | 12 | BATCH × 2 |
| LR0 | 0.02 | 0.001 | A: 线性缩放; B: AdamW |
| optimizer | 默认 SGD | AdamW | B 显式指定 |
| WORKERS | 8 | 8 | |
| SAVE_PERIOD | 10 | 10 | |
| PATIENCE | 0 | 0 | 关闭 EarlyStopping |
| CLOSE_MOSAIC | 40 | 40 | |
| SEED | 42 | 42 | |
| amp | True | True | |
| 目标 mAP50 | 90% | 90% | |
| 参考 mAP50-95 | 70% | 70% | |
```

以上为整理自完整对话的最终文档。训练最终指标需要从服务器上补充（`resume_final_metrics.json` 或 `results.csv`）。
