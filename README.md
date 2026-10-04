# DDS-Mamba

## 安装

在项目根目录执行以下命令：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m dds_mamba doctor
```

Linux/WSL2 激活命令为 `source .venv/bin/activate`。安装 GPU PyTorch 时，应按 [PyTorch 官方历史版本页面](https://pytorch.org/get-started/previous-versions/) 选择与驱动匹配的 2.2.2 CUDA 包。

默认 `reference` 后端使用纯 PyTorch 实现 Mamba-1，可以在 Windows、CPU 和 CUDA 上运行。CUDA 扩展后端使用 Linux/WSL2、`mamba-ssm==2.2.2` 和 `--backend mamba_ssm`。切换后端前运行 `python scripts/check_mamba_backend.py` 校验同权重输出和梯度。

## 运行测试

```powershell
python -m unittest discover -s tests -v
python -m dds_mamba smoke --out runs/smoke --device cpu
```

`smoke` 自动生成两段小视频，完成训练、反向传播、参数更新、检查点重载和结果写出。它明确使用测试编码器和缩小的模型，产生的检查点会被正式推理命令拒绝。

## 下载真实预训练编码器

```powershell
python -m dds_mamba download --assets assets
python scripts/validate_real_encoders.py --assets assets --device cuda
```

下载官方 MAE ViT-B/16 和 DINOv2 ViT-S/14，共约 432 MB。下载和加载均验证文件长度与 SHA-256，实际运行还会记录资产哈希。验证脚本使用合成图像检查真实编码器和完整模型的前向、反向计算。

编码器保持冻结；模板 128×128，搜索区域 256×256，身份裁剪 224×224。模板仅编码一次，保留确定的行优先 patch 顺序。正式训练和推理严格加载上述官方预训练权重。

## LaSOT 数据准备

下载原始 Protocol II 数据，将官方 `training_set.txt`、`testing_set.txt` 放在数据集根目录。标准目录：

```text
LaSOT/
  training_set.txt
  testing_set.txt
  car/
    car-17/
      img/00000001.jpg
      groundtruth.txt
      full_occlusion.txt
      out_of_view.txt
```

```powershell
python -m dds_mamba split --root D:/datasets/LaSOT --out data_manifests
```

生成 `train.json`、`dev.json`、`test.json`，分别对应 896、224、280 个序列。默认采用确定性的 SHA-256 划分算法，输出清单按序列 ID 排序。指定验证序列清单的参数为 `--validation-list dev_sequences.txt`。程序检查序列数量、训练/测试重叠、图像目录和标注长度，发现缺失序列会报错。

## 训练

```powershell
python -m dds_mamba train --config configs/train.json --assets assets --train-manifest data_manifests/train.json --dev-manifest data_manifests/dev.json --out runs/seed_2024 --seed 2024 --device cuda
```

Linux CUDA 扩展运行时增加 `--backend mamba_ssm`。也可使用 `configs/train_cuda.json`。

默认训练设置：连续 16 帧；每轮 8960 个 clip；40 轮；batch size 1；AdamW；初始学习率 2e-4；权重衰减 1e-4；5 轮预热后余弦衰减；梯度裁剪 1；种子 2024。数据增强包含尺度 [0.8,1.2]、平移不超过裁剪宽度的 0.2、亮度/对比度/饱和度 ±0.2、水平翻转概率 0.5，无时间反转。

训练由在线控制器生成搜索区域；标签只用于损失与辅助 teacher/负样本裁剪。不会用真实框覆盖位置状态、外观状态、Kalman 后验或记忆库。teacher 概率按前 20 轮的半余弦计划降至零，teacher 和负样本分支读取停止梯度的状态与上下文，均不提交状态。

程序写出 `last.pt`、`best.pt`、`history.json` 和 `run.json`。检查点不重复存储冻结编码器，记录其哈希，包含优化器、随机数状态、配置与数据清单哈希。

```powershell
python -m dds_mamba train --config configs/train.json --assets assets --train-manifest data_manifests/train.json --dev-manifest data_manifests/dev.json --out runs/seed_2024 --seed 2024 --device cuda --resume runs/seed_2024/last.pt
```

`--max-steps 1` 将每轮训练限制为一个 clip，用于快速检查数据和训练流程。

验证集选模使用原始输出框的 IoU 积分，按序列等权平均并排除初始化帧。该诊断保留空框的原始输出；官方评估器执行其自身的无效框处理规则。指标定义见 [方法说明](docs/METHOD.md) 和 [评估流程](docs/OFFICIAL_EVALUATION.md)。

三种子训练：

```powershell
python scripts/train_three_seeds.py --train-manifest data_manifests/train.json --dev-manifest data_manifests/dev.json --assets assets --out runs/three_seeds --backend reference --device cuda
```

## 推理与视频跟踪

```powershell
python -m dds_mamba predict --manifest data_manifests/test.json --checkpoint runs/seed_2024/best.pt --assets assets --out predictions/lasot --device cuda
```

只读取首帧初始化框，不在推理中读取后续标签。每个序列输出一个 `x,y,w,h` 文本文件与逐帧诊断文件。恢复确认之前的 incoming-lost 帧写出零框；第三个连续弱 incoming-active 帧仍输出运动预测，下一帧才以 lost 模式处理。恢复帧保留长时外观状态和 QACU 统计，记忆只老化不写入。

`run.json` 记录 incoming-active/lost 分别计时及总 FPS，排除初始化、前 100 次 warm-up、图像读取/解码、CPU→GPU 传输，包含裁剪、归一化、编码器和控制器计算。

用户视频需要先解码为按帧编号排序的图片，再建立 [自定义清单](examples/custom_manifest.json)，填写图片目录、总帧数和首帧框。推理不需要后续标注；视频帧的分辨率应保持一致。

## 其他基准数据接口

```powershell
python -m dds_mamba prepare --root D:/datasets/Anti-UAV300 --benchmark anti-uav300 --split test-dev --out data_manifests/anti_test.json
python -m dds_mamba prepare --root D:/datasets/WebUAV-3M --benchmark webuav --split Test --out data_manifests/web_test.json
python -m dds_mamba prepare --root D:/datasets/DUT-VTUAV --benchmark vtuav --split test_LT --out data_manifests/vtuav_test.json
```

这些命令读取解码后的 RGB 图像目录，`--split` 指定数据集的测试目录。VTUAV 保留逐视频帧预测，由官方评估器对齐稀疏标注帧。

换用对应 `--manifest` 运行同一个 `predict` 命令。VOT-LT2020 的在线工具包接口在 `scripts/vot2020_runner.py`，依赖 2020 版 Python/TraX helper；输出置信度采用 incoming-mode 的候选排名分数，空框对应 0，详见 [方法说明](docs/METHOD.md)。

## 评估

```powershell
python -m dds_mamba diagnostic-evaluate --manifest data_manifests/test.json --predictions predictions/lasot --out runs/raw_diagnostics.json
python -m dds_mamba export --manifest data_manifests/test.json --predictions predictions/lasot --format lasot-txt --out official_results
```

第一个命令计算原始预测诊断。第二个为 LaSOT 评估工具生成 `DDS-Mamba_tracking_result/<sequence>.txt`。AUC、TA、cAUC、MSR/MPR 和 VOT F-score 的调用方式见 [评估流程](docs/OFFICIAL_EVALUATION.md)。

从各次运行的官方逐序列 CSV（列名含 `sequence` 和指标列）计算 10000 次序列 bootstrap：

```powershell
python -m dds_mamba bootstrap --input seed2024.csv --input seed2025.csv --input seed2026.csv --metric success_auc --out runs/auc_ci.json
```

配对方法差异可以增加三个对应 `--other` CSV。程序核对序列 ID，先跨种子按序列平均，再重采样序列。

## 项目结构

| 文件 | 作用 |
| --- | --- |
| `dds_mamba/model.py` | 模板条件投影、位置/外观分支、空间门控、置信图、身份投影 |
| `dds_mamba/mamba.py` | Mamba-1 参考实现、可选 CUDA 后端、正确梯度的有界投影 |
| `dds_mamba/controller.py` | QACU、active/lost、弱帧计数、连续恢复确认 |
| `dds_mamba/memory.py` | RFMB 检索、可靠度与年龄效用、低效用替换 |
| `dds_mamba/kalman.py` | 8 维归一化运动状态、固定 Q/R、创新门限 |
| `dds_mamba/losses.py` | 框、focal、去相关、时间、身份、范数和中心对齐损失 |
| `dds_mamba/training.py` | 在线 16 帧展开、增强、teacher/负样本监督、训练与续训 |
| `dds_mamba/data.py` | LaSOT/UAV 数据集与清单验证 |
| `configs/` | 训练参数和 CUDA 配置 |
| `tests/` | 机制、梯度、数据与数学一致性检查 |
| `docs/METHOD.md` | 模型结构、控制器和训练配置 |
