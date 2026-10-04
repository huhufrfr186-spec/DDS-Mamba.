# 官方基准评估接口

项目写出原始逐帧框、轨迹诊断和计时日志。本文件说明正式基准评估的输入格式和调用流程；指标由数据集官方评估器计算，并保存评估器版本、配置、原始预测、日志、检查点和数据清单哈希。

## LaSOT Protocol II

[官方 MATLAB 工具](https://github.com/HengLan/LaSOT_Evaluation_Toolkit)。

其 `utils/eval_tracker.m` 读取 `<tracking_results>/<tracker>_tracking_result/<sequence>.txt`，不是必须使用 MAT 结构。命令：

```text
python -m dds_mamba export --manifest data_manifests/test.json --predictions predictions/lasot --format lasot-txt --out official_results
```

将生成的 `DDS-Mamba_tracking_result` 放入官方工具的 `tracking_results`。在工具的 `utils/config_tracker.m` 注册 `struct('name','DDS-Mamba','publish','DDS-Mamba')`，主脚本选择 280 序列测试设置，再运行 `run_tracker_performance_evaluation.m`。依照工具说明分别设置 `norm_dst` 计算 precision 和 normalized precision。保留官方 `calc_seq_err_robust` 的无效框处理，不提前将本项目空输出改为最近框。

数据用原始 1400 序列 Protocol II，不含后续 150 序列 extension；训练的 224 development 序列也不是官方测试 280 序列。

## Anti-UAV300 RGB

[官方工具与数据发布](https://github.com/ZhaoJ9014/Anti-UAV)。

读取 300 发布版的 RGB 首帧框进行单目标跟踪。数据应解码为 `<split>/<video>/RGB/*.jpg`（或直接 RGB 图片）并包含 `RGB_label.json`。不能使用 410/600 红外版替代。不同发布版的 evaluator 可能读取 CSV、JSON list 或包含 `res` 的 JSON 对象；项目有 `csv` 与 `anti-uav-json` 导出，后者提供 `{"res": [...]}`，应根据你实际的官方 reader 确认格式。

缺失目标统一写 [0,0,0,0]。TA 的存在感知评分由官方工具计算。

## WebUAV-3M

[官方发布与工具入口](https://github.com/983632847/WebUAV-3M)。选择 RGB Test 的 780 视频，保留 `absent.txt`。不读语言或音频。

官方文档列出的评估入口包括 `WebUAV-3M_Overall_Evaluation.py`、`WebUAV-3M_Accuracy_Evaluation.py` 等。将逐视频结果按所下载工具版本的 reader 格式放进 `results/Baseline_Results`，计算 Pre/nPre/AUC/cAUC/mAcc。

`mat-struct` 导出 MATLAB 结构，`csv` 导出逐帧表格。选择与评估器 reader 对应的格式。

## DUT-VTUAV-V

[官方发布入口](https://github.com/zhang-pengyu/DUT-VTUAV)。

使用官方可见光 long-term `test_LT`，通过官方链接下载 MATLAB 工具，放置结果到 `BB_results_RGB`，设置 `GenerateMat_LT_RGB_only.m` 的 `basePath`，生成报告，再用 `plot_LT_RGB_only.m` 计算 MSR/MPR。项目 `vtuav-txt` 导出用制表符分隔逐帧 xywh；发布版若需要其它序列命名/文件后缀，应按其 reader 配置。

稀疏标注由官方工具按标注帧索引对齐。项目推理使用首帧初始化框，逐帧输出预测结果。

## VOT-LT2020

[官方工具](https://github.com/votchallenge/toolkit) 与 [integration helper](https://github.com/votchallenge/integration)。须使用 VOT2020 long-term stack 与当时支持 `report(region, confidence)` 的 Python helper。

工具中的 tracker command 指向：

```text
python /absolute/project/scripts/vot2020_runner.py --checkpoint /absolute/best.pt --assets /absolute/assets --device cuda
```

wrapper 接收首帧区域和图片，后续无重新初始化，输出区域与 confidence。置信度采用 incoming-mode 的候选排名分数，空框为 0；该配置见方法说明。PR 曲线与 F-score 由 VOT2020 long-term stack 生成。

## 逐序列统计

官方 evaluator 导出 CSV 后用项目 `bootstrap` 命令进行种子平均和配对序列 bootstrap。输入单位由 evaluator 决定（例如 [0,1] 或百分比），代码保留单位，不擅自乘 100。
