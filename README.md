# 世界模型辅助的视觉语言导航决策优化（VLN）

<a name="demo-videos"></a>

## 演示视频

两边使用相同的 R2R-CE 导航任务、起点和指令。画面取自两种策略在 Habitat 中本次运行的第一视角 RGB 相机，左侧是优化版（成功），右侧是原版 g3D-LF（失败）。先结束的一侧会停在最后一帧。点击播放器即可观看。

### 任务 207 · 经过壁炉，沿长走廊走到下一组门前停下

停止时距目标 **0.26 米**（优化版） / **22.66 米**（原版）。

https://github.com/user-attachments/assets/400d30ad-77e3-4234-96ff-9787c36d18d4

### 任务 1643 · 经过楼梯进入起居室，右转后在边桌旁停下

停止时距目标 **1.15 米**（优化版） / **8.87 米**（原版）。

https://github.com/user-attachments/assets/43ba5263-c68e-4f5a-8b1b-4e84967be1f3

### 任务 416 · 在视力表处左转，再右转进入厨房，在水槽旁停下

停止时距目标 **0.35 米**（优化版） / **8.46 米**（原版）。

https://github.com/user-attachments/assets/be326ab2-1547-4b7a-b9f5-9757763b8a36

## 项目简介

本项目基于全景版 [g3D-LF](https://github.com/MrZihan/g3D-LF)，让智能体根据语言指令在室内选路，并在目标附近停下。项目用整段导航的结果优化局部决策，并根据当前观测自适应地调整世界模型预测对决策的影响。

## 方法

- **GRPO 后训练。** 对同一条指令和同一起点采样多条轨迹，用任务是否成功、路径效率和终点到目标的距离计算回报，再用组内相对优势更新策略。裁剪目标和 KL 惩罚约束更新幅度。
- **可靠性融合。** 用一个仅 449 个参数的 MLP，从候选评分和预测视觉特征中提取 12 项统计量，动态调节全局评分和前瞻评分的融合权重，影响路点选择和停止决策。先用候选排序一致性训练融合模块，再用已记录的轨迹回报做离线微调。三维特征场保持冻结。

## 评估结果

在 Habitat 仿真中评估 R2R-CE `val_unseen`，两组历史评估覆盖相同的 11 个场景、1,839 条导航任务。SR 为成功率，SPL 为按路径效率加权的成功率。

| 模型 | SR ↑ | SPL ↑ |
| --- | ---: | ---: |
| 原版 g3D-LF（官方检查点） | 60.20% | 50.85% |
| GRPO + 可靠性融合 | **61.45%** | **51.95%** |
| 变化 | **+1.25 个百分点** | **+1.10 个百分点** |

## 代码

仓库收录了 PyTorch 核心实现。完整运行需要上游仿真环境、数据和模型权重，接口和评估设置见[集成说明](docs/integration.md)。

- [GRPO 训练目标](vln_policy/grpo.py)
- [候选特征、可靠性融合与离线微调目标](vln_policy/reliability.py)

## 参考与致谢

导航基座来自 [g3D-LF（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_g3D-LF_Generalizable_3D-Language_Feature_Fields_for_Embodied_Tasks_CVPR_2025_paper.pdf)。演示使用基于 Matterport3D 场景取得的导航结果，相关条款见 [Matterport3D 数据使用协议](https://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf)。
