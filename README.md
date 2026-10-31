# p01 原规模单案例

用途：按发布代码的网络规模完成一次学习与一次测试；不是全部论文实验。

## 固定设置
- PN 840；KC 与 KC-A 各 4000；MBON 与 MBON-A 各 1。
- 学习 p01.mat 的前半段，5567 ms；固定学习权重，测试同文件完整序列，11134 ms。
- 保留原始神经元参数、连接生成器和抑制性乘法 STDP。
- 时间步 1 ms；time_scale_factor=10 给硬件更多处理时间，不改变模型时间步或 STDP 时间参数。该设置是否足够仍由本次计数和记录检查决定。
- 不遍历其他数据，不增加另一组硬件实验。

## 平台提交
- Hardware: SpiNNaker
- 来源：包含 run.py、spynnaker.cfg、src/ 两个模块和 data/p01.mat 的仓库或 ZIP。
- Command: run.py --run
- sPyNNaker version: 7.4.1（运行入口也会核对实际安装版本，版本不符时在申请硬件前退出）
- 其他软件版本覆盖项、Extra 各项先留空；本入口不依赖 Elephant 绘图分析。
- 平台需要能访问代码源。浏览器已登录 GitHub，不代表平台有权限下载私有仓库；不要把登录令牌放进仓库 URL。

## 默认只检查输入
在工作根目录 /home/zeqome/project_connectome_snn/lu1 执行：

```bash
scratch/mb-sequence-learning/direct_spinnaker/venv/bin/python -B code/mb-sequence-learning/run.py --data scratch/mb-sequence-learning/data_check_20261024/recovered_mat/840/p01.mat
```

这不导入模型，也不申请硬件。--run 入口为平台准备；不要拿它直接重建服务器直连登录流程。

## 输出与判读
outputs/learning/ 和 outputs/testing/ 分别保存权重、脉冲、电压、原始 KC-A 电压时间戳、核心计数及 checks.json；outputs/run_summary.json 保存阶段状态和实际软件版本。
结束或异常退出时生成 outputs.zip。平台归档和下载能力仍需实际作业验证；不能只在日志中打印成功。

学习阶段电压时间戳缺失、MBON 电压长度不符、核心计数缺失或存在过载/丢包/延迟等异常时停止，不继续测试。若运行失败，先检查 checks.json、core_provenance.csv 和作业日志，不盲目重跑。

COMPLETED 只表示本案例流程及检查完成，不能据此声称复现了路线识别能力。后续还需要未学习对照和路线标签核实。

论文：https://doi.org/10.1126/scirobotics.adg3679
作者代码与数据：https://zenodo.org/records/8289547
本仓库的 p01.mat 为作者发布数据的选定样本；没有包含服务器其他项目、凭据或历史实验结果。


## 26 文件批量学习与测试
- 入口为 batch.py。默认只核查输入；平台 Command 为 batch.py --run。
- 现有 26 个 MAT 文件按文件名排序。每个文件独立学习前半段，然后固定该次学习的权重，测试全部 26 个完整文件；合计 26 次学习、676 次测试。
- 沿用 run.py 的模型参数、版本检查、记录检查、time_scale_factor=10 和异常释放；任一次异常停止整个批量任务，不自动重试。
- 累计模型时间约 2.01 小时；放慢 10 倍后仅模拟约 20.1 小时，不包含 702 次部署、读取和归档。平台单作业时限及实际配额消耗尚未验证，5000 core-hours 不保证足够。
- 每次学习的输出在 outputs/learn_p01/ 等目录；学习权重只保存于 learning/kc2kc_weights.npy。testing/p01/ 等子目录保留测试神经活动、摘要、原始时间戳和核心检查，不重复保存固定权重。
- 每个学习文件完成或失败后生成 outputs/learn_p01.zip 等独立 ZIP。outputs/batch_summary.json 汇总已完成的学习文件；各目录 run_summary.json 记录学习和各测试阶段完成情况。
- 部分文件半程时长为 x.5 ms，记录完整性以模拟器实际完成的 1 ms 时间网格检查，不对预期点数直接向下取整。
- 这是现有数据的交叉学习与测试，不包含长路线数据、SeqSLAM/PM、ROC/AUC 分析，不能称为论文全部结果复现。p00.mat 不在现有数据中，不虚构缺失样本或路线标签。
