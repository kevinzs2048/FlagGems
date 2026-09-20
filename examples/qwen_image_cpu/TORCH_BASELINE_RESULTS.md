# Qwen-Image-2.1：纯 PyTorch CPU 性能实测

## 测量口径

Apple M5 Pro，18 CPU 核，64 GiB 内存，PyTorch 2.11.0。
原始 BF16 模型 `/Users/kevin/Qwen-Image-2.1`，无权重量化。
Transformer Linear 权重精确提升到 FP32，计算后返回 BF16 激活；
CPU Attention 和 VAE Conv 使用 FP32 计算。这不是原生 BF16 GEMM，也不是 W8A8。

模型、40 个去噪步骤、CFG=1、种子 42、前缀 KV cache 和 case 001 提示词固定。
每次请求重新编码提示词。计时包括文本编码、全部去噪、VAE 和 PIL 转换，
不包括加载、初始 latent 构造及 PNG 保存。

普通 PyTorch/ATen 算子调用系统 Apple Accelerate；不加载 FlagGems 后端、
Torch-FL、Triton kernel 或我们的 KleidiAI/SME2 扩展。
代码本身由 FlagGems 仓库的 example 承接，生产 W8A8 入口保持原样。

## 已完成的完整请求

| 分辨率 | 步数 | 第一次 | 第二次 | 实测最短 |
| --- | ---: | ---: | ---: | ---: |
| 512×512 | 40 | 290.886 秒 | 299.832 秒 | 290.886 秒 |
| 1024×1024 | 40 | 1376.127 秒 | 1365.092 秒 | 1365.092 秒 |

最低值分别为 **4 分 51 秒**和 **22 分 45 秒**。
两轮范围相对最低值分别为 3.076% 和 0.808%。

最快请求的分项计时如下；VAE 栏计 decoder，余下少量 pipeline 工作计入总时间。

| 分辨率 | Transformer | Text Encoder | VAE decoder | 总时间 |
| --- | ---: | ---: | ---: | ---: |
| 512×512 | 283.422 秒 | 2.872 秒 | 4.553 秒 | 290.886 秒 |
| 1024×1024 | 1335.508 秒 | 2.617 秒 | 26.915 秒 | 1365.092 秒 |

上述“最短”只表示这些完整请求里的最小值，不是 CPU 的物理下界，也不是所有
PyTorch 版本、编译方法和布局的全局最优证明。

512 和 1024 各自两轮图片像素完全一致，已重新读取 PNG 逐字节核验，
同时确认保存的像素哈希与运行记录一致。
单个提示词的图片检查不能代替质量评测集：512 图片生成了四个云形挂饰，
但缺少正确文字；1024 首张生成了五个挂饰且文字不正确。
这些问题在此处未量化的 BF16 权重路线中也存在，不能仅凭该案例归因于 W8。

## 配置筛选与正确性

- 单路 Linear 的两轮完整模型短测：4/8/12/18 线程，稳定前向中位数分别为
  10.423/9.667/9.775/9.411 秒（512，三步请求，排除各次首步）。
- 真实 Q/MLP 权重，共 162 组矩阵乘测量；比较 1024/4096 行、4/8/18
  PyTorch 线程、1/2/3/4/6 个并发工作线程、按 M 或 N 切分。
  所有测量的 BF16 输出均与单路参考逐元素一致。
- 1024 全模型短测选择：6 个 Linear 工作线程、按 N 切分；
  8 线程稳定前向 32.250 秒，18 线程 29.913 秒。完整请求采用 18 线程。
- 线程池仅执行普通 `torch.nn.functional.linear`，没有自定义 native kernel。
- 提交后 21 项回归通过：Linear 数值、并发分块、参考 Transformer 前缀缓存、
  以及既有推理入口测试。
- 修正加载顺序，避免已替换 BF16 模块的 Python 引用环延迟释放内存。

系统并非隔离的空闲实验环境，macOS 后台服务仍在运行。测试前已有 swap，
测试过程中系统级 swap 用量发生增长；该指标包含其他进程，不能全部归因于模型。
原始系统快照与最终汇总一并保留，没有把本次最低值表述成无内存压力的硬件极限。

## 计算量

32 个 block，hidden=4096，MLP hidden=12288。1024 图像包含 4096 个目标 latent token。
仅每个 block 的四个 Attention Linear 和三个 MLP Linear，每步约 57.175 TFLOP；
再计入图像 token 之间的 QK/AV，每步约 65.971 TFLOP，40 步约 2.639 PFLOP。
这里一次乘加计两次操作，尚未计入文本前缀、Norm、VAE 等额外计算。
这解释了为何不能用权重大小除以内存带宽来估算完整推理时间。

## 复现与证据

- 运行方法：[TORCH_BASELINE.md](TORCH_BASELINE.md)。
- 实测代码提交：`7ab50ea4`（基于 `2e68ef81`）。
- 参考 Diffusers：`9511e982f22ea8fcc64faf277d90b3a21f8a83f6`。
- 完整计时、每步耗时、图片路径及后端审计：
  `/Users/kevin/qwen-image-2.1-repos/validation/torch-limit-full/report.json`。
- 图片：同目录 `torch-512-40-01.png`、`torch-512-40-02.png`、
  `torch-1024-40-01.png`、`torch-1024-40-02.png`。
- 代码哈希：同目录 `provenance.json`。
- 独立图片和源文件哈希核验、最短/最长/中位数：同目录 `verified-summary.json`。
- 原始矩阵乘数据：`validation/torch-gemm-probe.json`。
- 原始线程扫描：`validation/torch-limit-scan/report.json`。
- 回归记录：`validation/torch-baseline-postcommit-tests.xml`。

此前的 W8A8 生产路径性能见 [INTEGRATION_REPORT.md](INTEGRATION_REPORT.md)。
与这里对比时必须注明：量化精度和算子后端都不同，不能当作单算子加速比。

| 路线 | 512×512，40 步 | 1024×1024，40 步 |
| --- | ---: | ---: |
| 本次纯 PyTorch，原始 BF16 数值 / FP32 计算 | 290.886 秒 | 1365.092 秒 |
| 此前同机优化路径，112 层 W8A8 / 其他 BF16 | 158.657 秒 | 820.253 秒 |

第二行来自已有回归记录，不是与本次交错进行的配对测试。
