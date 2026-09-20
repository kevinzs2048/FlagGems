# Qwen-Image-2.1-0920 W8A8 重量化记录

## 模型与交付物

- 输入：`/Users/kevin/Qwen-Image-2.1-0920`
- 输出：`/Users/kevin/Qwen-Image-2.1-0920-W8A8-PerChannel-112`
- 原始权重目录和上一版量化模型未修改。
- Transformer 仍为 32 blocks、hidden 4096、32 heads × 128；297 个原始
  tensor 的名称、维度和 dtype 与上一版一致。
- 新旧 Transformer 分片的 SHA-256 不同。抽查 block 2 Q projection，
  16,777,216 个权重值中有 8,992,218 个与旧版不同，确实是新的权重数值。
- 新配置移除了旧版未使用的 `causal_block`；VAE 配置的
  `scale_factor_spatial` 从 8 修正为 16。现有参考 pipeline 已固定使用 16。

## 量化规则

沿用同一 `same112` 策略：block 索引 2～29 的 Q/K projection 和 MLP
proj/gate，共 112 个 Linear。权重为 per-output-channel 对称 INT8，
FP32 scale，zero point=0；激活运行时动态 per-token 非对称 INT8，
FP32 scale，KleidiAI 舍入规则。INT32 累加，FP32 缩放，BF16 输出。

没有将策略换成 G32、没有二次量化、没有扩大到 196 个 Linear。
其余 Transformer 权重保持 BF16。Text Encoder 和 VAE 没有做 INT8 量化。
VAE 原文件是 FP32，独立拷贝保留；推理仍按既有 BF16 加载、FP32 CPU 卷积路径执行。

| 项目 | 大小（十进制 GB） |
| --- | ---: |
| 原始 Transformer tensor payload | 约 14.23 |
| 量化 Transformer tensor payload | 10.475823104 |
| 整个独立模型目录 | 约 29.38 |

整个目录包含实际的 Text Encoder、VAE、processor 和 scheduler 文件，
没有指向其他模型目录的软链接。核验报告记录了源 Transformer、输出分片和
每个未量化组件文件的 SHA-256。所有输出 tensor 都经过回读逐元素核验。
导出耗时 42.18 秒；完成标记只代表格式和文件导出成功。

## 数值检查

在新模型生成的 512×512、40 步 BF16 SME2 轨迹上，取第 1、20、40 步输入。
各分支接收相同 latent、prompt embeddings 和 timestep，均重新计算前缀。
以下是 Transformer 预测相对 L2 误差，不是最终图片的画质损失百分比。

| 比较 | 第 1 步 | 第 20 步 | 第 40 步 |
| --- | ---: | ---: | ---: |
| BF16 SME2 vs BF16 数值/FP32 计算 | 0.676% | 0.413% | 1.139% |
| W8A16 vs BF16 数值/FP32 计算 | 0.676% | 0.520% | 2.353% |
| W8A8 vs BF16 数值/FP32 计算 | 1.394% | 0.778% | 5.845% |
| W8A8 vs W8A16 | 1.407% | 0.712% | 5.434% |

保留 BF16 的 Linear 在量化误差比较中统一使用 ATen FP32 计算，避免混入
BF16 SME2 后端差异。表中各项不能直接相加或相减。

四个代表性 Q/MLP Linear、三个时刻，共 12 组真实输入/权重的内核检查通过：
A8 codes 和 zero point 与 Python 参考完全一致，scale 在容差内一致；
完整 K 维的整数 dot 与 PyTorch INT32 oracle 比较通过，FP32 输出的最大
相对 L2 误差为 **7.7782×10⁻⁸**。每组抽查两个输入行、64 个分布于完整 N
维的输出通道，未声称穷举所有算子元素。

提交后 **36 项测试通过**，覆盖导出/重载、文件及 dtype 保留、失败标记、
权重量化、激活量化、整数 oracle、内核尾块及边界情况。

## 完整推理回归

使用同一 case 001 英文提示词、seed 42、CFG=1、前缀 KV cache，纯 CPU。
采用现有优化 W8A8 profile，主线程数 18、MLP workers 12。
计时包含文本编码、40 步去噪、VAE、PIL 转换，不包括加载/预打包、warmup、
初始 latent 构造和 PNG 保存。

| 分辨率 | 新权重 | 原有同机记录 | W8 GEMM 调用数 |
| --- | ---: | ---: | ---: |
| 512×512，40 步 | 158.760 秒 | 158.657 秒 | 4480 = 112×40 |
| 1024×1024，40 步 | 811.285 秒 | 820.253 秒 | 4480 = 112×40 |

旧记录来自之前的运行，并非交错配对测试，不能把时间差归因于权重更新。
512 的 BF16/W8A8 图片均能生成卧室、床和四个云形挂饰，但细节不同，
目标文字没有被完全正确生成。1024 W8A8 图片中的四个云形挂饰和床铺正常，
右侧文字基本可辨，但字形仍有瑕疵。两张 W8 图片的尺寸、像素和 PNG 哈希均
与推理 JSON 核对通过；没有进行 1024 的 BF16 图片对照。
该单提示词测试不能代替全面画质评测。

## 代码与证据

- 导出器：`qwen_image_cpu/export_w8.py`，提交 `d40bc506`。
- 验证器：`qwen_image_cpu/validate_w8.py`，提交 `9c791ea6`。
- 两者都由 FlagGems 的 master 派生工作分支承接，没有修改 PyTorch 或接入 Torch-FL。
- 当前推理内核及 FlagTree-CPU 未修改。
- 方法与复现：[QUANTIZATION.md](QUANTIZATION.md)。
- 文件与源权重校验：输出模型中的 `quantization_report.json`。
- 完整数值报告与输入快照：
  `/Users/kevin/qwen-image-2.1-repos/validation/qwen-0920/precision/`。
- 端到端记录与 PNG：同级的 `512-40.json/png` 和 `1024-40.json/png`。
- 提交后测试：同级的 `postcommit-tests.xml`。
