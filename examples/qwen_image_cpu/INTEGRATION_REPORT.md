# Qwen-Image-2.1 CPU 仓库迁移与端到端回归

日期：2026-09-20。测试机：Apple M5 Pro，18 核，64 GiB 内存，macOS arm64。
本次工作为代码归属整理及指定上游基线移植，没有重新量化权重。

## 代码交付

| 仓库 | 上游基线 | 执行回归的实现提交 | 负责内容 |
|---|---|---|---|
| FlagGems | master `df6fcaad` | `6304eecf` | native 算子、Triton W8 候选、模型融合适配、推理入口和测试 |
| FlagTree-CPU | triton_v3.7.x `2c35990a` | `de687051c` | SME2 streaming ABI、full-K ZA 累加、512-bit specialization、Apple 非 streaming SVE 检查 |

两个工作分支分别为 `codex/qwen-image-cpu-master` 和
`codex/qwen-image-sme2-3.7`；本地已提交，尚未推送或创建 PR。

运行框架是 **PyTorch CPU**，默认算子为 FlagGems 承接的 native SME2/KleidiAI。
FlagTree-CPU 候选经过测试，但没有自动替换默认 native GEMM。
PyTorch、Torch-FL、pytorch-plugin-FL 均无改动；不需要 Torch-FL 注册设备。

模型相关逻辑位于 `examples/qwen_image_cpu/qwen_image_cpu/`；native 算子位于
`src/flag_gems/csrc/arm/qwen_image/`，JIT 构建入口位于 `_arm/quantized_linear/image_sme.py`。
`run.sh` 仅设置环境并启动 Python。原项目中的 16 个 native 源文件与迁移后
逐文件 SHA256 完全一致，未改变算术实现。构建缓存独立于源码目录。

## 精度与运行配置

- Transformer 的 112 个 Linear：W8A8，same112 policy。
- W8：symmetric per-output-channel INT8，FP32 scale；A8：动态 asymmetric per-token INT8。
- 其他权重 BF16；部分浮点计算/累加使用 FP32。不是全 BF16 计算，也不是 W4。
- seed 42，CFG 1，prefix KV cache 开启，batch 1，PyTorch 18 线程、MLP 12 workers。
- Prompt：测试集 case 001，四盏云形纸灯、右侧写有 “Sweet Dreams”。
- 参考 Diffusers：用户提供 ZIP 的 revision `9511e982f22ea8fcc64faf277d90b3a21f8a83f6`。

## 提交后的算子与编译器回归

- FlagGems 示例测试：**361 passed**，48.69 秒。
- FlagTree-CPU：先执行 `make`，再运行 target-info、SME2 streaming、SVE2/I8MM 测试：
  **43 passed，2 skipped**，10.57 秒。跳过项分别需要 Cortex-A720 store-pair codegen 和 128-bit SVE2/I8MM lowering，
  本机不满足对应测试前提。
- 编译器从本次分支重新构建；验证实际导入该 checkout 的 `libtriton.so`。
- Python 新增代码通过本仓库配置对应的 flake8 检查。
- 回归覆盖整数 oracle、packing、BF16 rounding、融合、worker 状态恢复、checkpoint
  格式拒绝、native/Triton 数值对比和 CLI 参数验证。

## 端到端计时范围

驻留端到端从 pipeline 调用开始，到返回图片结束，包含 Text Encoder、全部
40 步 denoise、VAE decode 和 PIL 输出构造。加载/预打包、2 步预热、文件写入、
哈希计算独立于此时间。初始 latent 的构造也在驻留计时之外。

加载时间是在本机权重文件和 native 编译缓存已经可用的条件下测得，不能当作
新机器安装、下载、首次编译的总耗时。后台批量出图暂停；1024 测速没有与
stable-diffusion.cpp 编译并行。常规 macOS 服务仍会产生运行噪声。

最终要求是 **1024×1024、40 步**；512×512、40 步用于功能冒烟测试。
此前开始的提交后 512×512、50 步测试已按用户新要求中止，不能列为完成结果。

## 完整生图结果

| 用途 | 分辨率 / 步数 | 驻留端到端 | 加载及预打包 | W8 GEMM 调用 |
|---|---|---:|---:|---:|
| 功能回归 | 512×512 / 40 | 158.657 秒 | 13.691 秒 | 4480 |
| 最终规格 | 1024×1024 / 40 | 820.253 秒（13 分 40 秒） | 9.405 秒 | 4480 |

每次都确认 112 个 W8 模块实际执行。1024 结果与迁移前 case 001 在相同
prompt、seed、分辨率、步数下 **RGBA 全部像素完全一致**：

```text
2dc35eb5cf9559b245227d1f156948fd3387ec05fa72d8544af63402a4033db6
```

1024 去噪日志约 795 秒，约占驻留端到端的 97%；最终时间还包含文本编码、
VAE 和输出构造。这里只能确定去噪为主体，不据此把所有去噪耗时等同于 GEMM。

### 同机、同规格旧路径复测

额外使用迁移到指定上游分支之前的路径，重跑相同 prompt、seed 的 512×512、
40 步请求，环境和 warmup 配置一致：

| 路径 | 驻留端到端 |
|---|---:|
| 旧基线路径 | 158.416 秒 |
| 本次指定分支实现 | 158.657 秒 |

差异 **+0.15%**，图片 RGBA 全像素完全一致。这轮未观察到明显性能回退。
每条路径各测一次，不能作为统计显著的加速或零退化保证。1024 新旧图片也完全
一致；1024 耗时相对于历史 case 001 高约 7.3%，相对于历史批量中位数高约
0.6%，仍在历史范围内；没有把这种非受控历史比较宣称为严格 A/B 通过。



## 结果解读边界

单个 prompt/seed 的像素回归不能证明 W8 量化无损，也不能代表整个测试集质量。
已看到 512 图的英文文字不准确；原有 1024 case 001 参考图同样有文字和数量偏差。
这些需要单独的 BF16/量化质量评测，不能用“能出图”代替 prompt 遵循率结论。

历史 1024×1024、40 步的 57 张图驻留时间范围约 764.5–855.4 秒，中位数
815.1 秒；其中 case 001 为 764.5 秒。这是历史运行范围，不是严格受控、同一时段
交替 A/B 的性能实验。不同 prompt 和机器运行状态会影响比较。

## stable-diffusion.cpp 对照

最新测试的 upstream master `1330ceba` 已在本机编译，CPU SME2、KleidiAI 和
Apple BLAS 可用，Metal 关闭。BF16 和 W8A8 两种 2.1 权重都在模型识别阶段
失败，尚无生成图片或可比较的推理耗时。

详见 [stable-diffusion.cpp 测试报告](STABLE_DIFFUSION_CPP_REVIEW.md)。
