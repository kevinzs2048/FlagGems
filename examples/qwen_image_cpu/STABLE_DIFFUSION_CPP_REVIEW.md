# stable-diffusion.cpp 同机可运行性测试

测试日期：2026-09-20。目标是 **Qwen-Image-2.1，1024×1024，40 步，纯 CPU**。

## 结论

当前测试的 upstream master **不能加载本次 Qwen-Image-2.1**，因此没有出图，
没有可用于比较的端到端性能数字。初始化退出耗时不是图片生成耗时。

对 BF16 原始 Transformer 和 same112 W8A8 Transformer 各做了一次实际加载。
两者都能够读到 safetensors 分片索引和模型 tensor 清单，但在架构识别阶段
返回退出码 1：

```text
get sd version from file failed: ''
new_sd_ctx_t failed
```

命令使用了明确的 Transformer/Text Encoder 分片索引及 VAE 文件，避免把
Diffusers 根目录加载器只查找 `unet/` 的问题误认为量化格式问题。
BF16 同样失败，说明先转 GGUF 不能补齐缺失的模型架构实现。

## 构建与来源

- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)：
  master `1330cebae8f2ba99249df846cc0c9444fcbd4308`，2026-09-19。
- 该提交固定的 GGML：`c6632cd905401abc58b6f5cdd52d228aa7ca1b88`。
- 未修改源码。AppleClang / Release，Metal OFF，CPU KleidiAI ON，Apple BLAS ON。
- 可执行文件运行时报告 `NEON=1, SME=1, SME2=1, ACCELERATE=1, KLEIDIAI=1`。
  这证明构建具备对应后端能力，不表示不受支持的模型已经执行了这些内核。
- 没有把已有的旧版 2512 benchmark 当作 2.1 对照。

```bash
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSD_METAL=OFF -DGGML_METAL=OFF -DGGML_METAL_EMBED_LIBRARY=OFF \
  -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple -DGGML_CPU_KLEIDIAI=ON \
  -DSD_WEBP=OFF -DSD_WEBM=OFF -DSD_SERVER_BUILD_FRONTEND=OFF
cmake --build build-cpu --target sd-cli -j 8
```

加载探测参数（同一参数分别应用于 BF16 与 W8 checkpoint）：

```bash
build-cpu/bin/sd-cli \
  --diffusion-model "$MODEL/transformer/diffusion_pytorch_model.safetensors.index.json" \
  --llm "$BF16_MODEL/text_encoder/model.safetensors.index.json" \
  --vae "$BF16_MODEL/vae/diffusion_pytorch_model.safetensors" \
  -p "$PROMPT" -W 1024 -H 1024 --steps 40 --cfg-scale 1 \
  --sampling-method euler --seed 42 --rng cpu --diffusion-fa -t 18 \
  -o result.png -v
```

## 源码定位

[模型识别](https://github.com/leejet/stable-diffusion.cpp/blob/1330cebae8f2ba99249df846cc0c9444fcbd4308/src/model_loader.cpp#L426)
使用旧版 `transformer_blocks.0.img_mod.1.weight` 等特征。
[Qwen 实现](https://github.com/leejet/stable-diffusion.cpp/blob/1330cebae8f2ba99249df846cc0c9444fcbd4308/src/model/diffusion/qwen_image.hpp#L18)
保留旧版双流结构和固定的 24-head / 3072 hidden、3584 text 输入配置，
自动探测主要调整层数。

本次 2.1 配置是 32 层、32 heads、4096 hidden/context，且参考实现使用
single-stream、block-causal attention、可缓存前缀、SwiGLU，以及新的 64-channel
VAE。这些差异需要真正的模型适配，不能只修改层数、重命名权重或改格式。

后续若要取得公平的 GGML 对照，应先适配并验证这些模型结构，再明确 W8 的
量化语义、缓存、scheduler、初始 latent 和计时范围。当前任务没有改造该引擎。
