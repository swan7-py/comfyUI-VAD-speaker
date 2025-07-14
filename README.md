# ComfyUI-VAD-speaker

这是我第一次做comfy UI的插件，初衷是为了帮助数字人视频生成，利用语音活动检测（VAD）来完成切割，之后可以逐段进行处理，因为基于 [FunASR](https://github.com/alibaba-damo-academy/FunASR) 实现。因此也具备说话人分割的语音识别(ASR)能力。

## 安装

### 安装步骤
1. 将插件放置于 `ComfyUI/custom_nodes/` 目录下
2. 安装依赖：
```bash
   git clone https://github.com/swan7-py/comfyUI-VAD-speaker.git
   cd ComfyUI/custom_nodes/ComfyUI-VAD-speaker
   pip install -r requirements.txt
```

### 模型下载
1、联网情况下，自动从modelscope上下载
2、手动下载：请将它们下载到 `ComfyUI/models/FunASR/` 目录中：

| 模型名称 | 对应功能 | Modelscope 链接 |
|----------|----------|----------------|
| `speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn` | 语音识别(ASR) | [下载链接](https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn/summary) |
| `speech_fsmn_vad_zh-cn-16k-common-pytorch` | 语音活动检测(VAD) | [下载链接](https://www.modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common/summary) |
| `punc_ct-transformer_cn-en-common-vocab471067-large` | 标点恢复 | [下载链接](https://www.modelscope.cn/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large/summary) |
| `speech_campplus_sv_zh-cn_16k-common` | 说话人分离 | [下载链接](https://www.modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/summary) |

模型目录：
```bash
ComfyUI/
└── models/
    └── FunASR/
        ├── speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn/
        │   ├── ...
        ├── speech_fsmn_vad_zh-cn-16k-common-pytorch/
        │   ├── ...
        ├── punc_ct-transformer_cn-en-common-vocab471067-large/
        │   ├── ...
        └── speech_campplus_sv_zh-cn_16k-common/
            ├── ...
```

##节点功能
### 1. VAD 语音分割器 (VADSplitter)

基于 FunASR VAD 模型的智能语音分割节点，用于检测语音活动并将音频分割为有效的语音片段。

#### 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| `audio` | AUDIO | 输入音频 |
| `vad_threshold` | FLOAT | VAD 检测阈值 (0.01-0.99)，值越高检测越严格 |
| `max_silence` | FLOAT | 最大允许静音时长 (秒)，超过此时长会分割语音段 |
| `max_segment` | FLOAT | 单个语音段最大时长 (秒)，超过此时长会强制分割 |
| `unload_model` | BOOLEAN | 处理完成后是否卸载模型以释放内存 |
| `min_voice_duration` | FLOAT (可选) | 最小语音段时长 (秒)，低于此值的片段会被过滤 |

#### 输出结果

| 输出名 | 类型 | 描述 |
|--------|------|------|
| `time_segments` | STRING | 语音段时间戳字符串 (格式: "0.00-1.23,1.25-2.45,...") |
| `segments` | LIST | 语音段时间戳列表 (格式: `[(0.00, 1.23), (1.25, 2.45), ...]`) |

---

### 2. 分段选择器 (SegmentSelector)

从 VAD 分割结果中选择特定语音段的时间信息。

#### 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| `time_segments` | STRING | VADsplitter 输出的时间段字符串 |
| `index` | INT | 要选择的分段索引 (从0开始) |

#### 输出结果

| 输出名 | 类型 | 描述 |
|--------|------|------|
| `start_sec` | FLOAT | 选中分段的开始时间 (秒) |
| `end_sec` | FLOAT | 选中分段的结束时间 (秒) |
| `duration` | FLOAT | 选中分段的持续时间 (秒) |

---

### 3. 完整 ASR 处理器 (FullASRProcessor)

完整的语音识别处理器，集成 VAD、ASR、标点恢复和说话人分离功能。

#### 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| `audio` | AUDIO | 输入音频 |
| `vad_threshold` | FLOAT | VAD 检测阈值 (0.01-0.99) |
| `max_silence` | FLOAT | 最大允许静音时长 (秒) |
| `max_segment` | FLOAT | 单个语音段最大时长 (秒) |
| `enable_punctuation` | BOOLEAN | 是否启用标点恢复功能 |
| `enable_speaker` | BOOLEAN | 是否启用说话人分离功能 |
| `unload_model` | BOOLEAN | 处理完成后是否卸载模型以释放内存 |
| `preset_spk_num` | INT (可选) | 预设说话人数量 (0表示自动检测) |
| `min_voice_duration` | FLOAT (可选) | 最小语音段时长 (秒) |

#### 输出结果

| 输出名 | 类型 | 描述 |
|--------|------|------|
| `asr_result` | STRING | 语音识别结果文本 (格式: "说话人: 文本 [开始时间-结束时间]") |



