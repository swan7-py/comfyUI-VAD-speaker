import folder_paths
import os
import torchaudio
import torch
import uuid
import json
from funasr import AutoModel

# 模型名称映射表
name_maps_ms = {
    "paraformer": "speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn",
    "fsmn-vad": "speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "ct-punc": "punc_ct-transformer_cn-en-common-vocab471067-large",
    "cam++": "speech_campplus_sv_zh-cn_16k-common",
}

class VADsplitter:
    """基于FunASR VAD的智能语音分割节点"""
    infer_ins_cache = None
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "vad_threshold": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01}),
                "max_silence": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.1}),
                "max_segment": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "unload_model": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "min_voice_duration": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 1.0, "step": 0.05})
            }
        }

    RETURN_TYPES = ("STRING", "ASRRESULT","LIST", "AUDIO")
    RETURN_NAMES = ("time_segments", "vad_result", "segments", "vad_audio")
    FUNCTION = "vad_slice"
    CATEGORY = "Swan"

    def vad_slice(self, audio, vad_threshold, max_silence, max_segment, unload_model, min_voice_duration=0.3):
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        
        # 模型加载
        if VADsplitter.infer_ins_cache is None:
            model_root = os.path.join(folder_paths.models_dir, "FunASR")
            model_dir = os.path.join(model_root, name_maps_ms["fsmn-vad"])
            os.makedirs(model_dir, exist_ok=True)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            VADsplitter.infer_ins_cache = AutoModel(
                model=model_dir,
                vad_threshold=vad_threshold,
                vad_max_silence_duration=int(max_silence * 1000),
                device=device,
                disable_update=True
            )
        
        # 保存音频到临时文件
        uuidv4 = str(uuid.uuid4())
        audio_save_path = os.path.join(temp_dir, f"{uuidv4}.wav")
        
        # 重采样为16kHz
        waveform = audio['waveform']
        sr = audio["sample_rate"]
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        torchaudio.save(audio_save_path, waveform.squeeze(0), 16000)
        
        # 执行VAD检测
        vad_result = VADsplitter.infer_ins_cache.generate(
            input=audio_save_path,
            batch_size_s=300,
            param_dict={
                "vad_detect_mode": "normal",
                "max_single_segment_time": int(max_segment * 1000)
            }
        )
        
        # 清理临时文件
        if os.path.exists(audio_save_path):
            os.remove(audio_save_path)
        
        if not vad_result:
            return ("", {})
        
        main_result = vad_result[0]
        
        if "value" not in main_result or not main_result["value"]:
            return ("", main_result)
        
        segments = []
        for seg in main_result["value"]:
            if isinstance(seg, list) and len(seg) >= 2:
                start_sec = seg[0] / 1000.0
                end_sec = seg[1] / 1000.0
                duration = end_sec - start_sec
                
                # 过滤过短语音段
                if duration < min_voice_duration:
                    continue
                
                # 处理长段落：强制切割成不超过max_segment的小段
                if duration > max_segment:
                    num_segments = int(duration // max_segment) + 1
                    seg_duration = duration / num_segments
                    for i in range(num_segments):
                        seg_start = start_sec + i * seg_duration
                        seg_end = min(start_sec + (i + 1) * seg_duration, end_sec)
                        segments.append((round(seg_start, 2), round(seg_end, 2)))
                else:
                    segments.append((round(start_sec, 2), round(end_sec, 2)))
        
        # 格式化为字符串输出
        time_ranges = [f"{s[0]:.2f}-{s[1]:.2f}" for s in segments]
        time_segments_str = ",".join(time_ranges)

        vad_audio = {
            "waveform": waveform,
            "sample_rate": 16000
        }        
        if unload_model:
            self.unload_model()
            
        return (time_segments_str, main_result,segments,vad_audio)
    
    def unload_model(self):
        """卸载模型释放资源"""
        if VADsplitter.infer_ins_cache is not None:
            VADsplitter.infer_ins_cache = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

class FullASRProcessor:
    """完整的语音识别处理器 - 集成VAD, ASR, 标点, 说话人分离"""
    infer_ins_cache = None
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "vad_threshold": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01}),
                "max_silence": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.1}),
                "max_segment": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "enable_punctuation": ("BOOLEAN", {"default": True}),
                "enable_speaker": ("BOOLEAN", {"default": True}),
                "unload_model": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "preset_spk_num": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                "min_voice_duration": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 1.0, "step": 0.05})
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "ASRRESULT", "ASRRESULT")
    RETURN_NAMES = ("vad_segments", "asr_result", "vad_data", "asr_data")
    FUNCTION = "process_audio"
    CATEGORY = "Swan"

    def process_audio(self, audio, vad_threshold, max_silence, max_segment, 
                     enable_punctuation, enable_speaker, unload_model,
                     preset_spk_num=0, min_voice_duration=0.3):
        # 模型加载
        if FullASRProcessor.infer_ins_cache is None:
            model_root = os.path.join(folder_paths.models_dir, "FunASR")
            
            # 获取模型路径
            asr_model = os.path.join(model_root, name_maps_ms["paraformer"])
            vad_model = os.path.join(model_root, name_maps_ms["fsmn-vad"])
            punc_model = os.path.join(model_root, name_maps_ms["ct-punc"]) if enable_punctuation else None
            spk_model = os.path.join(model_root, name_maps_ms["cam++"]) if enable_speaker else None
            
            # 确保目录存在
            for path in [asr_model, vad_model, punc_model, spk_model]:
                if path: os.makedirs(path, exist_ok=True)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # 初始化模型
            FullASRProcessor.infer_ins_cache = AutoModel(
                model=asr_model,
                vad_model=vad_model,
                punc_model=punc_model,
                spk_model=spk_model,
                device=device,
                disable_update=True
            )
        
        # 保存音频到临时文件
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        uuidv4 = str(uuid.uuid4())
        audio_path = os.path.join(temp_dir, f"{uuidv4}.wav")
        
        # 处理音频
        waveform = audio['waveform']
        sr = audio["sample_rate"]
        
        # 重采样为16kHz
        waveform_16k = torchaudio.functional.resample(waveform, sr, 16000)
        torchaudio.save(audio_path, waveform_16k.squeeze(0), 16000)
        
        try:
            # 构建参数字典
            param_dict = {
                "vad_threshold": vad_threshold,
                "vad_max_silence_duration": int(max_silence * 1000),
                "max_single_segment_time": int(max_segment * 1000),
                "punc": "true" if enable_punctuation else "false"
            }
            
            # 添加预设说话人数量
            if preset_spk_num > 0:
                param_dict["preset_spk_num"] = preset_spk_num
            
            # 执行处理
            result = FullASRProcessor.infer_ins_cache.generate(
                input=audio_path,
                param_dict=param_dict
            )
            
            # 处理结果
            if not result:
                return ("", "", {}, {})
            
            main_result = result[0]
            
            # 提取VAD分割结果
            vad_segments_str = self.extract_vad_segments(main_result, min_voice_duration)
            
            # 提取ASR结果
            asr_result_str = self.extract_asr_result(main_result)
            
            return (vad_segments_str, asr_result_str, main_result, main_result)
            
        finally:
            # 清理资源
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if unload_model:
                self.unload_model()
    
    def extract_vad_segments(self, main_result, min_duration):
        """从结果中提取VAD分割时间戳"""
        if "vad_segments" not in main_result:
            return ""
        
        segments = []
        for seg in main_result["vad_segments"]:
            if isinstance(seg, list) and len(seg) >= 2:
                start = seg[0] / 1000.0
                end = seg[1] / 1000.0
                duration = end - start
                if duration >= min_duration:
                    segments.append(f"{start:.2f}-{end:.2f}")
        
        return ",".join(segments)
    
    def extract_asr_result(self, main_result):
        """从结果中提取ASR文本结果"""
        if "sentence_info" not in main_result:
            return ""
        
        asr_lines = []
        for sentence in main_result["sentence_info"]:
            speaker = sentence.get("spk", "spk0")
            text = sentence.get("text", "")
            start = sentence.get("start", 0) / 1000.0
            end = sentence.get("end", 0) / 1000.0
            
            # 格式化为 "说话人: 文本 [开始-结束]"
            asr_lines.append(f"{speaker}: {text} [{start:.2f}-{end:.2f}]")
        
        return "\n".join(asr_lines)
    
    def unload_model(self):
        """卸载模型释放资源"""
        if FullASRProcessor.infer_ins_cache is not None:
            FullASRProcessor.infer_ins_cache = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("FullASRProcessor: 模型已卸载")

NODE_CLASS_MAPPINGS = {
    "FullASRProcessor": FullASRProcessor,
    "VADSplitter": VADsplitter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FullASRProcessor": "完整ASR处理器",
    "VADSplitter": "VAD语音分割器"
}