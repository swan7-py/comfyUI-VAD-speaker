import folder_paths
import os
import torchaudio
import torch
import uuid
from funasr import AutoModel
from modelscope import snapshot_download
from .audioutils import AudioProcessor

# 模型名称
name_maps_ms = {
    "paraformer": "speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn",
    "fsmn-vad": "speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "ct-punc": "punc_ct-transformer_cn-en-common-vocab471067-large",
    "cam++": "speech_campplus_sv_zh-cn_16k-common",
}

class SegmentSelector:
    """从VAD分割结果中选择指定段的时间信息"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "time_segments": ("STRING",),
                "index": ("INT", {"default": 0, "min": 0, "step": 1}),
                "fps": ("FLOAT", {"default": 25, "min": 8, "step": 1}),
                "denoise_enable": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "noise_reduction": ("INT", {"default": 12, "min":  0.01, "max": 97, "step":  0.01}),
                "noise_floor": ("INT", {"default": -50, "min": -80, "max": -20, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("AUDIO","FLOAT", "FLOAT", "FLOAT","INT","FLOAT")
    RETURN_NAMES = ("audio_segment","start_sec", "end_sec", "duration","frame_num","FPS_out")
    FUNCTION = "select_segment"
    CATEGORY = "Swan"

    def select_segment(self,audio, time_segments, index, fps,denoise_enable,noise_reduction,noise_floor):

        segments = []
        for segment_str in time_segments.split(','):
            if '-' in segment_str:
                try:
                    start_str, end_str = segment_str.split('-')
                    start = float(start_str)
                    end = float(end_str)
                    segments.append((start, end))
                except ValueError:                    
                    continue

        if index < 0 or index >= len(segments):
            raise ValueError(f"索引 {index} 超出范围 (0-{len(segments)-1})")
        
        segment = segments[index]
        start = segment[0]
        end = segment[1]
        duration = end - start
        FPS_out = fps
        frame_num = int(duration * fps)+1

         # 提取音频片段 - 使用原始浮点时间值
        waveform = audio['waveform']
        sample_rate = audio["sample_rate"]
        
        # 计算开始和结束的样本位置
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        
        
        # 截取音频片段
        audio_segment = waveform[..., start_sample:end_sample]
        # 去噪
        if denoise_enable:
            audio_segment = AudioProcessor.apply_denoise(audio_segment, sample_rate, noise_reduction,noise_floor)
            audio_segment = audio_segment.unsqueeze(0)
        clipped_audio = {
            "waveform": audio_segment,
            "sample_rate": sample_rate,
        }
   
        return (clipped_audio,start, end, duration, frame_num,FPS_out)


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
                "denoise_enable": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "min_voice_duration": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 1.0, "step": 0.05}),
                "noise_reduction": ("INT", {"default": 12, "min":  0.01, "max": 97, "step":  0.01}),
                "noise_floor": ("INT", {"default": -50, "min": -80, "max": -20, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("STRING", "LIST")
    RETURN_NAMES = ("time_segments", "segments")
    FUNCTION = "vad_slice"
    CATEGORY = "Swan"

    def vad_slice(self, audio, vad_threshold, max_silence, max_segment, unload_model,denoise_enable, min_voice_duration, noise_reduction,noise_floor):
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        
        if VADsplitter.infer_ins_cache is None:
            model_root = os.path.join(folder_paths.models_dir, "FunASR")
            model_dir = snapshot_download(f'iic/{name_maps_ms["fsmn-vad"]}',local_dir=f'{model_root}/{name_maps_ms["fsmn-vad"]}')
            os.makedirs(model_dir, exist_ok=True)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            VADsplitter.infer_ins_cache = AutoModel(
                model=model_dir,
                vad_threshold=vad_threshold,
                vad_max_silence_duration=int(max_silence * 1000),
                device=device,
                disable_update=True
            )
        
        uuidv4 = str(uuid.uuid4())
        audio_save_path = os.path.join(temp_dir, f"{uuidv4}.wav")
        
        waveform = audio['waveform']
        sr = audio["sample_rate"]
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        torchaudio.save(audio_save_path, waveform.squeeze(0), 16000)
        if denoise_enable:
            waveform = AudioProcessor.apply_denoise(waveform, 16000, noise_reduction,noise_floor)

        vad_result = VADsplitter.infer_ins_cache.generate(
            input=audio_save_path,
            batch_size_s=300,
            param_dict={
                "vad_detect_mode": "normal",
                "max_single_segment_time": int(max_segment * 1000)
            }
        )
        
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
                
                if duration < min_voice_duration:
                    continue                

                if duration > max_segment:
                    num_segments = int(duration // max_segment) + 1
                    seg_duration = duration / num_segments
                    for i in range(num_segments):
                        seg_start = start_sec + i * seg_duration
                        seg_end = min(start_sec + (i + 1) * seg_duration, end_sec)
                        segments.append((round(seg_start, 2), round(seg_end, 2)))
                else:
                    segments.append((round(start_sec, 2), round(end_sec, 2)))
        
        
        time_ranges = [f"{s[0]:.2f}-{s[1]:.2f}" for s in segments]
        time_segments_str = ",".join(time_ranges)
     
        if unload_model:
            self.unload_model()
            
        return (time_segments_str, segments)
    
    def unload_model(self):
        if VADsplitter.infer_ins_cache is not None:
            VADsplitter.infer_ins_cache = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

class FullASRProcessor:
    
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
                "denoise_enable": ("BOOLEAN", {"default": False}),
                "unload_model": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "preset_spk_num": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                "min_voice_duration": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 1.0, "step": 0.05}),
                "noise_reduction": ("INT", {"default": 12, "min":  0.01, "max": 97, "step":  0.01}),
                "noise_floor": ("INT", {"default": -50, "min": -80, "max": -20, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("asr_result",)
    FUNCTION = "process_audio"
    CATEGORY = "Swan"

    def process_audio(self, audio, vad_threshold, max_silence, max_segment, 
                     enable_punctuation, enable_speaker, unload_model,denoise_enable,
                     preset_spk_num, min_voice_duration,noise_reduction,noise_floor):
        if FullASRProcessor.infer_ins_cache is None:
            model_root = os.path.join(folder_paths.models_dir, "FunASR")
            asr_model = snapshot_download(f'iic/{name_maps_ms["paraformer"]}',local_dir=f'{model_root}/{name_maps_ms["paraformer"]}')
            vad_model = snapshot_download(f'iic/{name_maps_ms["fsmn-vad"]}',local_dir=f'{model_root}/{name_maps_ms["fsmn-vad"]}')           
            punc_model = snapshot_download(f'iic/{name_maps_ms["ct-punc"]}',local_dir=f'{model_root}/{name_maps_ms["ct-punc"]}') if enable_punctuation else None        
            spk_model = snapshot_download(f'iic/{name_maps_ms["cam++"]}',local_dir=f'{model_root}/{name_maps_ms["cam++"]}') if enable_speaker else None


            
            for path in [asr_model, vad_model, punc_model, spk_model]:
                if path: os.makedirs(path, exist_ok=True)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            FullASRProcessor.infer_ins_cache = AutoModel(
                model=asr_model,
                vad_model=vad_model,
                punc_model=punc_model,
                spk_model=spk_model,
                device=device,
                disable_update=True
            )
        
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        uuidv4 = str(uuid.uuid4())
        audio_path = os.path.join(temp_dir, f"{uuidv4}.wav")
        
        waveform = audio['waveform']
        sr = audio["sample_rate"]
        if denoise_enable:
            waveform = AudioProcessor.apply_denoise(waveform, sr, noise_reduction,noise_floor)        
        waveform_16k = torchaudio.functional.resample(waveform, sr, 16000)
        torchaudio.save(audio_path, waveform_16k.squeeze(0), 16000)
        
        try:
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
                return ("",)
            
            main_result = result[0]
            
            # 提取ASR结果
            asr_result_str = self.extract_asr_result(main_result)
            
            return (asr_result_str,)
            
        finally:
            # 清理资源
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if unload_model:
                self.unload_model()
    
    
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
    "SegmentSelector": SegmentSelector,
    "FullASRProcessor": FullASRProcessor,
    "VADSplitter": VADsplitter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SegmentSelector":"音频分段选择器",
    "FullASRProcessor": "ASR语音识别",
    "VADSplitter": "VAD语音分割器"
}
