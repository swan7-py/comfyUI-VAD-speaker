import torchaudio
import subprocess
import tempfile
import os

class AudioProcessor:
    """音频处理工具类，提供全局可用的音频处理方法"""
    
    @staticmethod
    def apply_denoise(audio_tensor, sample_rate, noise_reduction,noise_floor):
        """使用FFmpeg应用去噪处理"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as input_file, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
            
            input_path = input_file.name
            output_path = output_file.name
            
            # 保存原始音频到临时文件
            torchaudio.save(input_path, audio_tensor.squeeze(0), sample_rate)
            
            # 构建FFmpeg命令
            # 使用afftdn滤波器进行去噪，强度范围0.01-1.0
            
            command = [
                'ffmpeg',
                '-y',  # 覆盖输出文件
                '-i', input_path,
                '-af', f'afftdn=nr={noise_reduction}:nf={noise_floor}',  # 动态调整噪声降低参数
                '-ar', str(sample_rate),  # 保持原始采样率
                output_path
            ]
            
            try:
                # 运行FFmpeg命令
                subprocess.run(
                    command, 
                    check=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                )
                
                # 加载处理后的音频
                denoised_waveform, sr = torchaudio.load(output_path)
                
                # 确保采样率一致
                if sr != sample_rate:
                    denoised_waveform = torchaudio.functional.resample(denoised_waveform, sr, sample_rate)
                 
                
                return denoised_waveform
                
            except subprocess.CalledProcessError as e:
                print(f"FFmpeg去噪失败: {e.stderr.decode()}")
                # 返回原始音频作为回退
                return audio_tensor
            finally:
                # 清理临时文件
                try:
                    os.unlink(input_path)
                    os.unlink(output_path)
                except:
                    pass
    