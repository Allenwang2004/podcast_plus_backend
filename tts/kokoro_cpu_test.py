import torch
import time
import os
from kokoro import KPipeline
import soundfile as sf

# 强制使用 CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 禁用所有 GPU 设备
torch.set_default_device('cpu')  # 设置默认设备为 CPU

print("=" * 60)
print("Kokoro CPU 测试程序")
print("=" * 60)

# 检查设备状态
print(f"\n设备信息:")
print(f"  CUDA 可用: {torch.cuda.is_available()}")
print(f"  MPS 可用: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
print(f"  默认设备: CPU (强制模式)")

# 测试文本
test_texts = [
    "Hello, this is a test of the Kokoro text-to-speech system running on CPU.",
    "Let's see how fast it can generate audio without GPU acceleration.",
    "This is the third test sentence."
]

# 测试语音
test_voices = ["af_heart", "am_adam"]

def test_kokoro_cpu():
    """测试 Kokoro 在 CPU 上的运行"""
    try:
        print("\n" + "=" * 60)
        print("初始化 Kokoro Pipeline (CPU 模式)...")
        print("=" * 60)
        
        start_time = time.time()
        # 使用 repo_id 参数消除警告
        pipeline = KPipeline(lang_code='a', device='cpu', repo_id='hexgrad/Kokoro-82M')
        init_time = time.time() - start_time
        
        print(f"✓ Pipeline 初始化完成 (耗时: {init_time:.2f}秒)")
        print(f"✓ 模型已加载到 CPU")
        
        # 测试每个声音和文本的组合
        for voice_idx, voice in enumerate(test_voices):
            print(f"\n{'=' * 60}")
            print(f"测试语音 {voice_idx + 1}/{len(test_voices)}: {voice}")
            print(f"{'=' * 60}")
            
            for text_idx, text in enumerate(test_texts):
                test_num = voice_idx * len(test_texts) + text_idx
                print(f"\n[测试 {test_num + 1}] 文本: {text[:50]}...")
                
                try:
                    # 生成音频
                    gen_start = time.time()
                    generator = pipeline(text, voice=voice)
                    
                    audio_generated = False
                    for _, _, audio in generator:
                        gen_time = time.time() - gen_start
                        
                        # 保存音频文件
                        output_file = f"test_{test_num:02d}_{voice}.wav"
                        sf.write(output_file, audio, 24000)
                        
                        # 统计信息
                        duration = len(audio) / 24000  # 音频时长（秒）
                        speed_ratio = duration / gen_time if gen_time > 0 else 0
                        
                        print(f"  ✓ 生成成功")
                        print(f"    - 生成时间: {gen_time:.2f}秒")
                        print(f"    - 音频时长: {duration:.2f}秒")
                        print(f"    - 速度比: {speed_ratio:.2f}x (实时倍速)")
                        print(f"    - 保存文件: {output_file}")
                        
                        audio_generated = True
                        break  # 只取第一个结果
                    
                    if not audio_generated:
                        print(f"  ✗ 未生成音频")
                        
                except Exception as e:
                    print(f"  ✗ 生成失败: {str(e)}")
                    continue
        
        print(f"\n{'=' * 60}")
        print("测试完成!")
        print(f"{'=' * 60}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_kokoro_cpu()
    exit(0 if success else 1)


