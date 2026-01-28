"""
Generate Test Audio Samples for Aegis-AI Demo
==============================================
This script creates synthetic test samples to demonstrate the deepfake detection.
"""

import numpy as np
import soundfile as sf
import os

# Create output directory
os.makedirs("demo_audio_samples", exist_ok=True)

SAMPLE_RATE = 16000
DURATION = 5  # seconds

def generate_natural_voice_simulation(duration=5, sr=16000):
    """
    Generate audio that simulates natural human voice characteristics.
    Has natural variations in pitch, amplitude, and includes micro-pauses.
    """
    t = np.linspace(0, duration, int(sr * duration))
    
    # Base frequency with natural variations (human voice range)
    base_freq = 150 + 30 * np.sin(2 * np.pi * 0.5 * t)  # Slow pitch variation
    base_freq += np.random.randn(len(t)) * 10  # Random pitch jitter
    
    # Generate voice-like signal with harmonics
    signal = np.zeros_like(t)
    for harmonic in range(1, 6):
        amplitude = 1.0 / harmonic
        phase = np.cumsum(2 * np.pi * base_freq * harmonic / sr)
        signal += amplitude * np.sin(phase)
    
    # Add natural amplitude variations (breathing, emphasis)
    envelope = 0.5 + 0.3 * np.sin(2 * np.pi * 0.3 * t)  # Breathing rhythm
    envelope += 0.2 * np.sin(2 * np.pi * 1.5 * t)  # Speaking rhythm
    envelope = np.clip(envelope, 0.1, 1.0)
    
    # Add micro-pauses (natural speech gaps)
    for i in range(5):
        pause_start = int(np.random.uniform(0.5, duration - 0.3) * sr)
        pause_duration = int(np.random.uniform(0.05, 0.15) * sr)
        envelope[pause_start:pause_start + pause_duration] *= 0.1
    
    signal *= envelope
    
    # Add natural noise (breath, room tone)
    noise = np.random.randn(len(t)) * 0.02
    signal += noise
    
    # Add some formant-like filtering (voice resonances)
    from scipy import signal as sig
    b, a = sig.butter(4, [200, 3500], btype='band', fs=sr)
    signal = sig.filtfilt(b, a, signal)
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    return signal.astype(np.float32)


def generate_synthetic_voice_simulation(duration=5, sr=16000):
    """
    Generate audio that simulates AI-generated/synthetic voice characteristics.
    Has unnaturally consistent pitch, uniform amplitude, lacks micro-variations.
    """
    t = np.linspace(0, duration, int(sr * duration))
    
    # Very consistent base frequency (too perfect)
    base_freq = 160  # Fixed pitch - unnatural
    
    # Generate overly clean signal
    signal = np.zeros_like(t)
    for harmonic in range(1, 8):
        amplitude = 1.0 / (harmonic ** 1.5)  # Too clean harmonic structure
        signal += amplitude * np.sin(2 * np.pi * base_freq * harmonic * t)
    
    # Very uniform envelope (unnatural)
    envelope = 0.7 + 0.1 * np.sin(2 * np.pi * 2 * t)  # Too regular
    signal *= envelope
    
    # Minimal noise (too clean)
    noise = np.random.randn(len(t)) * 0.005
    signal += noise
    
    # Apply filtering
    from scipy import signal as sig
    b, a = sig.butter(4, [100, 4000], btype='band', fs=sr)
    signal = sig.filtfilt(b, a, signal)
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    return signal.astype(np.float32)


def generate_spoken_phrase_natural(text_hint, duration=5, sr=16000):
    """
    Generate more realistic speech-like audio with natural characteristics.
    """
    t = np.linspace(0, duration, int(sr * duration))
    
    # Variable pitch contour (like real speech)
    pitch_contour = 140 + 40 * np.sin(2 * np.pi * 0.4 * t)
    pitch_contour += 20 * np.sin(2 * np.pi * 1.2 * t + np.random.uniform(0, 2*np.pi))
    pitch_contour += np.random.randn(len(t)) * 15  # Jitter
    
    signal = np.zeros_like(t)
    phase = 0
    for i in range(len(t)):
        phase += 2 * np.pi * pitch_contour[i] / sr
        # Add harmonics with varying amplitudes
        signal[i] = (np.sin(phase) + 
                     0.5 * np.sin(2 * phase) + 
                     0.25 * np.sin(3 * phase) +
                     0.125 * np.sin(4 * phase))
    
    # Natural amplitude envelope with word-like patterns
    words = np.random.randint(4, 8)  # Number of "words"
    envelope = np.zeros_like(t)
    word_duration = duration / words
    
    for w in range(words):
        start = int(w * word_duration * sr)
        end = int((w + 0.8) * word_duration * sr)
        if end > len(envelope):
            end = len(envelope)
        word_env = np.hanning(end - start)
        envelope[start:end] = word_env
    
    # Add random variations
    envelope *= (0.7 + 0.3 * np.random.rand(len(envelope)))
    signal *= envelope
    
    # Add breath noise
    breath_noise = np.random.randn(len(t)) * 0.03
    signal += breath_noise * (1 - envelope + 0.1)
    
    # Filter
    from scipy import signal as sig
    b, a = sig.butter(3, [150, 3800], btype='band', fs=sr)
    signal = sig.filtfilt(b, a, signal)
    
    signal = signal / np.max(np.abs(signal)) * 0.75
    return signal.astype(np.float32)


def generate_spoken_phrase_synthetic(text_hint, duration=5, sr=16000):
    """
    Generate synthetic/AI-like speech audio.
    """
    t = np.linspace(0, duration, int(sr * duration))
    
    # Too consistent pitch (AI-like)
    pitch = 155 + 5 * np.sin(2 * np.pi * 0.8 * t)  # Very slight, regular variation
    
    signal = np.zeros_like(t)
    phase = 0
    for i in range(len(t)):
        phase += 2 * np.pi * pitch[i] / sr
        signal[i] = (np.sin(phase) + 
                     0.6 * np.sin(2 * phase) + 
                     0.35 * np.sin(3 * phase))
    
    # Too regular word patterns
    words = 5
    envelope = np.zeros_like(t)
    word_duration = duration / words
    
    for w in range(words):
        start = int(w * word_duration * sr)
        end = int((w + 0.75) * word_duration * sr)
        if end > len(envelope):
            end = len(envelope)
        envelope[start:end] = np.hanning(end - start)
    
    signal *= envelope
    
    # Very little noise (too clean)
    signal += np.random.randn(len(t)) * 0.008
    
    # Filter
    from scipy import signal as sig
    b, a = sig.butter(4, [120, 4200], btype='band', fs=sr)
    signal = sig.filtfilt(b, a, signal)
    
    signal = signal / np.max(np.abs(signal)) * 0.8
    return signal.astype(np.float32)


if __name__ == "__main__":
    print("🎤 Generating test audio samples for Aegis-AI Demo...")
    print("=" * 50)
    
    # Generate natural (real) voice samples
    print("\n✅ Generating REAL voice samples...")
    
    real_1 = generate_natural_voice_simulation(duration=5, sr=SAMPLE_RATE)
    sf.write("demo_audio_samples/real_voice_1.wav", real_1, SAMPLE_RATE)
    print("   → demo_audio_samples/real_voice_1.wav")
    
    real_2 = generate_spoken_phrase_natural("command verification", duration=5, sr=SAMPLE_RATE)
    sf.write("demo_audio_samples/real_voice_2.wav", real_2, SAMPLE_RATE)
    print("   → demo_audio_samples/real_voice_2.wav")
    
    # Generate synthetic (fake) voice samples
    print("\n⚠️ Generating FAKE (synthetic) voice samples...")
    
    fake_1 = generate_synthetic_voice_simulation(duration=5, sr=SAMPLE_RATE)
    sf.write("demo_audio_samples/fake_voice_1.wav", fake_1, SAMPLE_RATE)
    print("   → demo_audio_samples/fake_voice_1.wav")
    
    fake_2 = generate_spoken_phrase_synthetic("troop movement order", duration=5, sr=SAMPLE_RATE)
    sf.write("demo_audio_samples/fake_voice_2.wav", fake_2, SAMPLE_RATE)
    print("   → demo_audio_samples/fake_voice_2.wav")
    
    print("\n" + "=" * 50)
    print("✅ All test samples generated successfully!")
    print("\n📁 Files created in: demo_audio_samples/")
    print("   • real_voice_1.wav - Natural voice simulation")
    print("   • real_voice_2.wav - Natural speech pattern")
    print("   • fake_voice_1.wav - Synthetic voice (AI-like)")
    print("   • fake_voice_2.wav - Synthetic speech (AI-like)")
    print("\n💡 Use these samples to test the Aegis-AI demo!")
    print("   For better demo, also record your own voice and")
    print("   generate AI voices using ElevenLabs.io (free tier)")
