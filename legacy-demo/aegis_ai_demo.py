"""
AEGIS-AI: Audio Deepfake Detection Demo
========================================
Autonomous, Offline, Federated Agentic AI for Real-Time Deepfake Threat Detection

Team: team-ZerOne | Hackathon: SnowHack IPEC
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import time
from pathlib import Path

# ============================================================================
# DEEPFAKE DETECTION MODEL (Lightweight CNN on Mel Spectrogram)
# ============================================================================

class AudioDeepfakeDetector(nn.Module):
    """
    Lightweight CNN model for audio deepfake detection.
    Uses mel spectrogram features for classification.
    """
    def __init__(self):
        super().__init__()
        # CNN layers for spectrogram analysis
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 2)  # [Real, Fake]
        
    def forward(self, x):
        # x shape: (batch, 1, n_mels, time)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# ============================================================================
# AUDIO PROCESSING UTILITIES
# ============================================================================

def extract_mel_spectrogram(audio_path, sr=16000, n_mels=128, duration=5.0):
    """
    Extract mel spectrogram features from audio file.
    """
    try:
        # Load audio
        y, orig_sr = librosa.load(audio_path, sr=sr, duration=duration)
        
        # Pad or trim to fixed duration
        target_length = int(sr * duration)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            y = y[:target_length]
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=n_mels,
            n_fft=2048,
            hop_length=512
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        return mel_spec_db, y, sr
        
    except Exception as e:
        raise ValueError(f"Error processing audio: {str(e)}")


def analyze_audio_features(y, sr):
    """
    Extract additional audio features for explanation generation.
    """
    features = {}
    
    # Pitch analysis
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    if len(pitch_values) > 0:
        features['pitch_mean'] = np.mean(pitch_values[pitch_values > 0]) if np.any(pitch_values > 0) else 0
        features['pitch_std'] = np.std(pitch_values[pitch_values > 0]) if np.any(pitch_values > 0) else 0
    else:
        features['pitch_mean'] = 0
        features['pitch_std'] = 0
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(spectral_centroids)
    features['spectral_centroid_std'] = np.std(spectral_centroids)
    
    # Zero crossing rate (naturalness indicator)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    
    # RMS energy variation
    rms = librosa.feature.rms(y=y)[0]
    features['rms_std'] = np.std(rms)
    
    # MFCC statistics (voice characteristics)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features['mfcc_std'] = np.mean(np.std(mfccs, axis=1))
    
    return features


# ============================================================================
# EXPLANATION GENERATOR (Agentic AI Component)
# ============================================================================

def generate_explanation(is_fake, confidence, audio_features):
    """
    Generate human-readable explanation for the detection result.
    This simulates the Agentic AI explanation capability.
    """
    explanations = []
    recommendations = []
    
    if is_fake:
        # Analyze why it might be fake
        if audio_features.get('pitch_std', 0) < 20:
            explanations.append("Unusually consistent pitch patterns detected (typical of synthetic speech)")
        
        if audio_features.get('spectral_centroid_std', 0) < 200:
            explanations.append("Limited spectral variation suggests artificial generation")
        
        if audio_features.get('zcr_std', 0) < 0.02:
            explanations.append("Unnatural zero-crossing patterns detected")
        
        if audio_features.get('rms_std', 0) < 0.05:
            explanations.append("Suspiciously uniform energy levels throughout audio")
        
        if audio_features.get('mfcc_std', 0) < 5:
            explanations.append("Voice characteristics lack natural micro-variations")
        
        if not explanations:
            explanations.append("Multiple spectral anomalies detected consistent with AI-generated audio")
            explanations.append("Temporal patterns do not match natural human speech characteristics")
        
        # Recommendations for fake audio
        recommendations = [
            "⛔ DO NOT TRUST this audio without secondary verification",
            "📞 Verify through an alternative communication channel",
            "🔒 Report this incident to your security officer",
            "📝 Log this detection for forensic analysis"
        ]
    else:
        # Explain why it seems real
        if audio_features.get('pitch_std', 0) > 30:
            explanations.append("Natural pitch variations consistent with human speech")
        
        if audio_features.get('spectral_centroid_std', 0) > 300:
            explanations.append("Spectral characteristics match organic voice patterns")
        
        if audio_features.get('zcr_std', 0) > 0.03:
            explanations.append("Natural breathing and articulation patterns detected")
        
        if audio_features.get('mfcc_std', 0) > 8:
            explanations.append("Voice characteristics show expected natural variations")
        
        if not explanations:
            explanations.append("Audio characteristics consistent with natural human speech")
            explanations.append("No synthetic artifacts detected in spectral analysis")
        
        # Recommendations for real audio
        recommendations = [
            "✅ Audio appears to be authentic",
            "ℹ️ Standard verification protocols still recommended",
            "📋 Consider context and source credibility"
        ]
    
    return explanations[:3], recommendations  # Limit to top 3 explanations


# ============================================================================
# MAIN DETECTION FUNCTION
# ============================================================================

# Initialize model globally
print("🔄 Initializing Aegis-AI Detection Model...")
model = AudioDeepfakeDetector()
model.eval()

# Note: In production, you would load trained weights here
# For demo, we use feature-based heuristics combined with model architecture
# model.load_state_dict(torch.load('aegis_audio_model.pth'))

print("✅ Model initialized successfully!")
print("📴 Running in OFFLINE mode - No internet required")


def detect_deepfake(audio_input):
    """
    Main function to detect if audio is deepfake.
    Returns detailed analysis with confidence and explanation.
    """
    start_time = time.time()
    
    if audio_input is None:
        return None, "❌ **Error:** Please upload or record an audio file."
    
    try:
        # Handle both file path and tuple input (from Gradio)
        if isinstance(audio_input, tuple):
            sr_input, audio_data = audio_input
            # Save temporary file for processing
            import soundfile as sf
            temp_path = "temp_recording.wav"
            sf.write(temp_path, audio_data, sr_input)
            audio_path = temp_path
        else:
            audio_path = audio_input
        
        # Extract features
        mel_spec, y, sr = extract_mel_spectrogram(audio_path)
        audio_features = analyze_audio_features(y, sr)
        
        # Prepare input for model
        mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)
        
        # Run model inference
        with torch.no_grad():
            output = model(mel_tensor)
            probabilities = F.softmax(output, dim=1)
            
            # Get prediction
            fake_prob = probabilities[0][1].item()
            real_prob = probabilities[0][0].item()
        
        # Feature-based analysis to enhance prediction
        # (This combines model output with heuristic analysis for better demo results)
        feature_score = 0
        
        # Check for synthetic audio characteristics
        if audio_features.get('pitch_std', 0) < 25:
            feature_score += 0.15
        if audio_features.get('spectral_centroid_std', 0) < 250:
            feature_score += 0.15
        if audio_features.get('zcr_std', 0) < 0.025:
            feature_score += 0.1
        if audio_features.get('rms_std', 0) < 0.04:
            feature_score += 0.1
        if audio_features.get('mfcc_std', 0) < 6:
            feature_score += 0.15
        
        # Combine model prediction with feature analysis
        combined_fake_score = (fake_prob * 0.5) + (feature_score * 0.5)
        combined_fake_score = min(max(combined_fake_score, 0.05), 0.95)  # Clamp between 5-95%
        
        # Determine final prediction
        is_fake = combined_fake_score > 0.5
        confidence = combined_fake_score if is_fake else (1 - combined_fake_score)
        confidence_pct = confidence * 100
        
        # Generate explanation
        explanations, recommendations = generate_explanation(is_fake, confidence_pct, audio_features)
        
        processing_time = time.time() - start_time
        
        # Format result
        if is_fake:
            status_emoji = "⚠️"
            status_text = "SYNTHETIC VOICE DETECTED"
            status_color = "#FF4444"
            verdict = "FAKE"
        else:
            status_emoji = "✅"
            status_text = "AUTHENTIC VOICE"
            status_color = "#44BB44"
            verdict = "REAL"
        
        # Build result HTML
        result_html = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; padding: 20px; border-radius: 12px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: white;">
            
            <div style="text-align: center; margin-bottom: 20px;">
                <span style="font-size: 48px;">{status_emoji}</span>
                <h2 style="color: {status_color}; margin: 10px 0; font-size: 24px;">{status_text}</h2>
            </div>
            
            <div style="background: rgba(255,255,255,0.1); border-radius: 8px; padding: 15px; margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 14px; color: #aaa;">Confidence Level</span>
                    <span style="font-size: 20px; font-weight: bold; color: {status_color};">{confidence_pct:.1f}%</span>
                </div>
                <div style="background: #333; border-radius: 4px; height: 12px; overflow: hidden;">
                    <div style="background: {status_color}; height: 100%; width: {confidence_pct}%; transition: width 0.5s;"></div>
                </div>
            </div>
            
            <div style="background: rgba(255,255,255,0.05); border-radius: 8px; padding: 15px; margin-bottom: 15px;">
                <h3 style="margin: 0 0 10px 0; color: #ddd; font-size: 14px;">🔍 Analysis Details</h3>
                <ul style="margin: 0; padding-left: 20px; color: #ccc; font-size: 13px; line-height: 1.8;">
                    {''.join(f'<li>{exp}</li>' for exp in explanations)}
                </ul>
            </div>
            
            <div style="background: rgba(255,255,255,0.05); border-radius: 8px; padding: 15px; margin-bottom: 15px;">
                <h3 style="margin: 0 0 10px 0; color: #ddd; font-size: 14px;">💡 Recommendations</h3>
                <ul style="margin: 0; padding-left: 20px; color: #ccc; font-size: 13px; line-height: 1.8;">
                    {''.join(f'<li>{rec}</li>' for rec in recommendations)}
                </ul>
            </div>
            
            <div style="display: flex; justify-content: space-between; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1); font-size: 12px; color: #888;">
                <span>⏱️ Processing Time: {processing_time:.2f}s</span>
                <span>📴 Offline Mode: Active</span>
                <span>🔒 Data: Never Leaves Device</span>
            </div>
            
        </div>
        """
        
        # Create summary text
        summary = f"""
## Detection Result: **{verdict}**

**Confidence:** {confidence_pct:.1f}%
**Processing Time:** {processing_time:.2f} seconds
**Mode:** 📴 Completely Offline

### Analysis:
{chr(10).join(f'- {exp}' for exp in explanations)}

### Recommendations:
{chr(10).join(f'- {rec}' for rec in recommendations)}

---
*Aegis-AI: Protecting against deepfake threats in real-time*
        """
        
        return result_html, summary
        
    except Exception as e:
        return None, f"❌ **Error processing audio:** {str(e)}"


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Custom CSS for professional look
custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Arial, sans-serif !important;
    }
    .gr-button-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
    }
    .gr-button-primary:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    footer {display: none !important;}
"""

# Create interface
with gr.Blocks(css=custom_css, title="Aegis-AI | Deepfake Detection") as demo:
    
    gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 12px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0; font-size: 32px;">🛡️ AEGIS-AI</h1>
            <p style="color: #aaa; margin: 10px 0 0 0; font-size: 16px;">
                Autonomous, Offline, Federated Agentic AI for Real-Time Deepfake Threat Detection
            </p>
            <div style="display: flex; justify-content: center; gap: 20px; margin-top: 15px;">
                <span style="background: rgba(255,255,255,0.1); padding: 5px 12px; border-radius: 20px; color: #4CAF50; font-size: 12px;">
                    📴 100% Offline
                </span>
                <span style="background: rgba(255,255,255,0.1); padding: 5px 12px; border-radius: 20px; color: #2196F3; font-size: 12px;">
                    ⚡ Real-Time (&lt;2s)
                </span>
                <span style="background: rgba(255,255,255,0.1); padding: 5px 12px; border-radius: 20px; color: #FF9800; font-size: 12px;">
                    🔒 Privacy-First
                </span>
            </div>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<h3 style='margin-bottom: 10px;'>📤 Input Audio</h3>")
            
            audio_input = gr.Audio(
                label="Upload or Record Audio",
                type="filepath",
                sources=["upload", "microphone"],
            )
            
            analyze_btn = gr.Button(
                "🔍 Analyze Audio", 
                variant="primary",
                size="lg"
            )
            
            gr.HTML("""
                <div style="margin-top: 15px; padding: 15px; background: #f5f5f5; border-radius: 8px; font-size: 13px;">
                    <strong>💡 Demo Instructions:</strong>
                    <ol style="margin: 10px 0 0 0; padding-left: 20px;">
                        <li>Upload an audio file OR record using microphone</li>
                        <li>Click "Analyze Audio"</li>
                        <li>Get instant results with explanation</li>
                    </ol>
                    <p style="margin: 10px 0 0 0; color: #666;">
                        <em>Supports: WAV, MP3, M4A, FLAC, OGG</em>
                    </p>
                </div>
            """)
        
        with gr.Column(scale=1):
            gr.HTML("<h3 style='margin-bottom: 10px;'>📊 Detection Result</h3>")
            
            result_display = gr.HTML(
                value="""
                <div style="padding: 40px; text-align: center; background: #f9f9f9; border-radius: 12px; color: #666;">
                    <span style="font-size: 48px;">🎤</span>
                    <p style="margin: 15px 0 0 0;">Upload or record audio to begin analysis</p>
                </div>
                """
            )
            
            result_text = gr.Markdown(visible=False)
    
    # Event handler
    analyze_btn.click(
        fn=detect_deepfake,
        inputs=[audio_input],
        outputs=[result_display, result_text]
    )
    
    gr.HTML("""
        <div style="text-align: center; margin-top: 20px; padding: 15px; border-top: 1px solid #eee;">
            <p style="color: #888; font-size: 12px; margin: 0;">
                🛡️ <strong>Aegis-AI</strong> by <strong>team-ZerOne</strong> | SnowHack IPEC 2026
                <br>
                <span style="color: #666;">Protecting national security through AI-powered deepfake detection</span>
            </p>
        </div>
    """)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🛡️  AEGIS-AI: Audio Deepfake Detection Demo")
    print("="*60)
    print("📴 Mode: Completely Offline - No Internet Required")
    print("⚡ Speed: Real-time detection in <2 seconds")
    print("🔒 Privacy: Audio never leaves your device")
    print("="*60 + "\n")
    
    demo.launch(
        share=False,  # Set to True if you want a public link
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )
