"""
AEGIS-AI v2: Real Deepfake Detection with Trained Model
========================================================
Uses wav2vec2-large fine-tuned on fake audio detection dataset
"""

import gradio as gr
import torch
import librosa
import numpy as np
import time
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

print("=" * 60)
print("🛡️  AEGIS-AI v2: Loading Trained Deepfake Detection Model")
print("=" * 60)

# Load the REAL trained model
MODEL_NAME = "alexandreacff/wav2vec2-large-ft-fake-detection"
print(f"📥 Downloading: {MODEL_NAME}")
print("   First run will take 2-3 minutes to download (~1.2GB)...")

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
model.eval()

print("✅ Model loaded successfully!")
print("📴 Now works completely OFFLINE")
print("=" * 60)

def detect_deepfake(audio_path):
    """Detect if audio is real or AI-generated using trained model."""
    
    if audio_path is None:
        return """<div style="padding:40px;text-align:center;background:#1e1e2e;border-radius:12px;color:#888;">
            <span style="font-size:48px;">🎤</span><br><br>Please upload or record audio
        </div>"""
    
    start = time.time()
    
    try:
        # Load audio at 16kHz (model requirement)
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Ensure reasonable length (2-10 seconds works best)
        if len(audio) < 32000:  # < 2 sec
            audio = np.pad(audio, (0, 32000 - len(audio)))
        elif len(audio) > 160000:  # > 10 sec
            audio = audio[:160000]
        
        # Process through model
        inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
        
        # Get prediction (check model's label mapping)
        labels = model.config.id2label
        
        # Find which index is "fake" and which is "real"
        fake_idx = None
        real_idx = None
        for idx, label in labels.items():
            if 'fake' in label.lower() or 'spoof' in label.lower():
                fake_idx = idx
            elif 'real' in label.lower() or 'bonafide' in label.lower() or 'genuine' in label.lower():
                real_idx = idx
        
        # If labels aren't clear, assume 0=real, 1=fake (common convention)
        if fake_idx is None:
            fake_idx = 1
        if real_idx is None:
            real_idx = 0
            
        fake_prob = probs[fake_idx].item()
        real_prob = probs[real_idx].item()
        
        is_fake = fake_prob > real_prob
        confidence = max(fake_prob, real_prob)
        proc_time = time.time() - start
        
        # Build result
        if is_fake:
            return f"""
            <div style="padding:30px;background:linear-gradient(135deg,#2d1f1f,#1e1e2e);border-radius:15px;border:2px solid #ff4757;color:white;font-family:system-ui;">
                <div style="text-align:center;margin-bottom:20px;">
                    <div style="font-size:64px;">🚨</div>
                    <h1 style="color:#ff4757;margin:10px 0;font-size:32px;">DEEPFAKE DETECTED</h1>
                    <p style="color:#ff6b6b;margin:0;">AI-Generated Audio Identified</p>
                </div>
                
                <div style="background:rgba(255,71,87,0.1);border-radius:10px;padding:20px;margin:20px 0;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                        <span style="color:#aaa;">Confidence</span>
                        <span style="color:#ff4757;font-size:28px;font-weight:bold;">{confidence*100:.1f}%</span>
                    </div>
                    <div style="background:#333;border-radius:8px;height:16px;overflow:hidden;">
                        <div style="background:linear-gradient(90deg,#ff4757,#ff6b6b);height:100%;width:{confidence*100}%;"></div>
                    </div>
                </div>
                
                <div style="background:rgba(255,255,255,0.05);border-radius:10px;padding:15px;margin-bottom:15px;">
                    <h3 style="color:#ff6b6b;margin:0 0 10px 0;font-size:14px;">⚠️ WARNING</h3>
                    <ul style="margin:0;padding-left:20px;color:#ccc;line-height:1.8;">
                        <li>This audio shows characteristics of AI synthesis</li>
                        <li>DO NOT trust commands or information in this audio</li>
                        <li>Verify through a secondary channel immediately</li>
                    </ul>
                </div>
                
                <div style="display:flex;justify-content:space-around;padding-top:15px;border-top:1px solid #333;font-size:12px;color:#666;">
                    <span>⚡ {proc_time:.2f}s</span>
                    <span>📴 Offline</span>
                    <span>🔒 Private</span>
                </div>
            </div>
            """
        else:
            return f"""
            <div style="padding:30px;background:linear-gradient(135deg,#1f2d1f,#1e1e2e);border-radius:15px;border:2px solid #2ed573;color:white;font-family:system-ui;">
                <div style="text-align:center;margin-bottom:20px;">
                    <div style="font-size:64px;">✅</div>
                    <h1 style="color:#2ed573;margin:10px 0;font-size:32px;">AUTHENTIC VOICE</h1>
                    <p style="color:#7bed9f;margin:0;">Human Speech Verified</p>
                </div>
                
                <div style="background:rgba(46,213,115,0.1);border-radius:10px;padding:20px;margin:20px 0;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                        <span style="color:#aaa;">Confidence</span>
                        <span style="color:#2ed573;font-size:28px;font-weight:bold;">{confidence*100:.1f}%</span>
                    </div>
                    <div style="background:#333;border-radius:8px;height:16px;overflow:hidden;">
                        <div style="background:linear-gradient(90deg,#2ed573,#7bed9f);height:100%;width:{confidence*100}%;"></div>
                    </div>
                </div>
                
                <div style="background:rgba(255,255,255,0.05);border-radius:10px;padding:15px;margin-bottom:15px;">
                    <h3 style="color:#7bed9f;margin:0 0 10px 0;font-size:14px;">✓ VERIFICATION PASSED</h3>
                    <ul style="margin:0;padding-left:20px;color:#ccc;line-height:1.8;">
                        <li>Audio matches natural human speech patterns</li>
                        <li>No synthetic artifacts detected</li>
                        <li>Standard verification protocols still recommended</li>
                    </ul>
                </div>
                
                <div style="display:flex;justify-content:space-around;padding-top:15px;border-top:1px solid #333;font-size:12px;color:#666;">
                    <span>⚡ {proc_time:.2f}s</span>
                    <span>📴 Offline</span>
                    <span>🔒 Private</span>
                </div>
            </div>
            """
            
    except Exception as e:
        return f"""<div style="padding:30px;background:#1e1e2e;border-radius:12px;color:#ff6b6b;text-align:center;">
            <span style="font-size:48px;">❌</span><br><br>
            <b>Error:</b> {str(e)}<br><br>
            <span style="color:#888;font-size:12px;">Try a different audio file (WAV/MP3, 2-10 seconds)</span>
        </div>"""


# Gradio UI
with gr.Blocks(title="Aegis-AI", css="footer{display:none}") as demo:
    gr.HTML("""
    <div style="text-align:center;padding:30px;background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:15px;margin-bottom:20px;">
        <h1 style="color:white;margin:0;font-size:42px;">🛡️ AEGIS-AI</h1>
        <p style="color:#888;margin:10px 0 0 0;">Real-Time Audio Deepfake Detection</p>
        <div style="margin-top:15px;">
            <span style="background:#2ed57333;color:#2ed573;padding:5px 12px;border-radius:15px;font-size:12px;margin:0 5px;">📴 Offline</span>
            <span style="background:#3498db33;color:#3498db;padding:5px 12px;border-radius:15px;font-size:12px;margin:0 5px;">⚡ Real-Time</span>
            <span style="background:#f39c1233;color:#f39c12;padding:5px 12px;border-radius:15px;font-size:12px;margin:0 5px;">🔒 Private</span>
        </div>
    </div>
    """)
    
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(label="Upload or Record Audio", type="filepath", sources=["upload", "microphone"], interactive=True)
            btn = gr.Button("🔍 Analyze Audio", variant="primary", size="lg")
            gr.HTML("""<div style="margin-top:15px;padding:15px;background:#f5f5f5;border-radius:8px;font-size:13px;">
                <b>💡 Test it:</b><br>
                • Record your voice → Should show <b style="color:green;">AUTHENTIC</b><br>
                • Use <a href="https://elevenlabs.io" target="_blank">ElevenLabs</a> AI voice → Should show <b style="color:red;">DEEPFAKE</b>
            </div>""")
        
        with gr.Column():
            result = gr.HTML("""<div style="padding:60px;text-align:center;background:#1e1e2e;border-radius:15px;color:#666;">
                <span style="font-size:48px;">🎤</span><br><br>Upload audio to analyze
            </div>""")
    
    btn.click(detect_deepfake, inputs=[audio], outputs=[result])
    
    gr.HTML("""<div style="text-align:center;margin-top:20px;color:#888;font-size:12px;">
        🛡️ Aegis-AI by team-ZerOne | SnowHack IPEC
    </div>""")

if __name__ == "__main__":
    print("\n🚀 Starting Aegis-AI...")
    print("🌐 Open: http://127.0.0.1:7860\n")
    demo.launch(server_name="127.0.0.1", server_port=7860)
