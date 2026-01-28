# 🚀 FINAL INSTRUCTIONS - Train Your Model Now!

## ⚠️ Important Discovery

Your system has **Python 3.14** (released Oct 2025), which is too new for ONNX Runtime. 

**Solution:** Use Google Colab (Python 3.10, free GPU, all packages work).

---

## 🎯 Two Options - Choose One

### **Option 1: Super Simple (Recommended)** ⏱️ Fastest

1. **Open in VS Code:**
   ```
   File → Open File → ml/training/SIMPLE_COLAB.ipynb
   ```

2. **Connect to Colab:**
   - Click kernel selector (top right)
   - Choose "Google Colab" or "Connect to Colab"
   - Sign in with Google account
   - Wait for connection

3. **Run 4 Cells:**
   - **Cell 1:** Upload `train_audio_standalone.py` (VS Code will prompt)
   - **Cell 2:** Quick test (5-10 min) ← Run this first!
   - **Cell 3:** Full training (1-3 hours) ← Only if Cell 2 works
   - **Cell 4:** Download model

4. **Done!** You'll have `V1.0.0.onnx` in Downloads folder

---

### **Option 2: Direct Colab Website** ⏱️ Alternative

1. **Go to:** https://colab.research.google.com

2. **Upload notebook:**
   - File → Upload notebook
   - Choose `ml/training/SIMPLE_COLAB.ipynb`

3. **Enable GPU:**
   - Runtime → Change runtime type → GPU (T4)

4. **Run cells 1-4** as described above

---

## 📋 What You Need to Upload

When Cell 1 runs, it will ask you to upload **one file**:

```
📁 File to upload: ml/training/train_audio_standalone.py
📍 Location: C:\Users\kumar\Desktop\Projects\SnowHackIPEC\ml\training\train_audio_standalone.py
```

This file contains everything needed (500 lines, fully self-contained).

---

## ⏱️ Timeline

| Step | Time | What Happens |
|------|------|--------------|
| Cell 1: Upload | 1 min | Upload script + verify GPU |
| Cell 2: Quick test | 5-10 min | Downloads dataset, trains on 500 samples |
| Cell 3: Full training | 1-3 hours | Trains on full dataset (20 epochs) |
| Cell 4: Download | 1 min | Downloads V1.0.0.onnx to your computer |

**Total: ~1.5-3.5 hours** (mostly unattended)

---

## 🎉 After Training

Once you have `V1.0.0.onnx` downloaded:

```bash
# 1. Integrate the model
python ml/scripts/integrate_trained_model.py --model ~/Downloads/V1.0.0.onnx

# 2. The script will:
#    - Copy model to models/audio/
#    - Create latest.onnx symlink
#    - Verify it loads correctly
#    - Show you next steps

# 3. Test it
export ONNX_MODEL_PATH=models/audio/latest.onnx  # Linux/Mac
# Or on Windows:
set ONNX_MODEL_PATH=models/audio/latest.onnx

pytest services/api/tests/test_inference.py -v
# Should show: 6 passed, 0 skipped (was 4 passed, 2 skipped)

# 4. Start the API
cd services/api
uvicorn app.main:app --reload

# 5. Test real inference
curl http://localhost:8000/v1/models
```

---

## 🆘 Troubleshooting

### "Cannot connect to Colab"
- Make sure you're signed into a Google account
- Try using the Colab website directly (Option 2)
- Check VS Code has internet access

### "Upload file not found"
- Full path: `C:\Users\kumar\Desktop\Projects\SnowHackIPEC\ml\training\train_audio_standalone.py`
- If using VS Code extension, use the file picker dialog

### "Out of Memory"
- In Cell 2, change `--batch-size 16` to `--batch-size 8`
- In Cell 3, change `--batch-size 32` to `--batch-size 16`

### "Dataset download slow"
- Colab has fast internet (~100 Mbps)
- 7.6 GB should take 10-30 minutes
- If stuck, restart runtime and try again

---

## 📊 Expected Results

After training completes:

```
✓ TRAINING COMPLETE!
  ONNX: models/V1.0.0.onnx
  Accuracy: 85-95%
  Parameters: ~2.5M
```

**Model Performance:**
- Validation accuracy: **85-95%**
- Inference latency: **<100ms** per 10s audio
- Model size: **~10-50 MB**
- Memory usage: **<500 MB**

---

## 🎯 Current Status

```
✅ Phase 1: Infrastructure (127 tests passing)
✅ Phase 2: Training script (ready)
🔄 Phase 3: Model training (YOU ARE HERE)
⏳ Phase 4: Integration (30 min after training)
⏳ Phase 5: Validation (1-2 hours)
```

---

## 📚 Files Created

All ready and waiting for you:

| File | Purpose |
|------|---------|
| `ml/training/SIMPLE_COLAB.ipynb` | **→ START HERE** |
| `ml/training/train_audio_standalone.py` | Self-contained training script |
| `ml/scripts/integrate_trained_model.py` | Model integration helper |
| `COLAB_INSTRUCTIONS.md` | Detailed guide |
| `NEXT_STEPS.md` | What to do after training |

---

## 🚀 Ready to Start?

1. **Open VS Code**
2. **Open file:** `ml/training/SIMPLE_COLAB.ipynb`
3. **Connect to Colab** (kernel selector)
4. **Run Cell 1** (upload script)
5. **Run Cell 2** (quick test - 5-10 min)
6. ☕ **Wait**
7. **Run Cell 3** (full training - 1-3 hours)
8. ☕☕☕ **Wait**
9. **Run Cell 4** (download model)
10. 🎉 **Done!**

---

## 💡 Pro Tips

1. **Start with Cell 2 (quick test)** - Don't waste 3 hours if something is wrong!
2. **Keep Colab tab active** - Colab disconnects after ~30 min of inactivity
3. **Monitor training** - Watch the logs, validation accuracy should increase
4. **Save checkpoints** - The script auto-saves best model, but you can also download mid-training

---

**Questions?** Check:
- `COLAB_INSTRUCTIONS.md` for detailed Colab setup
- `ml/training/COLAB_TRAINING_GUIDE.md` for troubleshooting
- `NEXT_STEPS.md` for what to do after training

**Let's train your model!** 🚀
