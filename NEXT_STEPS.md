# Next Steps: Model Training & Integration

## Summary of Work Completed

✅ **Complete ML Training Pipeline Created**
- Full-featured training script with PyTorch/ONNX support
- Google Colab integration for cloud training (free GPU)
- Model integration and testing utilities
- Documentation and guides

**Files Created:**
- `ml/training/train_audio.py` - Complete training script
- `ml/training/train_audio_colab.ipynb` - Colab notebook
- `ml/training/COLAB_TRAINING_GUIDE.md` - Comprehensive guide
- `ml/scripts/integrate_trained_model.py` - Model integration helper
- `ml/scripts/create_golden_samples.py` - Test sample generator

**Tests Updated:**
- `test_inference.py` now has real model tests (auto-skip until model exists)
- Golden samples test validates accuracy ≥70%

**Current Status:**
- ✅ Backend API: 127 tests passing
- ✅ Training infrastructure: Complete
- ⏳ Model training: Ready to run on Colab
- ⏳ Model integration: Awaiting trained model

---

## 🚀 Next Steps (Action Required)

### Step 1: Train Model on Google Colab

**Time Required:** 1-3 hours (mostly waiting for training)

1. **Open Google Colab:**
   - Go to https://colab.research.google.com/
   - Upload `ml/training/train_audio_colab.ipynb`
   
2. **Enable GPU:**
   - Runtime → Change runtime type → GPU (T4)
   
3. **Follow the Notebook:**
   - Run cells in order
   - Start with Cell 7 (quick test, 5-10 min) to verify everything works
   - Then run Cell 8 (full training, 1-3 hours)
   
4. **Download Trained Files:**
   - Cell 10 will let you download:
     - `V1.0.0.onnx` (trained model)
     - `V1.0.0.json` (metadata)
     - `V1.0.0_best.pt` (PyTorch checkpoint, optional)

**Alternative:** If you want to use the Python script directly (not notebook):
```python
# In Colab cell:
!git clone https://github.com/your-username/aegis-ai.git
%cd aegis-ai
!python -m ml.training.train_audio \
    --manifest ml/datasets/manifests/asvspoof_2019.jsonl \
    --output-dir models/audio \
    --model-version V1.0.0 \
    --epochs 20 \
    --batch-size 32 \
    --device cuda
```

---

### Step 2: Integrate Model Locally

**Time Required:** 5 minutes

Once you have the trained model downloaded:

```bash
# Option A: Use the integration script (recommended)
python ml/scripts/integrate_trained_model.py --model ~/Downloads/V1.0.0.onnx

# Option B: Manual integration
cp ~/Downloads/V1.0.0.onnx models/audio/
cp ~/Downloads/V1.0.0.json models/audio/
cd models/audio
# On Windows:
mklink latest.onnx V1.0.0.onnx
# On Linux/Mac:
ln -s V1.0.0.onnx latest.onnx
```

---

### Step 3: Test the Model

**Time Required:** 2 minutes

```bash
# Set environment variable
export ONNX_MODEL_PATH=models/audio/latest.onnx  # Linux/Mac
# Or on Windows:
set ONNX_MODEL_PATH=models/audio/latest.onnx

# Run inference tests (should now have 0 skipped)
pytest services/api/tests/test_inference.py -v

# Start the API
cd services/api
uvicorn app.main:app --reload

# In another terminal, test the model endpoint
curl http://localhost:8000/v1/models
```

Expected response:
```json
{
  "audio": {
    "current": "V1.0.0",
    "path": "models/audio/latest.onnx"
  }
}
```

---

### Step 4: Create Golden Samples (Optional but Recommended)

**Time Required:** 5 minutes

If you have the ASVspoof dataset locally or downloaded on Colab:

```bash
# On Colab (after dataset download):
python ml/scripts/create_golden_samples.py \
    --manifest ml/datasets/manifests/asvspoof_2019.jsonl \
    --output-dir services/api/tests/fixtures/golden_samples \
    --num-bonafide 3 \
    --num-spoof 3

# Download the golden_samples folder from Colab
# Then on local machine:
pytest services/api/tests/test_inference.py::test_golden_samples_accuracy -v
```

---

### Step 5: Performance Validation

**Time Required:** 10 minutes

```bash
# Latency test (already in test suite)
pytest services/api/tests/test_inference.py::test_memory_stability_200_inferences -v -s

# Manual latency check
time curl -X POST http://localhost:8000/v1/analysis/{analysis_id}/run
```

**Expected Performance:**
- Inference latency: < 100ms per 10s audio
- Memory stable: < 50 MB growth over 200 inferences
- Validation accuracy: 85-95%

---

## 📋 Checklist

- [ ] Upload notebook to Colab
- [ ] Enable GPU runtime
- [ ] Run quick test training (Cell 7)
- [ ] Run full training (Cell 8)
- [ ] Download trained model files
- [ ] Integrate model locally
- [ ] Verify `/v1/models` endpoint
- [ ] Run inference tests (should pass, not skip)
- [ ] Create golden samples (optional)
- [ ] Test with real audio files
- [ ] Validate performance metrics

---

## 🎯 Expected Results

After completing all steps:
- ✅ Trained ONNX model at `models/audio/V1.0.0.onnx`
- ✅ API serving real deepfake detection (not dummy)
- ✅ All 129 tests passing (127 backend + 2 inference)
- ✅ Inference latency < 2 seconds
- ✅ Model accuracy 85-95% on validation set

---

## 📚 Resources

- **Colab Training Guide:** `ml/training/COLAB_TRAINING_GUIDE.md`
- **Training Script:** `ml/training/train_audio.py`
- **Colab Notebook:** `ml/training/train_audio_colab.ipynb`
- **Integration Script:** `ml/scripts/integrate_trained_model.py`
- **Model Documentation:** `README.md` (Model Deployment section)

---

## 🆘 Troubleshooting

### "Out of Memory" on Colab
- Reduce batch size: `--batch-size 16`
- Or use quick test mode first

### Download is slow
- Colab has good bandwidth (~100 Mbps)
- ASVspoof is 7.6 GB, takes 10-30 minutes
- Consider using a subset for initial testing

### Model won't load locally
- Check ONNX Runtime is installed: `pip install onnxruntime`
- Verify file path in environment variable
- Check file isn't corrupted (should be ~10-50 MB)

### Tests still skipping
- Ensure `models/audio/latest.onnx` exists
- Set `ONNX_MODEL_PATH` environment variable
- Check file permissions

---

## 💡 Tips

1. **Start with quick test:** Run Cell 7 in the notebook first (5-10 min) to verify everything works before committing to full training (1-3 hours)

2. **Save your work:** Colab can disconnect. The notebook auto-saves checkpoints, but download the model as soon as training completes.

3. **GPU allocation:** Colab limits GPU usage. If disconnected, you may need to wait or use a different account.

4. **Version control:** Keep track of model versions. The script uses semantic versioning (V1.0.0, V1.1.0, etc.)

5. **Iterate:** Once you have V1.0.0 working, you can experiment with:
   - Different architectures
   - Hyperparameter tuning
   - Data augmentation
   - Longer training

---

**Ready to start?** Open `ml/training/train_audio_colab.ipynb` in Google Colab and follow the cells! 🚀
