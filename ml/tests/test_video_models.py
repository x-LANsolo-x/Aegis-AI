"""
Tests for video deepfake detection models.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.training.models.video_detector import (
    XceptionVideoDetector,
    LightweightVideoDetector,
    create_video_detector,
    count_parameters,
    get_model_info,
)


# ============================================================================
# Model Creation Tests
# ============================================================================

def test_create_xception_model():
    """Test Xception model creation."""
    model = create_video_detector("xception")
    
    assert model is not None
    assert isinstance(model, XceptionVideoDetector)


def test_create_lightweight_model():
    """Test lightweight model creation."""
    model = create_video_detector("lightweight")
    
    assert model is not None
    assert isinstance(model, LightweightVideoDetector)


def test_invalid_architecture():
    """Test that invalid architecture raises error."""
    with pytest.raises(ValueError):
        create_video_detector("invalid_arch")


# ============================================================================
# Forward Pass Tests
# ============================================================================

def test_xception_forward_pass():
    """Test Xception forward pass with correct input size."""
    model = create_video_detector("xception")
    model.eval()
    
    # Xception expects 299x299 input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 299, 299)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    assert output.shape == (batch_size, 2)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_lightweight_forward_pass():
    """Test lightweight model forward pass."""
    model = create_video_detector("lightweight")
    model.eval()
    
    # Lightweight can handle various sizes, test with 224x224
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    assert output.shape == (batch_size, 2)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_single_image_inference():
    """Test inference with single image."""
    model = create_video_detector("xception")
    model.eval()
    
    input_tensor = torch.randn(1, 3, 299, 299)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    assert output.shape == (1, 2)


def test_batch_inference():
    """Test inference with different batch sizes."""
    model = create_video_detector("lightweight")
    model.eval()
    
    for batch_size in [1, 2, 8, 16]:
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (batch_size, 2)


# ============================================================================
# Model Properties Tests
# ============================================================================

def test_count_parameters():
    """Test parameter counting."""
    model = create_video_detector("lightweight")
    num_params = count_parameters(model)
    
    assert num_params > 0
    assert isinstance(num_params, int)


def test_xception_parameter_count():
    """Test Xception has reasonable parameter count."""
    model = create_video_detector("xception")
    num_params = count_parameters(model)
    
    # Xception should have several million parameters
    assert 1_000_000 < num_params < 50_000_000


def test_lightweight_parameter_count():
    """Test lightweight model is actually lightweight."""
    model = create_video_detector("lightweight")
    num_params = count_parameters(model)
    
    # Lightweight should be < 5M parameters
    assert num_params < 5_000_000


def test_get_model_info():
    """Test model info extraction."""
    model = create_video_detector("xception")
    info = get_model_info(model)
    
    assert "architecture" in info
    assert "parameters" in info
    assert "parameters_mb" in info
    assert info["architecture"] == "XceptionVideoDetector"
    assert info["parameters"] > 0


# ============================================================================
# Gradient Tests
# ============================================================================

def test_gradients_flow():
    """Test that gradients flow through the model."""
    model = create_video_detector("lightweight")
    model.train()
    
    input_tensor = torch.randn(2, 3, 224, 224, requires_grad=True)
    target = torch.tensor([0, 1])
    
    output = model(input_tensor)
    loss = torch.nn.functional.cross_entropy(output, target)
    loss.backward()
    
    # Check that gradients exist
    assert input_tensor.grad is not None
    
    # Check that model parameters have gradients
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_no_gradients_in_eval():
    """Test that no gradients in eval mode."""
    model = create_video_detector("xception")
    model.eval()
    
    input_tensor = torch.randn(2, 3, 299, 299)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    # Should not raise error and output should not require grad
    assert not output.requires_grad


# ============================================================================
# Output Range Tests
# ============================================================================

def test_output_is_logits():
    """Test that output is raw logits (not probabilities)."""
    model = create_video_detector("xception")
    model.eval()
    
    input_tensor = torch.randn(4, 3, 299, 299)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    # Logits can be any real number
    # After softmax, should be valid probabilities
    probs = torch.softmax(output, dim=1)
    
    assert torch.all((probs >= 0) & (probs <= 1))
    assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)


# ============================================================================
# Device Tests
# ============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_on_cuda():
    """Test model runs on CUDA."""
    model = create_video_detector("lightweight")
    model = model.cuda()
    
    input_tensor = torch.randn(2, 3, 224, 224).cuda()
    
    with torch.no_grad():
        output = model(input_tensor)
    
    assert output.is_cuda
    assert output.shape == (2, 2)


def test_model_on_cpu():
    """Test model runs on CPU."""
    model = create_video_detector("xception")
    model = model.cpu()
    
    input_tensor = torch.randn(2, 3, 299, 299).cpu()
    
    with torch.no_grad():
        output = model(input_tensor)
    
    assert not output.is_cuda
    assert output.shape == (2, 2)


# ============================================================================
# Robustness Tests
# ============================================================================

def test_all_zeros_input():
    """Test model handles all-zeros input."""
    model = create_video_detector("lightweight")
    model.eval()
    
    input_tensor = torch.zeros(1, 3, 224, 224)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    assert output.shape == (1, 2)
    assert not torch.isnan(output).any()


def test_all_ones_input():
    """Test model handles all-ones input."""
    model = create_video_detector("xception")
    model.eval()
    
    input_tensor = torch.ones(1, 3, 299, 299)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    assert output.shape == (1, 2)
    assert not torch.isnan(output).any()


def test_random_noise_input():
    """Test model handles noisy input."""
    model = create_video_detector("lightweight")
    model.eval()
    
    # Extreme noise
    input_tensor = torch.randn(2, 3, 224, 224) * 100
    
    with torch.no_grad():
        output = model(input_tensor)
    
    assert output.shape == (2, 2)
    # Model should still produce valid output (no NaN/Inf)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


# ============================================================================
# Model State Tests
# ============================================================================

def test_model_save_load():
    """Test model state can be saved and loaded."""
    model = create_video_detector("lightweight")
    
    # Save state
    state_dict = model.state_dict()
    
    # Create new model and load state
    new_model = create_video_detector("lightweight")
    new_model.load_state_dict(state_dict)
    
    # Test they produce same output
    model.eval()
    new_model.eval()
    
    input_tensor = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        output1 = model(input_tensor)
        output2 = new_model(input_tensor)
    
    assert torch.allclose(output1, output2, atol=1e-6)


def test_model_checkpoint():
    """Test full checkpoint save/load."""
    import tempfile
    
    model = create_video_detector("xception")
    optimizer = torch.optim.Adam(model.parameters())
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": 10,
        "loss": 0.5,
    }
    
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(checkpoint, f.name)
        
        # Load checkpoint
        loaded = torch.load(f.name)
        
        assert "model_state_dict" in loaded
        assert "optimizer_state_dict" in loaded
        assert loaded["epoch"] == 10
        assert loaded["loss"] == 0.5
        
        # Cleanup
        Path(f.name).unlink()


# ============================================================================
# Edge Cases
# ============================================================================

def test_dropout_disabled_in_eval():
    """Test that dropout is disabled in eval mode."""
    model = create_video_detector("xception", dropout=0.9)  # High dropout
    
    input_tensor = torch.randn(4, 3, 299, 299)
    
    # In eval mode, dropout should be disabled
    model.eval()
    with torch.no_grad():
        output1 = model(input_tensor)
        output2 = model(input_tensor)
    
    # Same input should give same output in eval mode
    assert torch.allclose(output1, output2)


def test_dropout_active_in_train():
    """Test that dropout is active in training mode."""
    model = create_video_detector("lightweight", dropout=0.9)
    
    input_tensor = torch.randn(4, 3, 224, 224)
    
    # In train mode, dropout should be active
    model.train()
    with torch.no_grad():
        output1 = model(input_tensor)
        output2 = model(input_tensor)
    
    # Outputs should be different due to dropout
    # (This test might occasionally fail if dropout happens to produce same result)
    assert not torch.allclose(output1, output2, atol=1e-3)
