import pytest
import torch
from torch.nn import functional as F

from mnr_quantum import HybridModel, SimpleCNN
from mnr_quantum.demo import make_demo_batch


@pytest.mark.parametrize("shape", [(1, 1, 28, 28), (3, 1, 16, 16), (2, 1, 24, 32), (1, 1, 4, 4)])
def test_grayscale_shapes_keep_batch_dimension(shape):
    logits = HybridModel()(torch.randn(shape))
    assert logits.shape == (shape[0], 10)
    assert torch.isfinite(logits).all()


def test_rgb_and_configurable_class_count():
    logits = HybridModel(n_qubits=2, num_classes=3, in_channels=3)(torch.randn(2, 3, 16, 20))
    assert logits.shape == (2, 3)


def test_batch_matches_individual_images():
    torch.manual_seed(5)
    model = HybridModel(n_qubits=3).eval()
    images = torch.randn(3, 1, 16, 16)
    with torch.no_grad():
        batch = model(images)
        individual = torch.cat([model(image.unsqueeze(0)) for image in images])
    torch.testing.assert_close(batch, individual)


def test_training_updates_cnn_projection_and_classifier():
    torch.manual_seed(7)
    model = HybridModel(n_qubits=2, num_classes=2)
    images, labels = make_demo_batch(4, 16, 7)
    names = ["cnn.conv1.weight", "cnn.conv2.weight", "cnn.fc1.weight", "fc2.weight", "fc3.weight"]
    parameters = dict(model.named_parameters())
    before = {name: parameters[name].detach().clone() for name in names}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = F.cross_entropy(model(images), labels)
    loss.backward()
    for name in names:
        gradient = parameters[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert gradient.abs().sum() > 0, name
    optimizer.step()
    for name in names:
        assert not torch.equal(before[name], parameters[name]), name


def test_state_dict_round_trip():
    model = HybridModel(n_qubits=2, num_classes=3)
    restored = HybridModel(n_qubits=2, num_classes=3)
    restored.load_state_dict(model.state_dict())
    images = torch.randn(2, 1, 16, 16)
    torch.testing.assert_close(model(images), restored(images))


@pytest.mark.parametrize("shape", [(0, 1, 16, 16), (2, 16, 16), (1, 3, 16, 16), (1, 1, 3, 16)])
def test_rejects_invalid_image_shape(shape):
    with pytest.raises(ValueError, match="images|height"):
        HybridModel()(torch.zeros(shape))


def test_rejects_integer_images():
    with pytest.raises(TypeError, match="floating-point"):
        HybridModel()(torch.ones(1, 1, 16, 16, dtype=torch.int64))


@pytest.mark.parametrize("kwargs", [{"in_channels": 0}, {"feature_dim": 0}])
def test_invalid_encoder_configuration(kwargs):
    with pytest.raises(ValueError):
        SimpleCNN(**kwargs)


@pytest.mark.parametrize("num_classes", [0, 1, 1.5, True])
def test_invalid_class_count(num_classes):
    with pytest.raises(ValueError, match="num_classes"):
        HybridModel(num_classes=num_classes)
