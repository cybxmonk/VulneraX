# External PyTorch Model Sources

## 🚀 Quick Download Script
Run the provided script to get popular models:
```bash
python download_external_models.py
```

## 📦 Popular Model Sources

### 1. Torchvision Models (Built-in)
```python
import torchvision.models as models

# Image Classification
resnet18 = models.resnet18(pretrained=True)
mobilenet = models.mobilenet_v2(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
alexnet = models.alexnet(pretrained=True)

# Save for VulneraX
torch.save(model, "my_model.pt")
```

### 2. Hugging Face Hub
**Website**: https://huggingface.co/models

**Popular Models**:
- `microsoft/resnet-50`
- `google/vit-base-patch16-224`
- `facebook/convnext-tiny-224`

**Download**:
```bash
pip install huggingface_hub
```
```python
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id="microsoft/resnet-50", filename="pytorch_model.bin")
```

### 3. PyTorch Hub
**Website**: https://pytorch.org/hub/

**Examples**:
```python
# Load from PyTorch Hub
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
torch.save(model, "resnet18_hub.pt")
```

### 4. Model Zoos & Repositories

#### Timm (PyTorch Image Models)
```bash
pip install timm
```
```python
import timm
model = timm.create_model('resnet18', pretrained=True)
torch.save(model, "timm_resnet18.pt")
```

#### MMClassification
```bash
pip install mmcls
```

#### TorchVision Model Zoo
- ResNet family (18, 34, 50, 101, 152)
- VGG family (11, 13, 16, 19)
- DenseNet family
- MobileNet family
- EfficientNet family

### 5. Research Paper Implementations

#### Papers With Code
**Website**: https://paperswithcode.com/
- Browse by task (Image Classification, Object Detection)
- Filter by PyTorch implementations
- Download official model weights

#### GitHub Repositories
- Search: "pytorch pretrained models"
- Look for official implementations
- Check for `.pt` or `.pth` files

## 📊 Sample Datasets

### Built-in PyTorch Datasets
```python
from torchvision import datasets, transforms

# CIFAR-10
cifar10 = datasets.CIFAR10(root='./data', train=False, download=True)

# MNIST
mnist = datasets.MNIST(root='./data', train=False, download=True)

# ImageNet (requires manual download)
imagenet = datasets.ImageNet(root='./data', split='val')
```

### Custom Dataset Creation
```python
import numpy as np

# Create test data
x_test = np.random.rand(100, 3, 224, 224).astype(np.float32)
y_test = np.random.randint(0, 1000, 100)

# Save for VulneraX
data = {'x_test': x_test, 'y_test': y_test}
np.save('my_dataset.npy', data)
```

## 🔧 Model Compatibility

### Input Requirements
- **Image Models**: Usually expect `[batch, 3, 224, 224]`
- **MNIST Models**: Usually expect `[batch, 1, 28, 28]`
- **Custom Models**: Check documentation

### Preprocessing
Most ImageNet models need normalization:
```python
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

### Common Issues & Solutions

#### Issue: Model expects different input size
**Solution**: Resize your data or use adaptive pooling

#### Issue: Wrong number of channels
**Solution**: Convert grayscale↔RGB or adjust model

#### Issue: Model in training mode
**Solution**: Call `model.eval()` before saving

## 📋 Recommended Models for Testing

### Beginner Friendly
1. **ResNet18** - Good balance of size/performance
2. **MobileNetV2** - Lightweight, fast attacks
3. **AlexNet** - Simple architecture, easy to fool

### Advanced Testing
1. **VGG16** - Deep network, interesting vulnerabilities
2. **DenseNet** - Skip connections, different attack patterns
3. **EfficientNet** - Modern architecture, robust features

### Specialized
1. **Vision Transformer (ViT)** - Attention-based, unique vulnerabilities
2. **ConvNeXt** - Modern CNN, good robustness
3. **RegNet** - Efficient design, interesting attack surface

## 🎯 Quick Start Checklist

- [ ] Run `python download_external_models.py`
- [ ] Check `external_models/` folder
- [ ] Verify `external_data/` folder  
- [ ] Open VulneraX app
- [ ] Select "Upload External Model"
- [ ] Choose downloaded model
- [ ] Upload corresponding data
- [ ] Launch attacks!

## 💡 Pro Tips

1. **Start small** - Use ResNet18 before VGG16
2. **Match data** - Use ImageNet data for ImageNet models
3. **Check shapes** - Verify input dimensions match
4. **Monitor memory** - Large models need more RAM
5. **Test locally** - Verify model works before uploading