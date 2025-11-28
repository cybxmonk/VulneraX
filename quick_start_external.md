# Quick Start: Using External Models with VulneraX

## ✅ Ready-to-Use External Models

Your VulneraX installation now includes popular PyTorch models for testing:

### 📁 Available Models (`external_models/`)
- **resnet18.pt** - ResNet-18 (11.7 MB) - Good for general testing
- **mobilenet_v2.pt** - MobileNetV2 (3.5 MB) - Lightweight, fast attacks  
- **vgg16.pt** - VGG-16 (528 MB) - Deep network, interesting vulnerabilities
- **alexnet.pt** - AlexNet (233 MB) - Classic CNN, easy to fool

### 📊 Compatible Test Data (`external_data/`)
- **imagenet_sample.npy** - 20 ImageNet-like samples (224x224 RGB)
- **cifar_sample.npy** - 50 CIFAR-10-like samples (32x32 RGB)

## 🚀 How to Use

### 1. Start VulneraX
```bash
streamlit run app.py
```

### 2. Upload External Model
1. Select **"Upload External Model"** in sidebar
2. Choose a model from `external_models/` folder
3. Wait for "Model loaded successfully" message

### 3. Upload External Data  
1. Select **"Upload External Data"** in sidebar
2. Choose data from `external_data/` folder
3. Select **"Keep as images (CNN)"** for image models

### 4. Configure Attack
- **Attack Type**: Start with FGSM
- **Epsilon**: Try 0.1-0.3 for visible effects
- **For PGD**: 20-40 iterations, 0.01 step size

### 5. Launch Attack
- Click **"🚀 Launch Attack"**
- View results and download adversarial examples

## 💡 Recommended Combinations

### Beginner Testing
- **Model**: `resnet18.pt` 
- **Data**: `imagenet_sample.npy`
- **Attack**: FGSM, ε=0.3

### Advanced Testing  
- **Model**: `vgg16.pt`
- **Data**: `imagenet_sample.npy` 
- **Attack**: PGD, ε=0.1, 40 iterations

### Quick Testing
- **Model**: `mobilenet_v2.pt`
- **Data**: `cifar_sample.npy`
- **Attack**: FGSM, ε=0.2

## 🔧 Troubleshooting

**Model loading fails?**
- Ensure you selected the correct .pt file
- Try a smaller model first (mobilenet_v2.pt)

**Data shape errors?**
- Always select "Keep as images (CNN)" for these models
- All provided models expect image input

**Low attack success?**
- Increase epsilon value
- Try PGD instead of FGSM
- Some models are naturally more robust

## 📈 Expected Results

- **ResNet18**: 60-80% attack success with ε=0.3
- **MobileNetV2**: 70-90% attack success (less robust)
- **VGG16**: 50-70% attack success (more robust)
- **AlexNet**: 80-95% attack success (easiest to fool)

## 🎯 Next Steps

1. **Try different epsilon values** (0.05, 0.1, 0.2, 0.3)
2. **Compare FGSM vs PGD** attack effectiveness
3. **Test model robustness** across different architectures
4. **Upload your own models** following the external model guide

Happy testing! 🛡️