import torch
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path
import requests
import os

def download_pretrained_models():
    """Download popular PyTorch models for testing"""
    
    # Create directory for external models
    Path("external_models").mkdir(exist_ok=True)
    
    print("Downloading PyTorch models for testing...")
    
    # 1. ResNet18 (Image Classification)
    print("1. Downloading ResNet18...")
    resnet18 = models.resnet18(pretrained=True)
    resnet18.eval()
    torch.save(resnet18, "external_models/resnet18.pt")
    print("   Saved: resnet18.pt")
    
    # 2. MobileNet V2 (Lightweight)
    print("2. Downloading MobileNetV2...")
    mobilenet = models.mobilenet_v2(pretrained=True)
    mobilenet.eval()
    torch.save(mobilenet, "external_models/mobilenet_v2.pt")
    print("   Saved: mobilenet_v2.pt")
    
    # 3. VGG16 (Classic CNN)
    print("3. Downloading VGG16...")
    vgg16 = models.vgg16(pretrained=True)
    vgg16.eval()
    torch.save(vgg16, "external_models/vgg16.pt")
    print("   Saved: vgg16.pt")
    
    # 4. AlexNet (Historical)
    print("4. Downloading AlexNet...")
    alexnet = models.alexnet(pretrained=True)
    alexnet.eval()
    torch.save(alexnet, "external_models/alexnet.pt")
    print("   Saved: alexnet.pt")

def download_sample_datasets():
    """Download sample datasets for testing"""
    
    Path("external_data").mkdir(exist_ok=True)
    
    print("\nCreating sample datasets...")
    
    # ImageNet sample (224x224 RGB images)
    import numpy as np
    
    # 1. ImageNet-like data (224x224x3)
    imagenet_data = np.random.rand(20, 3, 224, 224).astype(np.float32)
    imagenet_labels = np.random.randint(0, 1000, 20)
    
    imagenet_dict = {
        'x_test': imagenet_data,
        'y_test': imagenet_labels
    }
    np.save("external_data/imagenet_sample.npy", imagenet_dict)
    print("   Created: imagenet_sample.npy (20 samples, 224x224x3)")
    
    # 2. CIFAR-10 like data (32x32x3)
    cifar_data = np.random.rand(50, 3, 32, 32).astype(np.float32)
    cifar_labels = np.random.randint(0, 10, 50)
    
    cifar_dict = {
        'x_test': cifar_data,
        'y_test': cifar_labels
    }
    np.save("external_data/cifar_sample.npy", cifar_dict)
    print("   Created: cifar_sample.npy (50 samples, 32x32x3)")

def download_from_huggingface():
    """Download models from Hugging Face Hub"""
    
    try:
        from huggingface_hub import hf_hub_download
        
        print("\n3. Downloading from Hugging Face...")
        
        # Download a simple CNN model
        model_path = hf_hub_download(
            repo_id="microsoft/resnet-50",
            filename="pytorch_model.bin"
        )
        print(f"   Downloaded: {model_path}")
        
    except ImportError:
        print("   Hugging Face Hub not installed. Skipping...")
        print("   Install with: pip install huggingface_hub")

def create_model_info():
    """Create info file for external models"""
    
    model_info = {
        "resnet18.pt": {
            "description": "ResNet-18 for ImageNet classification",
            "input_shape": [3, 224, 224],
            "num_classes": 1000,
            "preprocessing": "ImageNet normalization required"
        },
        "mobilenet_v2.pt": {
            "description": "MobileNetV2 lightweight model",
            "input_shape": [3, 224, 224], 
            "num_classes": 1000,
            "preprocessing": "ImageNet normalization required"
        },
        "vgg16.pt": {
            "description": "VGG-16 deep convolutional network",
            "input_shape": [3, 224, 224],
            "num_classes": 1000,
            "preprocessing": "ImageNet normalization required"
        },
        "alexnet.pt": {
            "description": "AlexNet classic CNN architecture",
            "input_shape": [3, 224, 224],
            "num_classes": 1000,
            "preprocessing": "ImageNet normalization required"
        }
    }
    
    import json
    with open("external_models/model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    print("   Created: model_info.json")

def main():
    print("=" * 60)
    print("VulneraX External Model Downloader")
    print("=" * 60)
    
    # Download models
    download_pretrained_models()
    
    # Create sample data
    download_sample_datasets()
    
    # Create model info
    create_model_info()
    
    # Try Hugging Face (optional)
    download_from_huggingface()
    
    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)
    
    print("\nFiles created:")
    print("📁 external_models/")
    print("   - resnet18.pt (11.7 MB)")
    print("   - mobilenet_v2.pt (3.5 MB)")
    print("   - vgg16.pt (528 MB)")
    print("   - alexnet.pt (233 MB)")
    print("   - model_info.json")
    
    print("\n📁 external_data/")
    print("   - imagenet_sample.npy (224x224 RGB)")
    print("   - cifar_sample.npy (32x32 RGB)")
    
    print("\n🚀 Usage:")
    print("1. Run VulneraX app: streamlit run app.py")
    print("2. Select 'Upload External Model'")
    print("3. Choose a model from external_models/")
    print("4. Select 'Upload External Data'") 
    print("5. Choose data from external_data/")
    print("6. Launch adversarial attacks!")

if __name__ == "__main__":
    main()