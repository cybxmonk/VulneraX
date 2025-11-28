import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import io
from pathlib import Path

# Model Definitions
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DeepCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DeepCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SimpleMLP(nn.Module):
    def __init__(self, input_size=784, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Attack Functions
def fgsm_attack(model, x, epsilon=0.3):
    """FGSM Attack"""
    x = x.clone().detach().requires_grad_(True)
    
    with torch.enable_grad():
        output = model(x)
        target_labels = torch.argmax(output, dim=1)
        loss = F.cross_entropy(output, target_labels)
        model.zero_grad()
        loss.backward()
        x_adv = x + epsilon * x.grad.sign()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    return x_adv.detach()

def pgd_attack(model, x, epsilon=0.3, alpha=0.01, num_iter=40):
    """PGD Attack"""
    x_adv = x.clone().detach()
    
    for i in range(num_iter):
        x_adv.requires_grad_(True)
        
        with torch.enable_grad():
            output = model(x_adv)
            with torch.no_grad():
                orig_output = model(x)
                target_labels = torch.argmax(orig_output, dim=1)
            
            loss = F.cross_entropy(output, target_labels)
            loss.backward()
        
        x_adv = x_adv + alpha * x_adv.grad.sign()
        eta = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = torch.clamp(x + eta, 0.0, 1.0).detach()
    
    return x_adv

# Utility Functions
def load_model(model_source, model_name=None, uploaded_file=None):
    """Load model from built-in or external source"""
    if model_source == "built-in":
        model_classes = {
            "simple_cnn.pt": SimpleCNN,
            "deep_cnn.pt": DeepCNN,
            "simple_mlp.pt": SimpleMLP
        }
        
        model_class = model_classes[model_name]
        model = model_class()
        
        model_path = Path("test_models") / model_name
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        
        return model
    
    elif model_source == "external":
        try:
            # First try loading as full model (for pre-trained models)
            model = torch.load(uploaded_file, map_location='cpu', weights_only=False)
            if hasattr(model, 'eval'):
                model.eval()
            return model
        except Exception as e:
            try:
                # Fallback: try loading as state dict with weights_only=True
                state_dict = torch.load(uploaded_file, map_location='cpu', weights_only=True)
                return state_dict
            except:
                raise Exception(f"Failed to load model: {str(e)}")

def load_data_safely(file_source):
    """Safely load numpy data from file path or uploaded file"""
    if isinstance(file_source, (str, Path)):
        data = np.load(file_source, allow_pickle=True)
    else:
        # Handle uploaded file
        data = np.load(file_source, allow_pickle=True)
    
    if isinstance(data, np.ndarray) and data.shape == ():
        data = data.item()
    
    if isinstance(data, dict):
        # Try common key names
        x_data = None
        y_data = None
        
        # Check for x_data
        for key in ['x_test', 'x', 'data', 'X']:
            if key in data and data[key] is not None:
                x_data = data[key]
                break
        
        # Check for y_data  
        for key in ['y_test', 'y', 'labels', 'Y']:
            if key in data and data[key] is not None:
                y_data = data[key]
                break
        
        if x_data is None:
            # If no standard keys, use first array-like value
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    x_data = value
                    break
        
        if x_data is None:
            raise ValueError("No valid data found in file")
        
        # Ensure x_data is numpy array
        if not isinstance(x_data, np.ndarray):
            x_data = np.array(x_data)
        
        x_tensor = torch.tensor(x_data.astype(np.float32), dtype=torch.float32)
        return x_tensor, y_data
    else:
        # Handle direct array
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        x_tensor = torch.tensor(data.astype(np.float32), dtype=torch.float32)
        return x_tensor, None

def main():
    st.set_page_config(
        page_title="VulneraX",
        page_icon="🛡️",
        layout="wide"
    )
    
    st.title("🛡️ VulneraX - Adversarial Attack Generator")
    st.markdown("**Test neural network robustness against adversarial attacks**")
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model Selection
    st.sidebar.subheader("Model")
    model_source = st.sidebar.radio("Model Source:", ["Built-in Models", "Upload External Model"])
    
    model = None
    if model_source == "Built-in Models":
        available_models = ["simple_cnn.pt", "deep_cnn.pt", "simple_mlp.pt"]
        selected_model = st.sidebar.selectbox("Choose model:", available_models)
        model_info_text = "Using built-in model"
    else:
        uploaded_model = st.sidebar.file_uploader(
            "Upload PyTorch Model (.pt, .pth)", 
            type=["pt", "pth"],
            help="Upload a PyTorch model file"
        )
        selected_model = None
        model_info_text = "External model uploaded" if uploaded_model else "No model uploaded"
    
    st.sidebar.info(model_info_text)
    
    # Dataset Selection
    st.sidebar.subheader("Dataset")
    data_source = st.sidebar.radio("Data Source:", ["Built-in Datasets", "Upload External Data"])
    
    if data_source == "Built-in Datasets":
        data_dir = Path("test_data")
        available_datasets = [f.name for f in data_dir.glob("*.npy")] if data_dir.exists() else []
        
        if not available_datasets:
            st.error("No test datasets found!")
            return
        
        selected_dataset = st.sidebar.selectbox("Choose dataset:", available_datasets)
        uploaded_data = None
    else:
        uploaded_data = st.sidebar.file_uploader(
            "Upload Data (.npy, .npz)",
            type=["npy", "npz"],
            help="Upload numpy array with test data"
        )
        selected_dataset = None
    
    # Attack Configuration
    st.sidebar.subheader("Attack")
    attack_type = st.sidebar.selectbox("Type:", ["FGSM", "PGD"])
    epsilon = st.sidebar.slider("Epsilon", 0.0, 1.0, 0.3, 0.01)
    
    if attack_type == "PGD":
        num_iter = st.sidebar.slider("Iterations", 1, 100, 40)
        alpha = st.sidebar.slider("Step Size", 0.001, 0.1, 0.01, 0.001)
    
    # Main Content
    if st.sidebar.button("🚀 Launch Attack", type="primary"):
        try:
            # Load model
            with st.spinner("Loading model..."):
                if model_source == "Built-in Models":
                    model = load_model("built-in", selected_model)
                else:
                    if uploaded_model is None:
                        st.error("Please upload a model file")
                        return
                    model = load_model("external", uploaded_file=uploaded_model)
                    if isinstance(model, dict):
                        st.error("State dict uploaded. Please provide model architecture or use built-in models.")
                        return
                    
                    # Ensure model is in eval mode
                    if hasattr(model, 'eval'):
                        model.eval()
            
            # Load data
            with st.spinner("Loading data..."):
                if data_source == "Built-in Datasets":
                    data_path = Path("test_data") / selected_dataset
                    x_test, y_test = load_data_safely(data_path)
                else:
                    if uploaded_data is None:
                        st.error("Please upload data file")
                        return
                    x_test, y_test = load_data_safely(uploaded_data)
                
                # Auto-detect input shape
                original_shape = x_test.shape
                st.info(f"Data shape: {original_shape}")
                
                # Shape adjustment options
                if x_test.dim() > 2:
                    reshape_option = st.selectbox(
                        "Input format:",
                        ["Keep as images (CNN)", "Flatten (MLP)"],
                        help="Choose how to format input for your model"
                    )
                    
                    if reshape_option == "Flatten (MLP)":
                        x_test = x_test.view(x_test.size(0), -1)
                    elif x_test.dim() == 3:
                        x_test = x_test.unsqueeze(1)
                
                # Ensure data is in correct range [0, 1]
                if x_test.max().item() > 1.0:
                    x_test = x_test / 255.0
                
                # Use first 10 samples
                x_test = x_test[:10]
                st.info(f"Using {len(x_test)} samples with shape: {x_test.shape}")
            
            # Generate attack
            with st.spinner(f"Generating {attack_type} attack..."):
                if attack_type == "FGSM":
                    x_adv = fgsm_attack(model, x_test, epsilon)
                else:
                    x_adv = pgd_attack(model, x_test, epsilon, alpha, num_iter)
            
            # Evaluate
            with torch.no_grad():
                orig_pred = model(x_test)
                adv_pred = model(x_adv)
            
            orig_labels = torch.argmax(orig_pred, dim=1)
            adv_labels = torch.argmax(adv_pred, dim=1)
            
            # Calculate success rate safely
            different_predictions = torch.ne(orig_labels, adv_labels)
            success_rate = different_predictions.float().mean().item()
            
            # Results with explanation
            st.success("Attack completed!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Success Rate", f"{success_rate:.1%}")
            with col2:
                st.metric("Samples", len(x_test))
            with col3:
                st.metric("Epsilon", f"{epsilon:.3f}")
            
            # Explain results
            st.subheader("📊 Results Explanation")
            
            if success_rate >= 0.8:
                st.error(f"**High Vulnerability** ({success_rate:.1%} success rate)")
                st.write("🚨 The model is highly vulnerable to adversarial attacks. Most inputs were successfully fooled with minimal perturbations.")
            elif success_rate >= 0.5:
                st.warning(f"**Moderate Vulnerability** ({success_rate:.1%} success rate)")
                st.write("⚠️ The model shows moderate robustness but still has significant vulnerabilities.")
            elif success_rate >= 0.2:
                st.info(f"**Low Vulnerability** ({success_rate:.1%} success rate)")
                st.write("✅ The model demonstrates good robustness against this attack method.")
            else:
                st.success(f"**Very Robust** ({success_rate:.1%} success rate)")
                st.write("🛡️ The model is highly robust against this attack with current parameters.")
            
            # Attack-specific explanation
            if attack_type == "FGSM":
                st.write(f"**FGSM Attack**: Added noise in gradient direction with strength ε={epsilon:.3f}")
            else:
                st.write(f"**PGD Attack**: Applied {num_iter} iterative steps with ε={epsilon:.3f}, α={alpha:.3f}")
            
            # Recommendations
            st.write("**💡 Recommendations:**")
            if success_rate >= 0.5:
                st.write("- Consider adversarial training to improve robustness")
                st.write("- Try input preprocessing or defensive distillation")
                st.write("- Test with different attack methods for comprehensive evaluation")
            else:
                st.write("- Try stronger attacks (higher ε or more PGD iterations)")
                st.write("- Test other attack methods like C&W or AutoAttack")
                st.write("- Verify robustness on larger datasets")
            
            # Visualizations for image data
            if x_test.dim() == 4 and x_test.shape[1] == 1:
                st.subheader("Sample Results")
                
                num_show = min(5, len(x_test))
                for i in range(num_show):
                    col_orig, col_adv, col_diff = st.columns(3)
                    
                    with col_orig:
                        st.write(f"**Original {i+1}** (Pred: {orig_labels[i].item()})")
                        img_orig = x_test[i].squeeze().numpy()
                        st.image(img_orig, width=150)
                    
                    with col_adv:
                        st.write(f"**Adversarial {i+1}** (Pred: {adv_labels[i].item()})")
                        img_adv = x_adv[i].squeeze().numpy()
                        st.image(img_adv, width=150)
                    
                    with col_diff:
                        attack_success = orig_labels[i].item() != adv_labels[i].item()
                        status = "✅ Attack Success" if attack_success else "❌ Attack Failed"
                        st.write(f"**{status}**")
                        st.write(f"Orig: {orig_labels[i].item()} → Adv: {adv_labels[i].item()}")
                        img_diff = np.abs(img_adv - img_orig)
                        st.image(img_diff, caption="Perturbation", width=150)
            
            # Download
            buf = io.BytesIO()
            np.save(buf, x_adv.numpy())
            st.download_button(
                "📥 Download Results",
                buf.getvalue(),
                file_name=f"adversarial_{attack_type.lower()}.npy"
            )
            
        except Exception as e:
            st.error(f"Attack failed: {str(e)}")
    
    # Info
    with st.expander("ℹ️ Information"):
        st.markdown("""
        **Model Options:**
        - **Built-in**: Pre-trained SimpleCNN, DeepCNN, SimpleMLP
        - **External**: Upload your own PyTorch models (.pt, .pth)
        
        **Data Options:**
        - **Built-in**: Test datasets (MNIST-like, synthetic)
        - **External**: Upload your own numpy arrays (.npy, .npz)
        
        **Attack Types:**
        - **FGSM**: Fast Gradient Sign Method
        - **PGD**: Projected Gradient Descent
        
        **Parameters:**
        - **Epsilon**: Perturbation strength (0-1)
        - **Iterations**: PGD steps
        - **Step Size**: PGD step size
        """)

if __name__ == "__main__":
    main()