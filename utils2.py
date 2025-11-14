import torch.nn.functional as F
import numpy as np
import torch


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Registrace hooků (háčků) pro zachycení dat
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        # Ukládáme gradienty výstupu vrstvy
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx, original_length):
        # 1. Forward pass
        self.model.zero_grad()
        output = self.model(x)
        
        # 2. Backward pass z cílové třídy
        score = output[0, class_idx]
        score.backward(retain_graph=False)
        
        # 3. Zpracování gradientů a aktivací
        # Gradients shape: [Batch, Channels, Length] -> např. [1, 64, 128]
        # Global Average Pooling přes časovou dimenzi (Length) -> [1, 64, 1]
        pooled_gradients = torch.mean(self.gradients, dim=2, keepdim=True)
        
        # 4. Váhování aktivací gradienty
        # [1, 64, 128] * [1, 64, 1] -> broadcasting
        weighted_activations = self.activations * pooled_gradients
        # weighted_activations = self.activations * self.gradients

        
        # 5. Průměr přes kanály (vytvoření heatmapy) -> [1, 128]
        heatmap = torch.mean(weighted_activations, dim=2)
        
        # 6. ReLU (chceme jen pozitivní vliv)
        heatmap = F.relu(heatmap)
        
        # 7. Normalizace heatmapy (volitelné, ale dobré pro stabilitu)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
            
        # 8. Upsampling na původní délku okna (40 000)
        # Heatmapa je nyní malá (z latentního prostoru), musíme ji roztáhnout
        heatmap = heatmap.unsqueeze(0) # Přidat dimenzi pro interpolate [1, 1, Length]
        
        heatmap = F.interpolate(heatmap, size=original_length, mode='linear', align_corners=False)
        
        return heatmap.squeeze().cpu().numpy(), output
    
def normalize_signal(signal):
    """Normalizuje signál."""
    signal = np.array(signal, dtype=np.float32)
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    if std_val != 0:
        normalized_signal = (signal - mean_val) / std_val
    else:
        normalized_signal = signal - mean_val
    return normalized_signal