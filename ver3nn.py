print(f"   Estimated size: {estimate_model_size(bnn_lstm):.2f} MB")
    
    print("\n=== HYBRID GAN MODELS ===")
    
    print("\n6. Image GAN:")
    fake_images = image_generator(dummy_noise)
    real_pred = image_discriminator(dummy_images)
    fake_pred = image_discriminator(fake_images.detach())
    print(f"   Generated images shape: {fake_images.shape}")
    print(f"   Real prediction shape: {real_pred.shape}")
    print(f"   Fake prediction shape: {fake_pred.shape}")
    
    g_total, g_trainable = count_parameters(image_generator)
    d_total, d_trainable = count_parameters(image_discriminator)
    print(f"   Generator: {g_total:,} params, {estimate_model_size(image_generator):.2f} MB")
    print(f"   Discriminator: {d_total:,} params, {estimate_model_size(image_discriminator):.2f} MB")
    
    print("\n7. Sequence GAN:")
    fake_sequences = sequence_generator(dummy_noise_small)
    seq_real_pred = sequence_discriminator(dummy_sequences)
    seq_fake_pred = sequence_discriminator(fake_sequences.detach())
    print(f"   Generated sequences shape: {fake_sequences.shape}")
    print(f"   Real prediction shape: {seq_real_pred.shape}")
    print(f"   Fake prediction shape: {seq_fake_pred.shape}")
    
    g_total, g_trainable = count_parameters(sequence_generator)
    d_total, d_trainable = count_parameters(sequence_discriminator)
    print(f"   Generator: {g_total:,} params, {estimate_model_size(sequence_generator):.2f} MB")
    print(f"   Discriminator: {d_total:,} params, {estimate_model_size(sequence_discriminator):.2f} MB")
    
    print("\n8. Tabular GAN:")
    fake_tabular = tabular_generator(dummy_noise_tiny)
    tab_real_pred = tabular_discriminator(dummy_tabular)
    tab_fake_pred = tabular_discriminator(fake_tabular.detach())
    print(f"   Generated tabular shape: {fake_tabular.shape}")
    print(f"   Real prediction shape: {tab_real_pred.shape}")
    print(f"   Fake prediction shape: {tab_fake_pred.shape}")
    
    g_total, g_trainable = count_parameters(tabular_generator)
    d_total, d_trainable = count_parameters(tabular_discriminator)
    print(f"   Generator: {g_total:,} params, {estimate_model_size(tabular_generator):.2f} MB")
    print(f"   Discriminator: {d_total:,} params, {estimate_model_size(tabular_discriminator):.2f} MB")
    
    print("\n=== COMPREHENSIVE HYBRID ARCHITECTURE BENEFITS ===")
    print("✓ ANN Layers: Full precision for critical computations")
    print("✓ BNN Layers: 32x memory reduction, faster inference")
    print("✓ LSTM Layers: Sequential pattern learning and memory")
    print("✓ CNN Layers: Local feature extraction and spatial processing")
    print("✓ GAN Architecture: Generative modeling and data synthesis")
    print("✓ Attention Mechanisms: Focus on important elements")
    print("✓ Spectral Normalization: Improved GAN training stability")
    print("✓ Flexible Configuration: Choose which layers to binarize")
    
    print("\n=== EXTENDED USE CASES ===")
    print("📊 DATA GENERATION:")
    print("  • Synthetic image generation (faces, objects, scenes)")
    print("  • Time series synthesis (financial data, sensor readings)")
    print("  • Tabular data augmentation (privacy-preserving datasets)")
    print("  • Text generation and language modeling")
    
    print("\n🎯 CLASSIFICATION & PREDICTION:")
    print("  • Image classification with memory constraints")
    print("  • Sequence classification (NLP, audio, video)")
    print("  • Time series forecasting and anomaly detection")
    print("  • Real-time inference on edge devices")
    
    print("\n🔧 SPECIALIZED APPLICATIONS:")
    print("  • Medical imaging with privacy constraints")
    print("  • Financial modeling with regulatory requirements")
    print("  • IoT sensor data processing")
    print("  • Autonomous systems with limited compute")
    print("  • Scientific simulation and modeling")
    
    print("\n=== GAN TRAINING STRATEGIES ===")
    print("🏋️ HYBRID TRAINING BENEFITS:")
    print("  • Reduced memory footprint for larger batch sizes")
    print("  • Faster training with binarized operations")
    print("  • Stable gradients with full-precision critical paths")
    print("  • Energy-efficient training for sustainability")
    
    print("\n📈 PERFORMANCE OPTIMIZATIONS:")
    print("  • Strategic layer binarization (avoid input/output layers)")
    print("  • Gradient clipping for BNN stability")
    print("  • Learning rate scheduling for hybrid training")
    print("  • Batch normalization for binarized layers")
    
    print("\n=== MEMORY EFFICIENCY COMPARISON ===")
    
    # Create comparable models for detailed comparison
    full_precision_lstm = nn.LSTM(20, 64, 2, batch_first=True, bidirectional=True)
    total_fp, _ = count_parameters(full_precision_lstm)
    total_bnn, _ = count_parameters(bnn_lstm)
    
    print(f"Full Precision LSTM: {estimate_model_size(full_precision_lstm):.2f} MB")
    print(f"Binarized LSTM: {estimate_model_size(bnn_lstm):.2f} MB")
    print(f"LSTM Memory Reduction: {estimate_model_size(full_precision_lstm) / estimate_model_size(bnn_lstm):.1f}x smaller")
    
    # GAN comparison
    print(f"\nImage GAN Total Size: {estimate_model_size(image_generator) + estimate_model_size(image_discriminator):.2f} MB")
    print(f"Sequence GAN Total Size: {estimate_model_size(sequence_generator) + estimate_model_size(sequence_discriminator):.2f} MB")
    print(f"Tabular GAN Total Size: {estimate_model_size(tabular_generator) + estimate_model_size(tabular_discriminator):.2f} MB")
    
    print("\n=== HYBRID ARCHITECTURE SUMMARY ===")
    print("🔗 COMPONENT INTEGRATION:")
    print(f"   • Total Components: ANN + BNN + LSTM + CNN + GAN")
    print(f"   • Network Types: {len([hybrid_fc, hybrid_cnn, hybrid_sequence, hybrid_timeseries, image_generator, image_discriminator])}")
    print(f"   • Flexibility: Configurable binarization per layer")
    print(f"   • Efficiency: Up to 32x memory reduction")
    print(f"   • Applications: Classification, Generation, Forecasting")
    
    print("\n🚀 DEPLOYMENT ADVANTAGES:")
    print("   • Edge Computing: Reduced model size for mobile/IoT")
    print("   • Cloud Efficiency: Lower inference costs")
    print("   • Real-time Processing: Faster binary operations")
    print("   • Scalability: Memory-efficient large-scale deployment")
    
    print("\n=== IMPLEMENTATION NOTES ===")
    print("⚠️  TRAINING CONSIDERATIONS:")
    print("   • Use straight-through estimator for BNN gradients")
    print("   • Apply batch normalization after binarized layers")
    print("   • Careful initialization for hybrid networks")
    print("   • Monitor gradient flow through different layer types")
    
    print("\n✅ BEST PRACTICES:")
    print("   • Keep input/output layers full precision")
    print("   • Use spectral normalization in discriminators")
    print("   • Apply attention for sequence models")
    print("   • Balance efficiency vs. accuracy trade-offs")import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List

class BinarizeFunction(torch.autograd.Function):
    """Custom function for binarizing weights and activations"""
    
    @staticmethod
    def forward(ctx, input):
        # Binarize to -1 and +1
        return torch.sign(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass gradient through unchanged
        return grad_output

def binarize(input):
    return BinarizeFunction.apply(input)

class BinarizedLinear(nn.Module):
    """Binarized fully connected layer"""
    
    def __init__(self, in_features, out_features, bias=True):
        super(BinarizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Full precision weights for training
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Batch normalization for better training stability
        self.bn = nn.BatchNorm1d(out_features, momentum=0.1, affine=False)
    
    def forward(self, x):
        # Binarize input activations
        if self.training:
            # During training, use straight-through estimator
            binary_input = binarize(x)
        else:
            # During inference, hard binarization
            binary_input = torch.sign(x)
        
        # Binarize weights
        if self.training:
            binary_weight = binarize(self.weight)
        else:
            binary_weight = torch.sign(self.weight)
        
        # Linear transformation with binarized weights and inputs
        output = F.linear(binary_input, binary_weight, self.bias)
        
        # Apply batch normalization
        if output.dim() > 1:
            output = self.bn(output)
        
        return output

class BinarizedLSTMCell(nn.Module):
    """Binarized LSTM Cell with binary weights but full precision states"""
    
    def __init__(self, input_size, hidden_size, bias=True):
        super(BinarizedLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        # Input-to-hidden weights (will be binarized)
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size) * 0.1)
        # Hidden-to-hidden weights (will be binarized)  
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size) * 0.1)
        
        if bias:
            self.bias_ih = nn.Parameter(torch.zeros(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.zeros(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
            
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(4 * hidden_size)
    
    def forward(self, input, hidden):
        hx, cx = hidden
        
        # Binarize weights
        if self.training:
            binary_weight_ih = binarize(self.weight_ih)
            binary_weight_hh = binarize(self.weight_hh)
        else:
            binary_weight_ih = torch.sign(self.weight_ih)
            binary_weight_hh = torch.sign(self.weight_hh)
        
        # Linear transformations with binary weights
        gi = F.linear(input, binary_weight_ih, self.bias_ih)
        gh = F.linear(hx, binary_weight_hh, self.bias_hh)
        i_r, i_i, i_n, i_c = gi.chunk(4, 1)
        h_r, h_i, h_n, h_c = gh.chunk(4, 1)
        
        # Apply layer normalization
        gates = self.layer_norm(gi + gh)
        resetgate, inputgate, newgate, cellgate = gates.chunk(4, 1)
        
        # LSTM computations (keep states in full precision)
        resetgate = torch.sigmoid(resetgate)
        inputgate = torch.sigmoid(inputgate)
        newgate = torch.tanh(newgate)
        cellgate = torch.sigmoid(cellgate)
        
        cy = (cellgate * cx) + (inputgate * newgate)
        hy = resetgate * torch.tanh(cy)
        
        return hy, cy

class BinarizedLSTM(nn.Module):
    """Binarized LSTM layer"""
    
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, 
                 batch_first=False, dropout=0.0, bidirectional=False):
        super(BinarizedLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        num_directions = 2 if bidirectional else 1
        
        self.lstm_cells = nn.ModuleList()
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions
                self.lstm_cells.append(BinarizedLSTMCell(layer_input_size, hidden_size, bias))
        
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
    
    def forward(self, input, hx=None):
        batch_size = input.size(0) if self.batch_first else input.size(1)
        seq_len = input.size(1) if self.batch_first else input.size(0)
        
        if not self.batch_first:
            input = input.transpose(0, 1)  # Convert to batch_first
        
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = (torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size, device=input.device),
                  torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size, device=input.device))
        
        h, c = hx
        outputs = []
        
        for t in range(seq_len):
            x = input[:, t, :]
            new_h = []
            new_c = []
            
            for layer in range(self.num_layers):
                if self.bidirectional:
                    # Forward direction
                    idx = layer * 2
                    h_forward, c_forward = self.lstm_cells[idx](x, (h[idx], c[idx]))
                    
                    # Backward direction (process in reverse)
                    idx = layer * 2 + 1
                    x_backward = input[:, seq_len - 1 - t, :] if layer == 0 else x
                    h_backward, c_backward = self.lstm_cells[idx](x_backward, (h[idx], c[idx]))
                    
                    x = torch.cat([h_forward, h_backward], dim=1)
                    new_h.extend([h_forward, h_backward])
                    new_c.extend([c_forward, c_backward])
                else:
                    h_new, c_new = self.lstm_cells[layer](x, (h[layer], c[layer]))
                    x = h_new
                    new_h.append(h_new)
                    new_c.append(c_new)
                
                if self.dropout_layer and layer < self.num_layers - 1:
                    x = self.dropout_layer(x)
            
            outputs.append(x)
            h = torch.stack(new_h, dim=0)
            c = torch.stack(new_c, dim=0)
        
        output = torch.stack(outputs, dim=1)
        
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        return output, (h, c)

class HybridSequenceModel(nn.Module):
    """Hybrid model combining CNN, LSTM, BNN, and ANN layers for sequence data"""
    
    def __init__(self, input_size, hidden_sizes, lstm_hidden_size, num_classes,
                 sequence_length, use_cnn=True, use_bnn_lstm=False, 
                 bnn_layers=None, dropout_rate=0.2, num_lstm_layers=2):
        super(HybridSequenceModel, self).__init__()
        
        self.use_cnn = use_cnn
        self.use_bnn_lstm = use_bnn_lstm
        self.sequence_length = sequence_length
        
        if bnn_layers is None:
            bnn_layers = []
        
        # CNN feature extractor (optional)
        if use_cnn:
            self.cnn_features = nn.Sequential(
                nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(sequence_length)
            )
            lstm_input_size = 128
        else:
            self.cnn_features = None
            lstm_input_size = input_size
        
        # LSTM layer (regular or binarized)
        if use_bnn_lstm:
            self.lstm = BinarizedLSTM(
                input_size=lstm_input_size,
                hidden_size=lstm_hidden_size,
                num_layers=num_lstm_layers,
                batch_first=True,
                dropout=dropout_rate if num_lstm_layers > 1 else 0.0,
                bidirectional=True
            )
            classifier_input_size = lstm_hidden_size * 2  # Bidirectional
        else:
            self.lstm = nn.LSTM(
                input_size=lstm_input_size,
                hidden_size=lstm_hidden_size,
                num_layers=num_lstm_layers,
                batch_first=True,
                dropout=dropout_rate if num_lstm_layers > 1 else 0.0,
                bidirectional=True
            )
            classifier_input_size = lstm_hidden_size * 2  # Bidirectional
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(classifier_input_size, classifier_input_size // 2),
            nn.Tanh(),
            nn.Linear(classifier_input_size // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Hybrid classifier layers
        self.classifier_layers = nn.ModuleList()
        self.classifier_activations = nn.ModuleList()
        self.classifier_dropouts = nn.ModuleList()
        
        prev_size = classifier_input_size
        all_sizes = [classifier_input_size] + hidden_sizes + [num_classes]
        
        for i in range(len(all_sizes) - 1):
            current_size = all_sizes[i + 1]
            
            # Decide whether to use BNN or ANN layer
            if i in bnn_layers:
                layer = BinarizedLinear(prev_size, current_size)
                activation = nn.Hardtanh(inplace=True)
            else:
                layer = nn.Linear(prev_size, current_size)
                activation = nn.ReLU(inplace=True)
            
            self.classifier_layers.append(layer)
            self.classifier_activations.append(activation)
            
            if i < len(all_sizes) - 2:
                self.classifier_dropouts.append(nn.Dropout(dropout_rate))
            else:
                self.classifier_dropouts.append(nn.Identity())
            
            prev_size = current_size
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN feature extraction
        if self.use_cnn:
            # x should be (batch, sequence, features)
            x = x.transpose(1, 2)  # Convert to (batch, features, sequence)
            x = self.cnn_features(x)
            x = x.transpose(1, 2)  # Convert back to (batch, sequence, features)
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        attended_output = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Classifier
        x = attended_output
        for i, (layer, activation, dropout) in enumerate(zip(
            self.classifier_layers, self.classifier_activations, self.classifier_dropouts)):
            
            x = layer(x)
            
            if i < len(self.classifier_layers) - 1:
                x = activation(x)
                x = dropout(x)
        
        return x

class HybridTimeSeries(nn.Module):
    """Hybrid model specifically designed for time series prediction"""
    
    def __init__(self, input_features, lstm_hidden_size, output_features, 
                 sequence_length, forecast_horizon=1, use_bnn_lstm=False, 
                 num_lstm_layers=2, dropout_rate=0.1):
        super(HybridTimeSeries, self).__init__()
        
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        
        # Feature extraction with hybrid conv layers
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        
        # Add binarized conv layer for efficiency
        self.bnn_conv = BinarizedConv1d(32, 64, kernel_size=3, padding=1)
        
        # LSTM layers
        if use_bnn_lstm:
            self.lstm = BinarizedLSTM(
                input_size=64,
                hidden_size=lstm_hidden_size,
                num_layers=num_lstm_layers,
                batch_first=True,
                dropout=dropout_rate
            )
        else:
            self.lstm = nn.LSTM(
                input_size=64,
                hidden_size=lstm_hidden_size,
                num_layers=num_lstm_layers,
                batch_first=True,
                dropout=dropout_rate
            )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden_size // 2, output_features * forecast_horizon)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Feature extraction
        x = x.transpose(1, 2)  # (batch, features, sequence)
        x = self.feature_extractor(x)
        x = self.bnn_conv(x)
        x = x.transpose(1, 2)  # (batch, sequence, features)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Use last time step output
        last_output = lstm_out[:, -1, :]
        
        # Generate predictions
        output = self.output_projection(last_output)
        
        # Reshape to (batch, forecast_horizon, features)
        output = output.view(batch_size, self.forecast_horizon, -1)
        
        return output

class BinarizedConv1d(nn.Module):
    """Binarized 1D convolutional layer for time series"""
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, bias=True):
        super(BinarizedConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size) * 0.1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.bn = nn.BatchNorm1d(out_channels, momentum=0.1, affine=False)
    
    def forward(self, x):
        if self.training:
            binary_input = binarize(x)
            binary_weight = binarize(self.weight)
        else:
            binary_input = torch.sign(x)
            binary_weight = torch.sign(self.weight)
        
        output = F.conv1d(binary_input, binary_weight, self.bias,
                         self.stride, self.padding)
        output = self.bn(output)
        return output

class HybridGenerator(nn.Module):
    """Hybrid Generator network combining ANN, BNN, and LSTM layers"""
    
    def __init__(self, latent_dim, output_shape, hidden_sizes=None, 
                 use_lstm=False, lstm_hidden_size=128, bnn_layers=None,
                 output_activation='tanh', generator_type='fc'):
        super(HybridGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.use_lstm = use_lstm
        self.generator_type = generator_type
        
        if hidden_sizes is None:
            hidden_sizes = [256, 512, 1024]
        if bnn_layers is None:
            bnn_layers = [0]  # First layer binarized by default
        
        self.bnn_layers = bnn_layers
        
        if generator_type == 'fc':
            self._build_fc_generator(hidden_sizes, output_activation)
        elif generator_type == 'conv':
            self._build_conv_generator(output_activation)
        elif generator_type == 'sequence':
            self._build_sequence_generator(hidden_sizes, lstm_hidden_size, output_activation)
    
    def _build_fc_generator(self, hidden_sizes, output_activation):
        """Build fully connected generator"""
        layers = []
        activations = []
        
        # Calculate output size
        if isinstance(self.output_shape, tuple):
            output_size = np.prod(self.output_shape)
        else:
            output_size = self.output_shape
        
        all_sizes = [self.latent_dim] + hidden_sizes + [output_size]
        
        # LSTM preprocessing (optional)
        if self.use_lstm:
            self.lstm_preprocessor = nn.LSTM(
                input_size=1,
                hidden_size=64,
                num_layers=2,
                batch_first=True
            )
            lstm_output_size = 64
            all_sizes[0] = lstm_output_size
        
        # Build generator layers
        for i in range(len(all_sizes) - 1):
            current_size = all_sizes[i + 1]
            prev_size = all_sizes[i]
            
            # Choose layer type
            if i in self.bnn_layers:
                layer = BinarizedLinear(prev_size, current_size)
                activation = nn.Hardtanh(inplace=True)
            else:
                layer = nn.Linear(prev_size, current_size)
                if i == len(all_sizes) - 2:  # Output layer
                    activation = self._get_output_activation(output_activation)
                else:
                    activation = nn.ReLU(inplace=True)
            
            layers.append(layer)
            activations.append(activation)
        
        self.layers = nn.ModuleList(layers)
        self.activations = nn.ModuleList(activations)
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(size) for size in all_sizes[1:-1]
        ])
    
    def _build_conv_generator(self, output_activation):
        """Build convolutional generator for images"""
        if len(self.output_shape) != 3:  # (C, H, W)
            raise ValueError("Conv generator requires 3D output shape (C, H, W)")
        
        channels, height, width = self.output_shape
        
        # Initial dense layer
        self.initial_dense = nn.Linear(self.latent_dim, 128 * 8 * 8)
        
        # Convolutional layers
        conv_layers = []
        
        # Layer 1: 128 -> 256 channels (upscale 8x8 -> 16x16)
        if 0 in self.bnn_layers:
            conv_layers.extend([
                BinarizedConvTranspose2d(128, 256, 4, 2, 1),
                nn.ReLU(inplace=True)
            ])
        else:
            conv_layers.extend([
                nn.ConvTranspose2d(128, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ])
        
        # Layer 2: 256 -> 128 channels (upscale 16x16 -> 32x32)
        if 1 in self.bnn_layers:
            conv_layers.extend([
                BinarizedConvTranspose2d(256, 128, 4, 2, 1),
                nn.ReLU(inplace=True)
            ])
        else:
            conv_layers.extend([
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ])
        
        # Final layer: 128 -> output channels
        conv_layers.extend([
            nn.ConvTranspose2d(128, channels, 4, 2, 1),
            self._get_output_activation(output_activation)
        ])
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # LSTM for temporal conditioning (optional)
        if self.use_lstm:
            self.lstm_conditioner = BinarizedLSTM(
                input_size=self.latent_dim,
                hidden_size=64,
                num_layers=2,
                batch_first=True
            )
    
    def _build_sequence_generator(self, hidden_sizes, lstm_hidden_size, output_activation):
        """Build sequence generator with LSTM"""
        seq_length, feature_dim = self.output_shape
        
        # Initial projection
        self.initial_projection = nn.Linear(self.latent_dim, lstm_hidden_size)
        
        # LSTM core
        if 'lstm' in [str(i) for i in self.bnn_layers]:
            self.lstm_core = BinarizedLSTM(
                input_size=lstm_hidden_size,
                hidden_size=lstm_hidden_size,
                num_layers=2,
                batch_first=True
            )
        else:
            self.lstm_core = nn.LSTM(
                input_size=lstm_hidden_size,
                hidden_size=lstm_hidden_size,
                num_layers=2,
                batch_first=True
            )
        
        # Output projection layers
        output_layers = []
        prev_size = lstm_hidden_size
        
        for i, size in enumerate(hidden_sizes + [feature_dim]):
            if i in self.bnn_layers and i != len(hidden_sizes):  # Don't binarize final output
                layer = BinarizedLinear(prev_size, size)
                activation = nn.Hardtanh(inplace=True)
            else:
                layer = nn.Linear(prev_size, size)
                if i == len(hidden_sizes):  # Final layer
                    activation = self._get_output_activation(output_activation)
                else:
                    activation = nn.ReLU(inplace=True)
            
            output_layers.extend([layer, activation])
            prev_size = size
        
        self.output_projection = nn.Sequential(*output_layers[:-1])  # Remove last activation
        self.final_activation = output_layers[-1]
    
    def _get_output_activation(self, activation):
        """Get output activation function"""
        if activation == 'tanh':
            return nn.Tanh()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'linear' or activation is None:
            return nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, z, seq_length=None):
        batch_size = z.size(0)
        
        if self.generator_type == 'fc':
            return self._forward_fc(z, batch_size)
        elif self.generator_type == 'conv':
            return self._forward_conv(z, batch_size)
        elif self.generator_type == 'sequence':
            return self._forward_sequence(z, batch_size, seq_length)
    
    def _forward_fc(self, z, batch_size):
        x = z
        
        # LSTM preprocessing
        if self.use_lstm:
            x = x.unsqueeze(-1)  # Add feature dimension
            x, _ = self.lstm_preprocessor(x)
            x = x[:, -1, :]  # Take last output
        
        # Forward through layers
        for i, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            x = layer(x)
            
            # Apply batch norm (except for output layer)
            if i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            
            x = activation(x)
        
        # Reshape to output shape
        if isinstance(self.output_shape, tuple):
            x = x.view(batch_size, *self.output_shape)
        
        return x
    
    def _forward_conv(self, z, batch_size):
        # Initial dense transformation
        x = self.initial_dense(z)
        x = x.view(batch_size, 128, 8, 8)
        
        # LSTM conditioning (optional)
        if self.use_lstm:
            z_seq = z.unsqueeze(1).repeat(1, 10, 1)  # Create sequence
            lstm_out, _ = self.lstm_conditioner(z_seq)
            # Use LSTM output to modulate features (simplified)
            conditioning = lstm_out[:, -1, :].unsqueeze(-1).unsqueeze(-1)
            x = x + conditioning[:, :128]  # Add to first 128 channels
        
        # Convolutional upsampling
        x = self.conv_layers(x)
        
        return x
    
    def _forward_sequence(self, z, batch_size, seq_length):
        if seq_length is None:
            seq_length = self.output_shape[0]
        
        # Initial projection
        x = self.initial_projection(z)
        
        # Expand for sequence generation
        x = x.unsqueeze(1).repeat(1, seq_length, 1)
        
        # LSTM processing
        lstm_out, _ = self.lstm_core(x)
        
        # Generate output sequence
        outputs = []
        for t in range(seq_length):
            step_input = lstm_out[:, t, :]
            step_output = self.output_projection(step_input)
            step_output = self.final_activation(step_output)
            outputs.append(step_output)
        
        return torch.stack(outputs, dim=1)

class HybridDiscriminator(nn.Module):
    """Hybrid Discriminator network combining ANN, BNN, and LSTM layers"""
    
    def __init__(self, input_shape, hidden_sizes=None, use_lstm=False,
                 lstm_hidden_size=128, bnn_layers=None, discriminator_type='fc',
                 use_spectral_norm=False):
        super(HybridDiscriminator, self).__init__()
        
        self.input_shape = input_shape
        self.use_lstm = use_lstm
        self.discriminator_type = discriminator_type
        self.use_spectral_norm = use_spectral_norm
        
        if hidden_sizes is None:
            hidden_sizes = [1024, 512, 256]
        if bnn_layers is None:
            bnn_layers = [1, 2]  # Middle layers binarized
        
        self.bnn_layers = bnn_layers
        
        if discriminator_type == 'fc':
            self._build_fc_discriminator(hidden_sizes)
        elif discriminator_type == 'conv':
            self._build_conv_discriminator()
        elif discriminator_type == 'sequence':
            self._build_sequence_discriminator(hidden_sizes, lstm_hidden_size)
    
    def _build_fc_discriminator(self, hidden_sizes):
        """Build fully connected discriminator"""
        if isinstance(self.input_shape, tuple):
            input_size = np.prod(self.input_shape)
        else:
            input_size = self.input_shape
        
        # LSTM preprocessing (optional)
        if self.use_lstm:
            self.lstm_preprocessor = BinarizedLSTM(
                input_size=input_size,
                hidden_size=128,
                num_layers=2,
                batch_first=True
            )
            input_size = 128
        
        layers = []
        activations = []
        dropouts = []
        
        all_sizes = [input_size] + hidden_sizes + [1]
        
        for i in range(len(all_sizes) - 1):
            current_size = all_sizes[i + 1]
            prev_size = all_sizes[i]
            
            # Choose layer type
            if i in self.bnn_layers:
                layer = BinarizedLinear(prev_size, current_size)
                activation = nn.LeakyReLU(0.2, inplace=True)
            else:
                layer = nn.Linear(prev_size, current_size)
                if self.use_spectral_norm:
                    layer = nn.utils.spectral_norm(layer)
                
                if i == len(all_sizes) - 2:  # Output layer
                    activation = nn.Sigmoid()
                else:
                    activation = nn.LeakyReLU(0.2, inplace=True)
            
            layers.append(layer)
            activations.append(activation)
            
            # Add dropout (except output layer)
            if i < len(all_sizes) - 2:
                dropouts.append(nn.Dropout(0.3))
            else:
                dropouts.append(nn.Identity())
        
        self.layers = nn.ModuleList(layers)
        self.activations = nn.ModuleList(activations)
        self.dropouts = nn.ModuleList(dropouts)
    
    def _build_conv_discriminator(self):
        """Build convolutional discriminator for images"""
        if len(self.input_shape) != 3:
            raise ValueError("Conv discriminator requires 3D input shape (C, H, W)")
        
        channels, height, width = self.input_shape
        
        conv_layers = []
        
        # Layer 1: Input -> 64 channels
        if 0 in self.bnn_layers:
            conv_layers.extend([
                BinarizedConv2d(channels, 64, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        else:
            layer = nn.Conv2d(channels, 64, 4, 2, 1)
            if self.use_spectral_norm:
                layer = nn.utils.spectral_norm(layer)
            conv_layers.extend([
                layer,
                nn.LeakyReLU(0.2, inplace=True)
            ])
        
        # Layer 2: 64 -> 128 channels
        if 1 in self.bnn_layers:
            conv_layers.extend([
                BinarizedConv2d(64, 128, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        else:
            layer = nn.Conv2d(64, 128, 4, 2, 1)
            if self.use_spectral_norm:
                layer = nn.utils.spectral_norm(layer)
            conv_layers.extend([
                layer,
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        
        # Layer 3: 128 -> 256 channels
        if 2 in self.bnn_layers:
            conv_layers.extend([
                BinarizedConv2d(128, 256, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        else:
            layer = nn.Conv2d(128, 256, 4, 2, 1)
            if self.use_spectral_norm:
                layer = nn.utils.spectral_norm(layer)
            conv_layers.extend([
                layer,
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        
        # Final classification layer
        final_layer = nn.Conv2d(256, 1, 4, 1, 0)
        if self.use_spectral_norm:
            final_layer = nn.utils.spectral_norm(final_layer)
        
        conv_layers.extend([
            final_layer,
            nn.Sigmoid()
        ])
        
        self.conv_layers = nn.Sequential(*conv_layers)
    
    def _build_sequence_discriminator(self, hidden_sizes, lstm_hidden_size):
        """Build sequence discriminator with LSTM"""
        seq_length, feature_dim = self.input_shape
        
        # LSTM core
        if 'lstm' in [str(i) for i in self.bnn_layers]:
            self.lstm_core = BinarizedLSTM(
                input_size=feature_dim,
                hidden_size=lstm_hidden_size,
                num_layers=2,
                batch_first=True,
                bidirectional=True
            )
            lstm_output_size = lstm_hidden_size * 2
        else:
            self.lstm_core = nn.LSTM(
                input_size=feature_dim,
                hidden_size=lstm_hidden_size,
                num_layers=2,
                batch_first=True,
                bidirectional=True
            )
            lstm_output_size = lstm_hidden_size * 2
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_size // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Classification layers
        layers = []
        activations = []
        
        prev_size = lstm_output_size
        all_sizes = hidden_sizes + [1]
        
        for i, size in enumerate(all_sizes):
            if i in self.bnn_layers:
                layer = BinarizedLinear(prev_size, size)
                activation = nn.LeakyReLU(0.2, inplace=True)
            else:
                layer = nn.Linear(prev_size, size)
                if self.use_spectral_norm:
                    layer = nn.utils.spectral_norm(layer)
                
                if i == len(all_sizes) - 1:  # Output layer
                    activation = nn.Sigmoid()
                else:
                    activation = nn.LeakyReLU(0.2, inplace=True)
            
            layers.append(layer)
            activations.append(activation)
            prev_size = size
        
        self.classifier_layers = nn.ModuleList(layers)
        self.classifier_activations = nn.ModuleList(activations)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        if self.discriminator_type == 'fc':
            return self._forward_fc(x, batch_size)
        elif self.discriminator_type == 'conv':
            return self._forward_conv(x)
        elif self.discriminator_type == 'sequence':
            return self._forward_sequence(x)
    
    def _forward_fc(self, x, batch_size):
        # Flatten input
        if x.dim() > 2:
            x = x.view(batch_size, -1)
        
        # LSTM preprocessing
        if self.use_lstm:
            x = x.unsqueeze(1)  # Add sequence dimension
            x, _ = self.lstm_preprocessor(x)
            x = x.squeeze(1)  # Remove sequence dimension
        
        # Forward through layers
        for layer, activation, dropout in zip(self.layers, self.activations, self.dropouts):
            x = layer(x)
            x = activation(x)
            x = dropout(x)
        
        return x
    
    def _forward_conv(self, x):
        return self.conv_layers(x).view(x.size(0), -1)
    
    def _forward_sequence(self, x):
        # LSTM processing
        lstm_out, _ = self.lstm_core(x)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        attended_output = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Classification
        x = attended_output
        for layer, activation in zip(self.classifier_layers, self.classifier_activations):
            x = layer(x)
            x = activation(x)
        
        return x

class BinarizedConvTranspose2d(nn.Module):
    """Binarized transpose convolution for generator upsampling"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BinarizedConvTranspose2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size) * 0.1)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1, affine=False)
    
    def forward(self, x):
        if self.training:
            binary_input = binarize(x)
            binary_weight = binarize(self.weight)
        else:
            binary_input = torch.sign(x)
            binary_weight = torch.sign(self.weight)
        
        output = F.conv_transpose2d(binary_input, binary_weight, None,
                                   self.stride, self.padding)
        output = self.bn(output)
        return output

class BinarizedConv2d(nn.Module):
    """Binarized convolutional layer"""
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, bias=True):
        super(BinarizedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Full precision weights for training
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 
                                              kernel_size, kernel_size) * 0.1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Batch normalization
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1, affine=False)
    
    def forward(self, x):
        # Binarize input activations
        if self.training:
            binary_input = binarize(x)
        else:
            binary_input = torch.sign(x)
        
        # Binarize weights
        if self.training:
            binary_weight = binarize(self.weight)
        else:
            binary_weight = torch.sign(self.weight)
        
        # Convolution with binarized weights and inputs
        output = F.conv2d(binary_input, binary_weight, self.bias,
                         self.stride, self.padding)
        
        # Apply batch normalization
        output = self.bn(output)
        
        return output

class HybridANNBNN(nn.Module):
    """Hybrid Neural Network combining ANN and BNN layers"""
    
    def __init__(self, input_size, hidden_sizes, num_classes, 
                 bnn_layers=None, dropout_rate=0.2):
        super(HybridANNBNN, self).__init__()
        
        # If bnn_layers not specified, binarize middle layers
        if bnn_layers is None:
            bnn_layers = list(range(1, len(hidden_sizes)))
        
        self.bnn_layers = bnn_layers
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.activations = nn.ModuleList()
        
        # Input layer dimensions
        prev_size = input_size
        all_sizes = [input_size] + hidden_sizes + [num_classes]
        
        # Build layers
        for i in range(len(all_sizes) - 1):
            current_size = all_sizes[i + 1]
            
            # Decide whether to use BNN or ANN layer
            if i in bnn_layers:
                # Use binarized layer
                layer = BinarizedLinear(prev_size, current_size)
                # Binarized layers use hard tanh activation
                activation = nn.Hardtanh(inplace=True)
            else:
                # Use full precision layer
                layer = nn.Linear(prev_size, current_size)
                # Regular layers use ReLU
                activation = nn.ReLU(inplace=True)
            
            self.layers.append(layer)
            self.activations.append(activation)
            
            # Add dropout (except for output layer)
            if i < len(all_sizes) - 2:
                self.dropouts.append(nn.Dropout(dropout_rate))
            else:
                self.dropouts.append(nn.Identity())
            
            prev_size = current_size
    
    def forward(self, x):
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Forward pass through all layers
        for i, (layer, activation, dropout) in enumerate(zip(
            self.layers, self.activations, self.dropouts)):
            
            x = layer(x)
            
            # Don't apply activation to output layer
            if i < len(self.layers) - 1:
                x = activation(x)
                x = dropout(x)
        
        return x

class HybridCNN(nn.Module):
    """Hybrid CNN combining ANN and BNN convolutional layers"""
    
    def __init__(self, input_channels=3, num_classes=10, bnn_conv_layers=None):
        super(HybridCNN, self).__init__()
        
        if bnn_conv_layers is None:
            bnn_conv_layers = [1, 2]  # Binarize middle conv layers
        
        self.bnn_conv_layers = bnn_conv_layers
        
        # Define architecture
        self.features = nn.ModuleList()
        
        # Layer 0: Regular conv (preserve input precision)
        if 0 in bnn_conv_layers:
            self.features.append(BinarizedConv2d(input_channels, 32, 3, padding=1))
        else:
            self.features.append(nn.Conv2d(input_channels, 32, 3, padding=1))
            self.features.append(nn.BatchNorm2d(32))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.MaxPool2d(2, 2))
        
        # Layer 1: Potentially binarized
        if 1 in bnn_conv_layers:
            self.features.append(BinarizedConv2d(32, 64, 3, padding=1))
        else:
            self.features.append(nn.Conv2d(32, 64, 3, padding=1))
            self.features.append(nn.BatchNorm2d(64))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.MaxPool2d(2, 2))
        
        # Layer 2: Potentially binarized
        if 2 in bnn_conv_layers:
            self.features.append(BinarizedConv2d(64, 128, 3, padding=1))
        else:
            self.features.append(nn.Conv2d(64, 128, 3, padding=1))
            self.features.append(nn.BatchNorm2d(128))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.AdaptiveAvgPool2d((4, 4)))
        
        # Classifier (keep full precision for final layer)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Example usage and training utilities
def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def estimate_model_size(model):
    """Estimate model size in MB"""
    total_size = 0
    for name, param in model.named_parameters():
        # Binarized weights effectively use 1 bit per weight
        if any(bnn_layer in name for bnn_layer in ['BinarizedLinear', 'BinarizedConv2d']):
            # For binarized layers, weights are effectively 1-bit
            size = param.numel() * 0.125 / (1024 * 1024)  # 1 bit per parameter
        else:
            # Regular layers use 32-bit floats
            size = param.numel() * 4 / (1024 * 1024)  # 4 bytes per parameter
        total_size += size
    return total_size

# Example models
if __name__ == "__main__":
    # Example 1: Hybrid fully connected network
    hybrid_fc = HybridANNBNN(
        input_size=784,  # 28x28 for MNIST
        hidden_sizes=[512, 256, 128],
        num_classes=10,
        bnn_layers=[1],  # Only middle layer is binarized
        dropout_rate=0.3
    )
    
    # Example 2: Hybrid CNN
    hybrid_cnn = HybridCNN(
        input_channels=3,  # RGB images
        num_classes=10,
        bnn_conv_layers=[1, 2]  # Middle conv layers are binarized
    )
    
    # Example 3: Hybrid Sequence Model (CNN + LSTM + BNN + ANN)
    hybrid_sequence = HybridSequenceModel(
        input_size=50,  # Feature dimension
        hidden_sizes=[256, 128],  # Classifier hidden layers
        lstm_hidden_size=128,
        num_classes=10,
        sequence_length=100,
        use_cnn=True,
        use_bnn_lstm=True,  # Use binarized LSTM
        bnn_layers=[0],  # First classifier layer is binarized
        num_lstm_layers=2
    )
    
    # Example 4: Time Series Prediction Model
    hybrid_timeseries = HybridTimeSeries(
        input_features=5,  # Number of input features per time step
        lstm_hidden_size=64,
        output_features=1,  # Predicting 1 feature
        sequence_length=50,
        forecast_horizon=10,  # Predict 10 steps ahead
        use_bnn_lstm=False,  # Use regular LSTM for better accuracy
        num_lstm_layers=2
    )
    
    # Example 6: Hybrid GAN Models
    
    # Image GAN
    image_generator = HybridGenerator(
        latent_dim=100,
        output_shape=(3, 64, 64),  # RGB 64x64 images
        generator_type='conv',
        bnn_layers=[0],  # Binarize first conv layer
        use_lstm=False,
        output_activation='tanh'
    )
    
    image_discriminator = HybridDiscriminator(
        input_shape=(3, 64, 64),
        discriminator_type='conv',
        bnn_layers=[1, 2],  # Binarize middle conv layers
        use_spectral_norm=True
    )
    
    # Sequence GAN for time series generation
    sequence_generator = HybridGenerator(
        latent_dim=64,
        output_shape=(50, 5),  # 50 time steps, 5 features
        generator_type='sequence',
        bnn_layers=['lstm', 0],  # Binarize LSTM and first output layer
        use_lstm=True,
        output_activation='tanh'
    )
    
    sequence_discriminator = HybridDiscriminator(
        input_shape=(50, 5),
        discriminator_type='sequence',
        bnn_layers=['lstm', 0],  # Binarize LSTM and first classifier layer
        lstm_hidden_size=64
    )
    
    # FC GAN for tabular data
    tabular_generator = HybridGenerator(
        latent_dim=32,
        output_shape=100,  # 100 features
        generator_type='fc',
        hidden_sizes=[128, 256, 512],
        bnn_layers=[0, 1],  # Binarize first two layers
        use_lstm=True,
        output_activation='sigmoid'
    )
    
    tabular_discriminator = HybridDiscriminator(
        input_shape=100,
        discriminator_type='fc',
        hidden_sizes=[512, 256, 128],
        bnn_layers=[1],  # Binarize middle layer
        use_lstm=False,
        use_spectral_norm=True
    )
    
    # Example 5: Pure Binarized LSTM
    bnn_lstm = BinarizedLSTM(
        input_size=20,
        hidden_size=64,
        num_layers=2,
        batch_first=True,
        dropout=0.2,
        bidirectional=True
    )
    
    # Test GAN models with dummy data
    dummy_noise = torch.randn(16, 100)  # Batch of 16, latent dim 100
    dummy_images = torch.randn(16, 3, 64, 64)  # Batch of 16, RGB 64x64
    dummy_sequences = torch.randn(16, 50, 5)  # Batch of 16, 50 steps, 5 features
    dummy_tabular = torch.randn(16, 100)  # Batch of 16, 100 features
    dummy_noise_small = torch.randn(16, 64)  # Smaller latent for sequence
    dummy_noise_tiny = torch.randn(16, 32)  # Tiny latent for tabular
    
    print("=== HYBRID ANN-BNN-LSTM-GAN NEURAL NETWORKS ===\n")
    
    # Test original models with updated dummy data
    dummy_input_fc = torch.randn(32, 784)
    dummy_input_cnn = torch.randn(32, 3, 32, 32)
    dummy_input_seq = torch.randn(32, 100, 50)
    dummy_input_ts = torch.randn(32, 50, 5)
    dummy_input_lstm = torch.randn(32, 25, 20)
    
    print("1. Hybrid FC Network:")
    print(f"   Output shape: {hybrid_fc(dummy_input_fc).shape}")
    total, trainable = count_parameters(hybrid_fc)
    print(f"   Parameters: {total:,} total, {trainable:,} trainable")
    print(f"   Estimated size: {estimate_model_size(hybrid_fc):.2f} MB")
    
    print("\n2. Hybrid CNN:")
    print(f"   Output shape: {hybrid_cnn(dummy_input_cnn).shape}")
    total, trainable = count_parameters(hybrid_cnn)
    print(f"   Parameters: {total:,} total, {trainable:,} trainable")
    print(f"   Estimated size: {estimate_model_size(hybrid_cnn):.2f} MB")
    
    print("\n3. Hybrid Sequence Model (CNN + LSTM + BNN):")
    print(f"   Output shape: {hybrid_sequence(dummy_input_seq).shape}")
    total, trainable = count_parameters(hybrid_sequence)
    print(f"   Parameters: {total:,} total, {trainable:,} trainable")
    print(f"   Estimated size: {estimate_model_size(hybrid_sequence):.2f} MB")
    
    print("\n4. Hybrid Time Series Model:")
    print(f"   Output shape: {hybrid_timeseries(dummy_input_ts).shape}")
    total, trainable = count_parameters(hybrid_timeseries)
    print(f"   Parameters: {total:,} total, {trainable:,} trainable")
    print(f"   Estimated size: {estimate_model_size(hybrid_timeseries):.2f} MB")
    
    print("\n5. Binarized LSTM:")
    lstm_output, (h, c) = bnn_lstm(dummy_input_lstm)
    print(f"   Output shape: {lstm_output.shape}")
    print(f"   Hidden state shape: {h.shape}")
    print(f"   Cell state shape: {c.shape}")
    total, trainable = count_parameters(bnn_lstm)
    print(f"   Parameters: {total:,} total, {trainable:,} trainable")
    print(f"   Estimated size: {estimate_model_size(bnn_lstm):.2f} MB")
    
    print("\n=== HYBRID ARCHITECTURE BENEFITS ===")
    print("✓ ANN Layers: Full precision for critical computations")
    print("✓ BNN Layers: 32x memory reduction, faster inference")
    print("✓ LSTM Layers: Sequential pattern learning and memory")
    print("✓ CNN Layers: Local feature extraction")
    print("✓ Attention: Focus on important sequence elements")
    print("✓ Flexible: Configure which layers to binarize")
    
    print("\n=== USE CASES ===")
    print("• Natural Language Processing (text classification, sentiment)")
    print("• Time Series Forecasting (stock prices, sensor data)")
    print("• Video Analysis (action recognition, object tracking)")
    print("• Audio Processing (speech recognition, music classification)")
    print("• Sensor Data Analysis (IoT, medical signals)")
    print("• Financial Modeling (algorithmic trading, risk assessment)")
    
    print("\n=== MEMORY EFFICIENCY COMPARISON ===")
    
    # Create comparable models for comparison
    full_precision_lstm = nn.LSTM(20, 64, 2, batch_first=True, bidirectional=True)
    total_fp, _ = count_parameters(full_precision_lstm)
    total_bnn, _ = count_parameters(bnn_lstm)
    
    print(f"Full Precision LSTM: {estimate_model_size(full_precision_lstm):.2f} MB")
    print(f"Binarized LSTM: {estimate_model_size(bnn_lstm):.2f} MB")
    print(f"Memory Reduction: {estimate_model_size(full_precision_lstm) / estimate_model_size(bnn_lstm):.1f}x smaller")
