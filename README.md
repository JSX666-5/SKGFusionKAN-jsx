# SKGFusionKAN

**Enhanced Feature Extraction for IoT Network Intrusion Detection Using GNNs and KAN**

SKGFusionKAN is a novel IoT intrusion detection approach that:
- **SKGraphSAGE**: Enhances GraphSAGE with a multi-scale selective kernel attention mechanism
- **Gated Fusion**: Dynamically integrates multi-scale node and edge features
- **KAN Classifier**: Leverages Kolmogorov-Arnold Networks for superior nonlinear modeling


##  Requirements

```bash
python >= 3.8
torch >= 2.0.0
dgl >= 1.1.0
scikit-learn >= 1.2.0
pandas >= 2.0.0
numpy >= 1.24.0
category-encoders >= 2.6.0
imbalanced-learn >= 0.10.0
networkx >= 3.1
tqdm >= 4.65.0
```

**Installation**:
```bash
# Install PyTorch (with CUDA support recommended)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install DGL (GPU version for CUDA 11.8)
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html

# Install other dependencies
pip install -r requirements.txt
```

---

##  Datasets

Supported datasets:
- **NF-BoT-IoT** / **NF-BoT-IoT-v2**
- **NF-ToN-IoT** / **NF-ToN-IoT-v2**
- **NF-CSE-CIC-IDS2018-v2**

### Data Preparation

1. Download the datasets from their official sources.

2. Create necessary directories:
```bash
mkdir -p data model reports binary_model binary_reports predictions binary_predictions
```

3. Place the raw dataset files in the `data/` directory with the following structure:
```
data/
├── NF-BoT-IoT.csv
├── NF-BoT-IoT-v2.csv
├── NF-ToN-IoT.csv
├── NF-ToN-IoT-v2.csv
└── NF-CSE-CIC-IDS2018-v2.csv
```

4. Run the preprocessing script to convert data to graph format:
```bash
cd experiments
python data_preprocess.py
```

---

##  Running Experiments

### Main Experiment (SKGFusionKAN)

**Step 1**: Edit the experiment parameters in `experiment_binary.py`:
```python
# Line 289: Set dataset
dataset = 'NF-BoT-IoT'  # Options: 'NF-BoT-IoT', 'NF-BoT-IoT-v2', 'NF-ToN-IoT', 'NF-ToN-IoT-v2', 'NF-CSE-CIC-IDS2018-v2'

# Line 283-286: Set model components
attention_name = 'SKAttention'  # Options: 'SE', 'SKAttention', 'CPCA', 'SCSA', 'CBAM', None
fusion_name = 'GatedFusion'     # Options: 'DFF', 'SDM', 'MSAF', 'SqueezeAndExciteFusionAdd', 'TIF', 'WCMF', 'GatedFusion'
mlp_name = 'KAN'                # Options: 'MLP', 'KAN'
```

**Step 2**: Run the experiment:
```bash
cd experiments
python experiment_binary.py
```

### Baseline Methods

```bash
# GAT (Graph Attention Network)
python experiment_gat.py

# E-GraphSAGE
python experiment_egat.py

# EdgeGAT
python experiment_edgegat.py

# Anomal-E (Self-supervised)
python experiment_anomale.py

# Traditional ML models (Random Forest, XGBoost, etc.)
python experiment_ml.py
```

---

##  Project Structure

```
SKGFusionKAN/
├── module/                    # Core modules
│   ├── efficientKan.py        # Kolmogorov-Arnold Networks
│   ├── SKNet.py               # Selective Kernel Attention
│   ├── GatedFusion.py         # Gated Fusion Mechanism
│   ├── CBAM.py                # Convolutional Block Attention Module
│   ├── CPCA2d.py              # Channel-wise PCA Attention
│   ├── SCSA.py                # Spatial and Channel Squeeze & Excitation
│   ├── DFF2d.py               # Deep Feature Fusion
│   ├── SDM2d.py               # Squeeze-Dense Module
│   ├── MSAF2d.py              # Multi-Scale Attention Fusion
│   ├── TIF.py                 # Triple Interaction Fusion
│   ├── WCMF.py                # Weighted Channel-wise Multimodal Fusion
│   └── ...
├── experiments/               # Experiment scripts
│   ├── experiment_binary.py   # SKGFusionKAN main experiment
│   ├── experiment_gat.py      # GAT baseline
│   ├── experiment_egat.py     # E-GraphSAGE baseline
│   ├── experiment_edgegat.py  # EdgeGAT baseline
│   ├── experiment_anomale.py  # Anomal-E baseline
│   ├── experiment_ml.py       # Traditional ML baselines
│   └── data_preprocess.py     # Data preprocessing
├── data/                      # Raw and processed datasets
├── model/                     # Saved multiclass models
├── binary_model/              # Saved binary classification models
├── reports/                   # Multiclass classification reports (JSON)
├── binary_reports/            # Binary classification reports (JSON)
├── predictions/               # Multiclass predictions
├── binary_predictions/        # Binary predictions
├── requirements.txt           # Dependencies
└── README.md
```

---

##  Results

### Binary Classification (F1 Score)
| Dataset | SKGFusionKAN | GAT | E-GraphSAGE | Anomal-E |
|---------|--------------|-----|-------------|----------|
| NF-BoT-IoT | 98.12% | 96.65% | 97.15% | 98.29% |
| NF-ToN-IoT | 99.01% | 91.88% | 97.03% | 96.61% |
| NF-BoT-IoT-v2 | 99.96% | 98.38% | 99.95% | 99.95% |
| NF-ToN-IoT-v2 | 98.22% | 94.20% | 98.13% | 97.57% |

### Multiclass Classification (F1 Score)
| Dataset | SKGFusionKAN | GAT | E-GraphSAGE | Anomal-E |
|---------|--------------|-----|-------------|----------|
| NF-BoT-IoT | 83.97% | 82.10% | 80.62% | 82.82% |
| NF-ToN-IoT | 62.30% | 37.34% | 59.14% | 61.38% |
| NF-BoT-IoT-v2 | 98.58% | 87.01% | 98.47% | 90.36% |
| NF-ToN-IoT-v2 | 95.73% | 87.05% | 93.95% | 89.57% |

### Ablation Study Results
| Component Removed | NF-BoT-IoT F1 | NF-ToN-IoT F1 | NF-BoT-IoT-v2 F1 | NF-ToN-IoT-v2 F1 |
|-------------------|---------------|---------------|------------------|------------------|
| Full Model (Ours) | 83.97% | 62.30% | 98.58% | 95.73% |
| w/o SKAttention | 82.09% | 60.49% | 98.45% | 90.12% |
| w/o GatedFusion | 82.23% | 60.97% | 98.44% | 94.42% |
| w/o KAN | 82.49% | 61.36% | 98.45% | 94.45% |

---

##  License

This project is licensed under the MIT License.



