# Reproduction Instructions: SISA Machine Unlearning

**Environment:**
* **OS:** macOS (Apple Silicon M2)
* **Python:** 3.11.x
* **PyTorch:** 2.x (MPS Backend)

**Setup:**
```bash
python3.11 -m venv venv

source venv/bin/activate

pip install torch torchvision torchaudio numpy pandas matplotlib scikit-learn psutil
```

**Run Command**
```bash
cd sisa_unlearning

python sisa_main.py
```