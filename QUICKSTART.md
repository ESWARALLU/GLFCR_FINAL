# GLF-CR Enhanced Research Project

## Quick Start Guide

### 1. Environment Setup
```bash
# Activate environment (if using conda/venv)
# conda activate glf-cr-env

# Verify GPU
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2. Run Baseline Test
```bash
cd codes
python test_CR.py --model_name "baseline" --notes "Your notes here"
```

### 3. Check Results
- **CSV Summary:** `results/results_summary.csv`
- **Full History:** `results/results_history.json`
- **Individual Runs:** `results/result_*.json`

### 4. Start New Experiment
1. Copy `experiments/EXPERIMENT_TEMPLATE.md` → `experiments/exp_001_description.md`
2. Document your changes
3. Train/test model
4. Update research notes

---

## Folder Structure

```
GLF-CR/
├── codes/              # Source code
├── ckpt/               # Model checkpoints
├── data/               # Dataset
├── results/            # Test results & metrics
│   └── visualizations/ # Generated images
├── experiments/        # Experiment logs
├── logs/               # Training logs
├── notebooks/          # Jupyter notebooks for analysis
├── papers/             # Related papers & references
└── docs/               # Documentation
```

---

## Workflow

### For Each Phase:
1. **Plan** - Review phase objectives in `IMPLEMENTATION_PLAN.md`
2. **Implement** - Add code, test incrementally
3. **Document** - Fill experiment template
4. **Evaluate** - Run tests, compare results
5. **Iterate** - Adjust based on findings
6. **Update** - Note findings in `RESEARCH_NOTES.md`

### Best Practices:
- ✅ Run baseline before major changes
- ✅ Use meaningful model names
- ✅ Add descriptive notes to all runs
- ✅ Save checkpoints regularly
- ✅ Document hyperparameters
- ✅ Create visualizations for qualitative analysis
- ✅ Commit code changes with clear messages

---

## Useful Commands

### Testing
```bash
# Basic test
python test_CR.py

# With tracking
python test_CR.py --model_name "phase1_v1" --notes "8-head cross-attention"

# Specific checkpoint
python test_CR.py --checkpoint ../ckpt/phase1_best.pth
```

### Training
```bash
# See train_CR_net.py for training options
python train_CR_net.py --help
```

### Analysis
```bash
# Compare results in CSV
cat results/results_summary.csv

# View latest result
cat results/results_history.json | tail -n 50
```

---

## Performance Targets

| Phase | Component | Target PSNR | Target SSIM |
|-------|-----------|-------------|-------------|
| Baseline | GLF-CR | 23.25 | 0.8084 |
| Phase 1 | + Cross-Attn | 24.75-25.25 | 0.82-0.83 |
| Phase 2 | + Perceptual | 25.75-26.75 | 0.83-0.85 |
| Phase 3 | + Texture | 26.25-27.75 | 0.84-0.86 |
| Phase 4 | + Global-Local | 27.75-30.25 | 0.88-0.92 |

**Overall Goal:** +5-7 dB PSNR, +0.10-0.15 SSIM

---

## Resources

- **Implementation Plan:** `IMPLEMENTATION_PLAN.md`
- **Detailed Breakdown:** `DETAILED_PHASE_BREAKDOWN.md`
- **Enhancement Proposals:** `transformer_enhancement_proposal.md`
- **Migration Guide:** `MIGRATION_GUIDE.md`
- **Research Notes:** `RESEARCH_NOTES.md`

---

## Contact & Support

For issues or questions, refer to:
- Original GLF-CR paper
- PyTorch documentation
- CUDA troubleshooting guide

---

**Last Updated:** 2025-11-20
**Status:** Ready for Phase 1 Implementation
