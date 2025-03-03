### PUQ: Phase-wise Uncertainty Guided QMRI Reconstruction

This repository provides the code for the Phase-wise Uncertainty Guided method for quantitative magnetic resonance imaging (qMRI) reconstruction. PUQ leverages phase-wise uncertainty to guide the parameter fitting process, enhancing reconstruction accuracy. The method incorporates Monte Carlo Dropout (MC Dropout) for uncertainty measurement and employs a two-stage reconstruction and fitting framework.

**The corresponding paper has been submitted to MICCAI 2025.**

---

#### Requirements

- Python 3.8
- PyTorch >= 1.12.0
- NumPy >= 1.24.4
- SciPy >= 1.10.1

---

#### Usage

##### Data Preparation
Ensure that your input data is in the correct format. You can follow the structure in **dataset.py**. We provide one test case dataset in our paper for model evaluation.

##### Training
PUG training follows a two-step process:

1. **Train the reconstruction model:**
   ```bash
   python train_twostep.py -c ./config/PUQ_recon.yaml
   ```

2. **Extract and arrange outputs as pixel data:**
   ```bash
   python pixels_gen.py -c ./config/PUQ_recon.yaml -t model_uq -d t1 -ut evar -s 100
   ```

3. **Train the parameter fitting model:**
   ```bash
   python train_regress.py -c ./config/PUQ_fit.yaml
   ```

All necessary training commands are included in `commands.sh`.

##### Testing
The testing procedure is outlined in **test.ipynb**. We provide trained models, which can be downloaded in https://1drv.ms/f/c/3a8b097d5fc24529/ErJv1gS-knVAmulEwYvOhasB8RuStx2gHztZ6QGyql64Fg?e=0yh2ub and placed in the `./export/` directory for evaluation.

---

#### License
This code is released under the MIT License. See the LICENSE file for details.
