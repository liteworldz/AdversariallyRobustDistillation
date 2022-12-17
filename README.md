# Adversarially Robust Distillation
Based on the Paper Adversarially Robust Distillation (https://arxiv.org/pdf/1905.09747.pdf)
The code was also adopted from the original paper source code (https://github.com/goldblum/AdversariallyRobustDistillation), with minor modifications to serve this repo.
- Here we introduce a new method, that will enhance the knowledge distillation, witout the need for a pre-trained Teacher network



# env setup
```
conda create -n advkd python=3.9
conda activate advkd
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia 
pip install torchattacks argparse scipy tqdm
pip install labml-helpers
```

# Training Flags (loss ['paper': original paper loss, 'method': new method loss without a teacher]), you can also set other parameters like alpha, temperature, output folder and so on.
```
python main.py --loss paper --alpha 0.9
```

# Testing your network with PGD or AUTO attacks
```
python test.py --filename 'you_file_name' --attack PGD
```