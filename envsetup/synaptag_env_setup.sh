module purge
module load python-waterboa/2024.06

eval "$(conda shell.bash hook)"
conda create -n synaptag python=3.11
conda activate synaptag
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install timm requests
pip install pandas numpy matplotlib seaborn scikit-learn transformers