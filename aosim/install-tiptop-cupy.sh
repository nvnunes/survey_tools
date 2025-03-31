# Upgrade GPU drivers:
sudo apt install nvidia-driver-550
sudo apt install nvidia-utils-550
sudo reboot

# Check GPU driver version:
nvidia-smi
cat /proc/driver/nvidia/version

# Install Conda:
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda init

# Install TipTop
conda create --name tiptop python=3.12
conda activate tiptop
conda install -c conda-forge cupy
cd ~/projects
git clone https://github.com/nvnunes/TIPTOP.git
cd TIPTOP
~/miniconda3/envs/tiptop/bin/pip3 install -e .

# To run TipTop test:
conda activate tiptop
cd ~/projects/TIPTOP
python3 tests/test_all.py
