######################################################
# Upgrade GPU drivers:
######################################################
sudo apt install nvidia-driver-550
sudo apt install nvidia-utils-550
sudo reboot

# Check GPU driver version:
nvidia-smi
cat /proc/driver/nvidia/version

######################################################
# Install Conda:
######################################################
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda init

######################################################
# Link to shared projects on the Mac:
######################################################
sudo apt install cifs-utils
sudo mkdir -p /mnt/nelson-projects
sudo mount -t cifs //192.168.86.38/Projects /mnt/nelson_projects -o username=nelsonnunes,vers=3.0,uid=$(id -u),gid=$(id -g),file_mode=0777,dir_mode=0777
cd ~/projects
ln -s /mnt/nelson_projects/survey_tools survey_tools
ln -s /mnt/nelson-projects/TIPTOP TIPTOP
ln -s /mnt/nelson-projects/P3 P3
ln -s /mnt/nelson-projects/MASTSEL MASTSEL

# To unmount the share:
# sudo umount /mnt/nelson_projects

######################################################
# Share output to Mac:
######################################################
mkdir ~/projects/TIPTIOP_Output
sudo apt install samba -y
sudo smbpasswd -a keita
sudo nano /etc/samba/smb.conf
# Add the following lines to the end of the file:
[TIPTOP_OUTPUT]
   path = /home/keita/projects/TIPTOP_Output
   available = yes
   valid users = keita
   read only = no
   browsable = yes
   public = yes
   writable = yes
   create mask = 0777
   directory mask = 0777
   force user = keita
sudo systemctl restart smbd
# Finder > Go > Connect to Server: smb://192.168.86.41/TIPTOP_Output

######################################################
# Create Conda environment
######################################################
conda create --name tiptop python=3.12
conda activate tiptop

######################################################
# Setup TIPTOP, P3, MASTSEL
######################################################
conda activate tiptop
conda install -c conda-forge cupy
# important that the following are done in the reverse
# order of dependency to ensure local versions are used
cd ~/projects/survey_tools
~/miniconda3/envs/tiptop/bin/pip3 install -e .
cd ..
cd ~/projects/TIPTOP
~/miniconda3/envs/tiptop/bin/pip3 install -e .
cd ..
cd ~/projects/P3
~/miniconda3/envs/tiptop/bin/pip3 install -e .
cd ..
cd ~/projects/MASTSEL
~/miniconda3/envs/tiptop/bin/pip3 install -e .
cd ..
