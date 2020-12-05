wget https://raw.githubusercontent.com/Barsovich/Scan-and-Tell/main/scripts/setup.sh
wget https://raw.githubusercontent.com/Barsovich/Scan-and-Tell/main/scripts/common.sh
chmod +x setup.sh
./setup.sh

cd Scan-and-Tell
source activate scan-and-tell
python scripts/download_data.py
python data/scannet/batch_load_scannet_data.py --model $1
