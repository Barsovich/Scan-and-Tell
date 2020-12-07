if [ "$1" != "--model" ]
then 
    echo "Please specify the model using --model <model> or -m <model>."
    exit 1
else
    model="$2"
    echo "Building the project for the model ${model}"
fi

wget https://raw.githubusercontent.com/Barsovich/Scan-and-Tell/main/scripts/setup.sh
wget https://raw.githubusercontent.com/Barsovich/Scan-and-Tell/main/scripts/common.sh
chmod +x setup.sh
./setup.sh

cd Scan-and-Tell
source activate scan-and-tell
chmod +x scripts/download_data.sh
cd scripts
./download_data.sh
cd ../data/scannet
python batch_load_scannet_data.py --model $model
