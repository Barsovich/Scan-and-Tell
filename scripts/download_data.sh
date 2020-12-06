pip install gdown
cd ../data/scannet
gdown https://drive.google.com/uc?id=11I5dQQSuEqEUz1f76jcZ9hR9RcB6aNjI
tar -xf scans_small.tar.gz
rm scans_small.tar.gz
mv scans_small scans