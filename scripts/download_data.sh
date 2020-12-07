source activate scan-and-tell
pip install gdown
cd ../data/scannet
rm -rf scans
gdown https://drive.google.com/uc?id=1C69dzu0a3tt7IY1mU9GpIkXO5JHr-gHc https://drive.google.com/file/d/1e5hn7Pzg22jsRrd8JIhMB2O98_J7Xgby/view?usp=sharing
tar -xf scans_small.tar.gz
rm scans_small.tar.gz
mv scans_small scans
rm meta_data/scannetv2_val.txt
rm meta_data/scannetv2_test.txt
rm meta_data/scannetv2_train.txt
rm meta_data/scannetv2.txt
mv meta_data/scannetv2_val_small.txt meta_data/scannetv2_val.txt
mv meta_data/scannetv2_train_small.txt meta_data/scannetv2_train.txt
mv meta_data/scannetv2_test_small.txt meta_data/scannetv2_test.txt
mv meta_data/scannetv2_small.txt meta_data/scannetv2.txt