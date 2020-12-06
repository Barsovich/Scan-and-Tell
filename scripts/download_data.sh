pip install gdown
cd ../data/scannet
gdown https://drive.google.com/uc?id=1zi6TBiVTqXaMJv28eDmKUtviA-mipZUb
tar -xf scans_small.tar.gz
rm scans_small.tar.gz
mv scans_small scans
rm meta_data/scannetv2_val.txt
rm meta_data/scannetv2_test.txt
rm meta_data/scannetv2_train.txt
mv meta_data/scannetv2_val_small.txt meta_data/scannetv2_val.txt
mv meta_data/scannetv2_train_small.txt meta_data/scannetv2_train.txt
mv meta_data/scannetv2_test_small.txt meta_data/scannetv2_test.txt