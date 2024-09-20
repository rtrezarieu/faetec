if [ "$1" = "is2re" ]; then
    echo "Downloading IS2RE 10k"
    cd data
    mkdir -p is2re/train/
    cd is2re/train
    pip install --upgrade --no-cache-dir gdown
    if [ ! -f is2re_10k.lmdb ]; then
        gdown https://drive.google.com/uc?id=19b7kOXBiHkhr_gzo-0iITIDN012puN-1 -O is2re_10k.lmdb
    fi
    cd ../
    mkdir -p val/
    cd val
    if [ ! -f val_id.lmdb ]; then
        gdown https://drive.google.com/uc?id=1ALTdSZuoc1KRmuf5KEDWFz6yciqv2zJq -O val_id.lmdb
    fi
    if [ ! -f val_ood_both.lmdb ]; then
        gdown https://drive.google.com/uc?id=1r4yL1fRNhdFcOf7r7EwhbvOHBPhfM88F -O val_ood_both.lmdb
    fi
elif [ "$1" = "dense" ]; then
    echo "Downloading OC20 Dense"
    cd data
    mkdir -p dense
    cd dense
else
    echo "Usage: download_data.sh [is2re|oc20]"
fi
