echo "Clone ..."
git clone https://github.com/ductri/diacritics_restoration_contest.git
cd diacritics_restoration_contest

echo "Downloading data"
mkdir -p  source/main/data_for_train/output
wget http://213.246.38.101:2609/source/main/data_for_train/output/my_train.csv \
    -O source/main/data_for_train/output/my_train.csv
wget http://213.246.38.101:2609/source/main/data_for_train/output/my_eval.csv \
    -O source/main/data_for_train/output/my_eval.csv

echo "Downloading vocab"
mkdir -p  source/main/vocab/output
wget http://213.246.38.101:2609/source/main/vocab/output/voc_tgt_1.0 \
    -O source/main/vocab/output/voc_tgt_1.0
wget http://213.246.38.101:2609/source/main/vocab/output/voc_src_1.0 \
    -O source/main/vocab/output/voc_src_1.0

echo "Downloading checkpoint"
mkdir -p source/main/train/output/saved_models/Model/2.1/
wget http://213.246.38.101:2609/source/main/train/output/saved_models/Model/2.1/60000.pt \
    -O source/main/train/output/saved_models/Model/2.1/60000.pt

ln -s "`pwd`/source/" /source
