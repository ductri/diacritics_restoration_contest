git clone https://github.com/ductri/diacritics_restoration_contest.git

cd diacritics_restoration_contest

mkdir -p  diacritics_restoration_contest/source/main/data_for_train/output
wget http://213.246.38.101:2609/source/main/data_for_train/output/my_train.csv \
    -O diacritics_restoration_contest/source/main/data_for_train/output/my_train.csv
wget http://213.246.38.101:2609/source/main/data_for_train/output/my_eval.csv \
    -O diacritics_restoration_contest/source/main/data_for_train/output/my_eval.csv

mkdir -p  diacritics_restoration_contest/source/main/vocab/output
wget http://213.246.38.101:2609/source/main/vocab/output/voc_tgt_1.0 \
    -O diacritics_restoration_contest/source/main/vocab/output/voc_tgt_1.0
wget http://213.246.38.101:2609/source/main/vocab/output/voc_src_1.0 \
    -O diacritics_restoration_contest/source/main/vocab/output/voc_src_1.0
