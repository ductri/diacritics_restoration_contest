#!/bin/bash


echo "Setup credentials ..."
mkdir /root/.ssh -p
cp '/content/gdrive/My Drive/my_google_colab/ssh/id_rsa' /root/.ssh/
cp '/content/gdrive/My Drive/my_google_colab/ssh/known_hosts' /root/.ssh/
cp '/content/gdrive/My Drive/my_google_colab/ssh/authorized_keys' /root/.ssh/

echo "Install packages ..."
apt-get install -q vim
apt-get install -q tmux
#dpkg -i -R '/content/gdrive/My Drive/my_google_colab/packages/'

echo "Install libraries ..."
pip install -q --upgrade '/content/gdrive/My Drive/my_google_colab/libs/naruto_skills-1.1-py3-none-any.whl'
pip install -q tensorflow==1.13.1
pip install -q tensorboard
pip install -q tensorboardX
pip install -q pygtrie
pip install -q tldextract
python -m nltk.downloader 'punkt'

echo "Mapping to /source ..."
ln -s -f '/content/gdrive/My Drive/my_google_colab/diacritics_restoration_contest/source' /source

echo "Starting  tensorboard ..."
tensorboard --logdir /source/main/train/output/logging --host 0.0.0.0 --port 6006 &

rm /usr/local/bin/python
ln -s /usr/bin/python3 /usr/local/bin/python