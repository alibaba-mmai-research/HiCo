cd ..
rm -rf code.tar.gz;
tar -zc --exclude='*.git' -f code.tar.gz pytorch-video-understanding-dev requirements.txt