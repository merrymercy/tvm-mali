sudo /etc/init.d/lightdm stop
sudo echo performance > /sys/class/misc/mali0/device/devfreq/ff9a0000.gpu/governor

export PYTHONPATH=$(pwd)/nnvm/python:$(pwd)/nnvm/tvm/python:$(pwd)/nnvm/tvm/topi/python:$(pwd)/incubator-mxnet/python

python mxnet_test.py --model all
python mali_imagenet_bench.py --model all
LD_LIBRARY_PATH=ComputeLibrary/build ./acl_test all

