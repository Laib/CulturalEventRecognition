CAFFE=/home/vmalpani/visual_search/caffe/build/tools/caffe
GPU=0

train:
	$(CAFFE) train -solver=quick_solver.prototxt -gpu $(GPU) -weights=imagenet_googlenet.caffemodel > train.log 2>&1 &
	tail -f train.log

