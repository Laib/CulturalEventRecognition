CAFFE=/home/vmalpani/visual_search/caffe/build/tools/caffe
GPU=0

train:
	$(CAFFE) train -solver=quick_solver.prototxt -weights=nin_imagenet.caffemodel -gpu $(GPU) > train.log 2>&1 &
	tail -f train.log

