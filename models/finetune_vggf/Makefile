CAFFE=/home/vmalpani/visual_search/caffe/build/tools/caffe
GPU=2

train:
	$(CAFFE) train -solver=quick_solver.prototxt -gpu $(GPU) -weights=VGG_CNN_F.caffemodel > train.log 2>&1 &
	tail -f train.log
