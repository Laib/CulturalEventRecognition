CAFFE=/home/vmalpani/visual_search/caffe/build/tools/caffe
GPU=1

train:
	$(CAFFE) train -solver=quick_solver.prototxt -gpu $(GPU) -weights=8conv3fc_DSN.caffemodel > train.log 2>&1 &
	tail -f train.log
