CAFFE=/home/vmalpani/visual_search/caffe/build/tools/caffe
GPU=1

train:
	$(CAFFE) train -solver=quick_solver.prototxt -gpu $(GPU) -weights=siat_scene_vgg_11.caffemodel > train.log 2>&1 &
	tail -f train.log
