CFLAGES = `pkg-config --cflags opencv`
MYFLAGES+= -I../include \
		  -I../build/include \
		  -I/usr/local/cuda/include \
		  -I/usr/include/hdf5/serial \
		  -I./dist/ \
		  -I../.build_release/lib
LIBS = `pkg-config --libs opencv`
MYLIBS+= -L../build/lib/ \
		 -lcaffe -lboost_system -lboost_thread -lglog -lpthread \
		 -lopencv_imgcodecs \
		 -DCAFFE_VERSION=1.0.0-rc3 -DNDEBUG -O2 -DUSE_CUDNN -DUSE_OPENCV \
	-DUSE_LEVELDB -DUSE_LMDB -isystem /usr/include/python2.7 -isystem \
	/usr/lib/python2.7/dist-packages/numpy/core/include -isystem \
	/usr/local/include -isystem /usr/include/hdf5/serial -isystem \
	../.build_release/src -isystem ./src -isystem ./include -isystem \
	/usr/local/cuda/include -Wall -Wno-sign-compare -lcaffe \
	-L/usr/lib -L/usr/local/lib -L/usr/lib \
	-L/usr/lib/aarch64-linux-gnu/hdf5/serial/ \
	-L/usr/local/cuda/lib64 -L/usr/local/cuda/lib -L../.build_release/lib \
	-lcudart -lcublas -lcurand -lglog -lgflags -lprotobuf -lboost_system \
	-lboost_filesystem -lboost_regex -lm -lhdf5_hl -lhdf5 -lleveldb -lsnappy \
	-llmdb -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs \
	-lopencv_videoio -lboost_thread -lstdc++ -lcudnn -lcblas -latlas 
all: json.o track
FLAGES_LIBS = `pkg-config opencv --cflags --libs`
json.o: json.cpp
	sudo g++ -c json.cpp
track: track.cpp
	sudo g++ -o $@ $< $(JSONFLAGS) $(CFLAGES) $(MYFLAGES) json.o $(LIBS) $(MYLIBS)
clean: 
	sudo rm track json.o

