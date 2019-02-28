include($${PWD}/../../../common.pri)
include($${PWD}/../../test_common.pri)

TEMPLATE = app
TARGET = flo_host_benchmarks.out

OBJECTS_DIR = obj
CUDA_OBJECTS_DIR = cudaobj

HEADERS += $$files(include/*.h, true)
SOURCES += $$files(src/*.cpp, true) 

INCLUDEPATH += $$PWD/include 

LIBS += -lbenchmark

