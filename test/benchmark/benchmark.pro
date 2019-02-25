include($${PWD}/../../common.pri)
include($${PWD}/../test_common.pri)

TEMPLATE = app
TARGET = flo_benchmarks.out

OBJECTS_DIR = obj
CUDA_OBJECTS_DIR = cudaobj

HEADERS += $$files(include/*.h, true)
CUDA_SOURCES += $$files(src/*.cu, true) 
CUDA_SOURCES += $$files(src/*.cpp, true) 

INCLUDEPATH += $$PWD/include 

LIBS += -lbenchmark

include($${PWD}/../../cuda_compiler.pri)

