include($${PWD}/../../common.pri)
include($${PWD}/../test_common.pri)

TEMPLATE = app
TARGET = flo_device_tests

OBJECTS_DIR = obj
CUDA_OBJECTS_DIR = cudaobj

SOURCES += $$files(src/*.cpp, true)
CUDA_SOURCES += $$files(src/*.cu, true) 

INCLUDEPATH += $$PWD/include 

include($${PWD}/../../cuda_compiler.pri)
