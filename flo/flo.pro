include($${PWD}/../common.pri)

TEMPLATE = lib
TARGET = flo
DESTDIR = lib

OBJECTS_DIR = obj
CUDA_OBJECTS_DIR = cudaobj

HEADERS += $$files(include/flo/*(.hpp | cuh), true)

equals(FLO_COMPILE_DEVICE_CODE, 1) {
    CUDA_SOURCES += $$files(src/*.cu, true)
    include($${PWD}/../cuda_compiler.pri)
}


