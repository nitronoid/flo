include($${PWD}/../../../common.pri)

TEMPLATE = app
TARGET = willmore_flow.out

OBJECTS_DIR = obj
CUDA_OBJECTS_DIR = cudaobj

CUDA_SOURCES += $$files(src/*.cu, true) 

INCLUDEPATH += $$PWD/include 

FLO_LIB_PATH = $${PWD}/../../../flo/lib
LIBS += -L$${FLO_LIB_PATH} -lflo 
QMAKE_RPATHDIR += $${FLO_LIB_PATH}

include($${PWD}/../../../cuda_compiler.pri)

