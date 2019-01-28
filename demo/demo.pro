include($${PWD}/../common.pri)

TEMPLATE = app
TARGET = flo_demo.out

OBJECTS_DIR = obj
CUDA_OBJECTS_DIR = cudaobj

SOURCES += $$files(src/*.cpp, true)
CUDA_SOURCES += $$files(src/*.cu, true) 

INCLUDEPATH += $$PWD/include 


LIBS += -L../flo/lib -lflo 
QMAKE_RPATHDIR += ../flo/lib

include($${PWD}/../cuda_compiler.pri)

