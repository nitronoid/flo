
INCLUDEPATH += $${PWD}/include

FLO_LIB_PATH = $${PWD}/../flo/lib
LIBS += -L$${FLO_LIB_PATH} -lflo 
QMAKE_RPATHDIR += $${FLO_LIB_PATH}

LIBS += -lgtest -lgmock 

DEFINES += MESH_PATH='$${PWD}/models'
DEFINES += MATRIX_PATH='$${PWD}/matrices'

