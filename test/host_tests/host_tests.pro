include($${PWD}/../../common.pri)
include($${PWD}/../test_common.pri)

TEMPLATE = app
TARGET = flo_host_tests.out

OBJECTS_DIR = obj

HEADERS += $$files(include/*.hpp, true)
SOURCES += $$files(src/*.cpp, true)

INCLUDEPATH += $$PWD/include 


