include($${PWD}/../common.pri)

TEMPLATE = lib
TARGET = flo
DESTDIR = lib

OBJECTS_DIR = obj

HEADERS += $$files(include/flo/*.hpp, true)
HEADERS += $$files(include/flo/*.inl, true)
SOURCES += $$files(src/*.cpp, true)

