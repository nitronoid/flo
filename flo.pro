include($${PWD}/common.pri)

TEMPLATE = app
TARGET = flo_demo

OBJECTS_DIR = obj

HEADERS += $$files(include/*.h, true)
HEADERS += $$files(include/*.inl, true)
SOURCES += $$files(src/*.cpp, true)

INCLUDEPATH += $$PWD/include 

#LIBS += -L../flow/lib -lflow 

#QMAKE_RPATHDIR += ../flow/lib

