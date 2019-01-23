DEPPATH = $${PWD}/dep
DEPS = $$system(ls $${DEPPATH})
!isEmpty(DEPS) {
  for(d, DEPS) {
    INCLUDEPATH += $${DEPPATH}/$${d}
    INCLUDEPATH += $${DEPPATH}/$${d}/include
  }
}

INCLUDEPATH += \
  /usr/include/eigen3 \
  /usr/local/include \
  /usr/local/include/libigl/include \
  /usr/include/suitesparse \
  /home/s4902673/SuiteSparse/include \
  /public/devel/2018/include \
  /public/devel/2018/include/eigen3 

INCLUDEPATH += $${PWD}/flo/include

#Linker search paths
LIBS += -L/home/s4902673/SuiteSparse/lib
# Linker libraries
LIBS += -pthread -lcholmod

QT -= opengl core gui
CONFIG += console c++14
CONFIG -= app_bundle

# Standard flags
QMAKE_CXXFLAGS += -std=c++14 -g -fdiagnostics-color
# Optimisation flags
QMAKE_CXXFLAGS += -Ofast -march=native -frename-registers -funroll-loops -ffast-math -fassociative-math
# Enable openmp
QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS += -fopenmp
# Intrinsics flags
QMAKE_CXXFLAGS += -mfma -mavx2 -m64 -msse -msse2 -msse3
# Enable all warnings
QMAKE_CXXFLAGS += -Wall -Wextra -pedantic-errors
# Vectorization info
QMAKE_CXXFLAGS += -ftree-vectorize -ftree-vectorizer-verbose=5 


