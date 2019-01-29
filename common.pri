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
  ${HOME}/SuiteSparse/include \
  # 3.3.7
  ${HOME}/eigen3 \
  ${HOME}/libigl/include \
  # This comes last so the home directory takes higher precidence
  /public/devel/2018/include 

INCLUDEPATH += $${PWD}/flo/include

#Linker search paths
LIBS += -L/home/s4902673/SuiteSparse/lib
# Linker libraries
LIBS += -lcholmod

QT -= opengl core gui
CONFIG += console c++11
CONFIG -= app_bundle

# Standard flags
QMAKE_CXXFLAGS += -std=c++11 -g -fdiagnostics-color
# Optimisation flags
QMAKE_CXXFLAGS += -Ofast -march=native -frename-registers -funroll-loops -ffast-math -fassociative-math
# Enable openmp
QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS += -fopenmp
# Intrinsics flags
QMAKE_CXXFLAGS += -mfma -mavx2 -m64 -msse -msse2 -msse3
# Enable all warnings
QMAKE_CXXFLAGS += -Wall -Wextra -pedantic-errors -Wno-sign-compare
# Vectorization info
QMAKE_CXXFLAGS += -ftree-vectorize -ftree-vectorizer-verbose=5 


