INCLUDEPATH += \
  /usr/include/eigen3 \
  /usr/local/include \
  /usr/local/include/libigl/include \
  # 3.3.7
  $${PWD}/dep/eigen \
  $${PWD}/dep/libigl/include \
  # This comes last so the home directory takes higher precidence
  /public/devel/2018/include 

INCLUDEPATH += $${PWD}/flo/include

FLO_COMPILE_DEVICE_CODE=0

#DEFINES += FLO_USE_DOUBLE_PRECISION
DEFINES += THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA

QT -= opengl core gui
CONFIG += console c++17
CONFIG -= app_bundle

# Standard flags
QMAKE_CXXFLAGS += -std=c++17 -g -fdiagnostics-color
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


