LIBS += -L${CUDA_PATH}/lib -L${CUDA_PATH}/lib64 -L${CUDA_PATH}/lib64/nvidia
LIBS += -lcudart -lcurand -licudata -lcudart_static -lcudadevrt

INCLUDEPATH += \
  ${CUDA_PATH}/include \
  ${CUDA_PATH}/include/cuda 

CUDA_INC += $$join(INCLUDEPATH, ' -I', '-I', ' ')

NVCCFLAGS += -ccbin ${HOST_COMPILER} -pg -g -lineinfo --std=c++14 -O3
NVCCFLAGS += -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}
NVCCFLAGS += -Xcompiler -fno-strict-aliasing -Xcompiler -fPIC 
NVCCFLAGS += --compiler-options -DNUM_ELEMENTS=10
NVCCFLAGS += -Xptxas -O3 --use_fast_math --restrict --expt-relaxed-constexpr --expt-extended-lambda
NVCCFLAGS += $$join(DEFINES, ' -D', '-D', ' ')
#NVCCFLAGS += -v
#NVCCFLAGS += -G

# Suppress warnings about __host__ and __device__ macros being used on defaulted ctors/dtors
NVCCFLAGS += -Xcudafe --display_error_number -Xcudafe --diag_suppress=2906 

NVCCBIN = ${CUDA_PATH}/bin/nvcc

CUDA_COMPILE_BASE = $${NVCCBIN} $${NVCCFLAGS} $${CUDA_INC} ${QMAKE_FILE_NAME}
CUDA_COMPILE = $${CUDA_COMPILE_BASE} -o ${QMAKE_FILE_OUT} $${LIBS}

# Compile cuda (device) code into object files
cuda.input = CUDA_SOURCES
cuda.output = $${CUDA_OBJECTS_DIR}/${QMAKE_FILE_BASE}.o
cuda.commands += $${CUDA_COMPILE} -dc  
cuda.CONFIG = no_link 
cuda.variable_out = CUDA_OBJ 
cuda.variable_out += OBJECTS
cuda.clean = $${CUDA_OBJECTS_DIR}/*.o
QMAKE_EXTRA_COMPILERS += cuda

# Link cuda object files into one object file with symbols that GCC can recognise
cudalink.input = CUDA_OBJ
cudalink.output = $${OBJECTS_DIR}/cuda_link.o
cudalink.commands = $${CUDA_COMPILE} -dlink
cudalink.CONFIG = combine
cudalink.dependency_type = TYPE_C
cudalink.depend_command = $${CUDA_COMPILE_BASE} -M
QMAKE_EXTRA_COMPILERS += cudalink


