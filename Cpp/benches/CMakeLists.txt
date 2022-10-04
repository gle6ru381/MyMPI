include(CompileOptions)
set(target_name bench)

add_executable(${target_name} main.cpp)

set_compile_options(${target_name})
link_mpi(${target_name})
