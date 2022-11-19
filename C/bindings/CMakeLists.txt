set(target_name cpp_mpi)
add_library(${target_name} INTERFACE mpi.h)

set(target_name cpp_mpi_bench_template)
add_library(${target_name} INTERFACE bench_template.h)
target_link_libraries(${target_name} INTERFACE cpp_mpi)