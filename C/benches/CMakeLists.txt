include(CompileOptions)
set(target_name bench_p2p)

add_executable(${target_name} p2p.cpp bench_template.h)

set_compile_options(${target_name})
link_mpi(${target_name})

set(target_name bench_bcast)

add_executable(${target_name} bcast.cpp bench_template.h)

set_compile_options(${target_name})
link_mpi(${target_name})