include(CompileOptions)
set(target_name p2p)

add_executable(${target_name} p2p.cpp)

set_compile_options(${target_name})
link_mpi(${target_name})
link_bench_template(${target_name})

set(target_name bcast)

add_executable(${target_name} bcast.cpp)

set_compile_options(${target_name})
link_mpi(${target_name})
link_bench_template(${target_name})

set(target_name reduce)
add_executable(${target_name} reduce.cpp)

set_compile_options(${target_name})
link_mpi(${target_name})
link_bench_template(${target_name})

set(target_name allreduce)
add_executable(${target_name} allreduce.cpp)

set_compile_options(${target_name})
link_mpi(${target_name})
link_bench_template(${target_name})

set(target_name gather)
add_executable(${target_name} gather.cpp)

set_compile_options(${target_name})
link_mpi(${target_name})
link_bench_template(${target_name})

set(target_name allgather)
add_executable(${target_name} allgather.cpp)

set_compile_options(${target_name})
link_mpi(${target_name})
link_bench_template(${target_name})