include(CompileOptions)
set(target_name test_p2p)

add_executable(${target_name} p2p.cpp)

set_compile_options(${target_name})
link_mpi(${target_name})