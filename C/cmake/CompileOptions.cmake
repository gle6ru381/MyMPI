function(set_compile_options target_name)
    string(TOLOWER ${CMAKE_BUILD_TYPE} build_type)
    if(${build_type} STREQUAL release)
        target_compile_options(${target_name} PRIVATE -O2)
    else()
        target_compile_options(${target_name} PRIVATE -g2 -Wall -Wextra -Werror -pedantic)
    endif()

    add_link_options(-static-libgcc -static-libstdc++)
    
    set_target_properties(
        ${target_name}
        PROPERTIES
            CXX_STANDART 17
            CXX_STANDART_REQUIRED ON
            CXX_EXTENSIONS OFF
    )

    if(CLANG_TIDY_EXE)
        set_target_properties(
            ${target_name}
            PROPERTIES
                CXX_CLANG_TIDY ${CLANG_TIDY_EXE}
        )
    endif()
endfunction()

function(link_mpi target_name)
    string(TOLOWER ${CMAKE_BUILD_TYPE} build_type)
    target_include_directories(${target_name} PRIVATE ${CMAKE_SOURCE_DIR}/bindings)
    target_link_libraries(${target_name} PRIVATE cpp_mpi)
    if (${build_type} STREQUAL release)
        target_link_libraries(${target_name} PRIVATE ${CMAKE_SOURCE_DIR}/../target/release/libmpi.so)
    elseif(${build_type} STREQUAL debug)
        target_link_libraries(${target_name} PRIVATE ${CMAKE_SOURCE_DIR}/../target/debug/libmpi.so)
    endif()
endfunction()
