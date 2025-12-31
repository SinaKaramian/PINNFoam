add_executable(torch_gpu_app
  main.cpp
)

target_link_libraries(torch_gpu_app
  PRIVATE smoke_lib
)
