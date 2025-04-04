cmake_minimum_required(VERSION 3.30)
project(character_recognition C)

set(CMAKE_C_STANDARD 23)

add_executable(character_recognition main.c
        src/perceptron-2classes-wh.c
        src/perceptron-2classes-wh.h
        src/perceptron-10classes.c
        src/perceptron-10classes.h
        src/perceptron-2classes-simple.c
        src/perceptron-2classes-simple.h
        src/perceptron-mnist.c
        src/perceptron-mnist.h
        include/mnist_lib.c
        include/mnist_lib.h
)

target_link_libraries(character_recognition PRIVATE m)