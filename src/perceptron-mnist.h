//
// Created by lucas on 3/29/25.
//

#ifndef PERCEPTRON_MNIST_H
#define PERCEPTRON_MNIST_H

#define INPUT_SIZE 784  // 28x28 pixels
#define HIDDEN_SIZE 100
#define OUTPUT_SIZE 10  // 10 chiffres
#define TRAIN_SIZE 60000
#define TEST_SIZE 10000

#define NB_ITERATIONS_IN_EPOCH 15000

#define TARGETED_PRECISION 0.98

#include "../include/mnist_lib.h"

typedef struct {
    float weights_ih[HIDDEN_SIZE][INPUT_SIZE];
    float weights_ho[OUTPUT_SIZE][HIDDEN_SIZE];

    float bias_h[HIDDEN_SIZE];
    float bias_o[OUTPUT_SIZE];

    float learning_rate;
} Perceptron;


void perceptron_mnist_init(Perceptron *perceptron);
float sigmoid(const float x);
float sigmoid_derivative(const float s);
void perceptron_mnist_neuron_propagation(const Perceptron *perceptron, const float input[INPUT_SIZE], float hidden_output[HIDDEN_SIZE], float output[OUTPUT_SIZE]);
int perceptron_mnist_learning(Perceptron* perceptron, const Mnist_Image* input, const float hidden_output[HIDDEN_SIZE], const float output[OUTPUT_SIZE]);
float perceptron_mnist_testing(Perceptron *perceptron, int nb_test_images);
void perceptron_mnist_draw_training_plot_precision();
void perceptron_mnist_draw_training_plot_error();
void perceptron_mnist_training(Perceptron* perceptron, int epochs);
float perceptron_mnist_testing_all(const Perceptron *perceptron);
void perceptron_mnist_run();

#endif //PERCEPTRON_MNIST_H
