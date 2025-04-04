//
// Created by lucas on 3/13/25.
//

#ifndef PERCEPTRON_2CLASSES_SIMPLE_H
#define PERCEPTRON_2CLASSES_SIMPLE_H

#define NB_CLASSES 2
#define WIDTH 6
#define HEIGHT 8
#define NB_PIXELS (WIDTH * HEIGHT)
#define MAX_ITERATIONS 100000


// Structure to represent the network
typedef struct {
    float weights[NB_PIXELS]; // Weights of the perceptron
    float theta;            // Activation threshold
    float epsilon;          // Apprenticeship rate
    int errors[NB_CLASSES];
} Perceptron_2c_simple;

// Structure to represent the entries
typedef struct {
    int pixels[NB_PIXELS];  // Binary pixel values (0 ou 1)
    int class;             // Desired class (0 ou 1)
} Pattern_2c_simple;

// Structure pour stocker les résultats de généralisation
typedef struct {
    float noise_percent;
    float zero_error_rate;
    float one_error_rate;
} GeneralisationResult_2c_simple;

void perceptron_2c_simple_init_perceptron(Perceptron_2c_simple *p, const int choice_weight_init);
int perceptron_2c_simple_load_pattern(char *filename, Pattern_2c_simple *pattern);
int perceptron_2c_simple_load_random_pattern(Pattern_2c_simple *pattern);
int perceptron_2c_simple_heaviside_activation_function(const Perceptron_2c_simple *perceptron, const float x);
float perceptron_2c_simple_calculate_potential(const Perceptron_2c_simple *perceptron, const Pattern_2c_simple *pattern);
int perceptron_2c_simple_neuron_propagation(const Perceptron_2c_simple *perceptron, const Pattern_2c_simple *pattern);
void perceptron_2c_simple_learning(Perceptron_2c_simple *perceptron, const Pattern_2c_simple *pattern, const int error);
void perceptron_2c_simple_draw_training_plot();
double perceptron_2c_simple_distance_a_hyperplan(Perceptron_2c_simple *perceptron, Pattern_2c_simple *pattern);
void perceptron_2c_simple_train_perceptron(Perceptron_2c_simple *perceptron);
void perceptron_2c_simple_display_pattern(const Pattern_2c_simple *pattern);
void perceptron_2c_simple_test_perceptron(const Perceptron_2c_simple *perceptron);
void perceptron_2c_simple_noise_pattern(const Pattern_2c_simple *original_pattern, Pattern_2c_simple *noisy_pattern, const float noise_percentage);
float perceptron_2c_simple_test_pattern_generalisation(const Perceptron_2c_simple *perceptron, char *filename, float noise_percentage, int nb_tests);
void perceptron_2c_simple_draw_generalisation_plot(const GeneralisationResult_2c_simple *results, const int nb_points);
void perceptron_2c_simple_create_generalisation_graph(const Perceptron_2c_simple *pattern, const int nb_points, const int nb_tests_per_point);
void perceptron_2c_simple_display_weights(Perceptron_2c_simple *p);
void perceptron_2c_simple_run(int choice_weight_init);

#endif //PERCEPTRON_2CLASSES_SIMPLE_H
