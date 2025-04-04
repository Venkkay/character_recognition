//
// Created by lucas on 3/10/25.
//

#ifndef PERCEPTRON_2CLASSES_H
#define PERCEPTRON_2CLASSES_H

#define NB_CLASSES 2
#define WIDTH 6
#define HEIGHT 8
#define NB_PIXELS (WIDTH * HEIGHT)
#define ALPHA 0.001
#define MAX_ITERATIONS 100000

// Structure to represent the network
typedef struct {
    float weights[NB_PIXELS]; // Weights of the perceptron
    float theta;            // Activation threshold
    float epsilon;          // Apprenticeship rate
    float errors[NB_CLASSES];
} Perceptron_2c_wh;

// Structure to represent the entries
typedef struct {
    int pixels[NB_PIXELS];  // Binary pixel values (0 ou 1)
    int class;             // Desired class (0 ou 1)
} Pattern_2c_wh;

// Structure pour stocker les résultats de généralisation
typedef struct {
    float noise_percent;
    float zero_error_rate;
    float one_error_rate;
} GeneralisationResult_2c_wh;

void perceptron_2c_wh_init_perceptron(Perceptron_2c_wh *p, const int choice_weight_init);
int perceptron_2c_wh_load_pattern(char *filename, Pattern_2c_wh *pattern);
int perceptron_2c_wh_load_random_pattern(Pattern_2c_wh *pattern);
int perceptron_2c_wh_heaviside_activation_function(const Perceptron_2c_wh *perceptron, const float x);
float perceptron_2c_wh_calculate_potential(const Perceptron_2c_wh *perceptron, const Pattern_2c_wh *pattern);
int perceptron_2c_wh_neuron_propagation(const Perceptron_2c_wh *perceptron, const Pattern_2c_wh *pattern);
void perceptron_2c_wh_learning(Perceptron_2c_wh *perceptron, const Pattern_2c_wh *pattern, const float error);
void perceptron_2c_wh_draw_training_plot();
double perceptron_2c_wh_distance_a_hyperplan(Perceptron_2c_wh *perceptron, Pattern_2c_wh *pattern);
void perceptron_2c_wh_train_perceptron(Perceptron_2c_wh *perceptron, int max_iterations);
void perceptron_2c_wh_display_pattern(const Pattern_2c_wh *pattern);
void perceptron_2c_wh_test_perceptron(const Perceptron_2c_wh *perceptron);
void perceptron_2c_wh_noise_pattern(const Pattern_2c_wh *original_pattern, Pattern_2c_wh *noisy_pattern, const float noise_percentage);
float perceptron_2c_wh_test_pattern_generalisation(const Perceptron_2c_wh *perceptron, char *filename, float noise_percentage, int nb_tests);
void perceptron_2c_wh_draw_generalisation_plot(const GeneralisationResult_2c_wh *results, const int nb_points);
void perceptron_2c_wh_create_generalisation_graph(const Perceptron_2c_wh *pattern, const int nb_points, const int nb_tests_per_point);
void perceptron_2c_wh_display_weights(Perceptron_2c_wh *p);
void perceptron_2c_wh_run(int choice_weight_init);

#endif //PERCEPTRON_2CLASSES_H
