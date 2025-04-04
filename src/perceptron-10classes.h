//
// Created by lucas on 3/13/25.
//

#ifndef PERCEPTRON_10CLASSES_H
#define PERCEPTRON_10CLASSES_H

#define NB_CLASSES 10
#define WIDTH 6
#define HEIGHT 8
#define NB_PIXELS WIDTH * HEIGHT
#define ALPHA 0.000001
#define MAX_ITERATIONS 100000


// Structure to represent the network
typedef struct {
    double weights[NB_CLASSES][NB_PIXELS+1]; // Weights of the perceptron
    double theta;            // Activation threshold
    double epsilon;          // Apprenticeship rate
    double errors[NB_CLASSES];
} Perceptron_10c;

// Structure to represent the entries
typedef struct {
    int pixels[NB_PIXELS];  // Binary pixel values (0 ou 1)
    int class;             // Desired class (0 ou 1)
    int desired_output[NB_CLASSES];
} Pattern_10c;

typedef struct {
    double potential[NB_CLASSES];
    int response[NB_CLASSES];
} OutputNeurons;


// Structure pour stocker les résultats de généralisation
typedef struct generalisation_result_10c{
    double noise_percent;
    double error_rates[NB_CLASSES];
} GeneralisationResult_10c;

void perceptron_10c_init_perceptron(Perceptron_10c *p, const int choice_weight_init);
int perceptron_10c_load_pattern(char *filename, Pattern_10c *pattern);
int perceptron_10c_load_random_pattern(Pattern_10c *pattern);
int perceptron_10c_activation_function(const double potentials[NB_CLASSES]);
void perceptron_10c_calculate_potential(const Perceptron_10c *perceptron, const Pattern_10c *pattern, double potentials[NB_CLASSES]);
int perceptron_10c_neuron_propagation(const Perceptron_10c *perceptron, const Pattern_10c *pattern);
double perceptron_10c_learning(Perceptron_10c *perceptron, const Pattern_10c *pattern, const double potentials[NB_CLASSES]);
void perceptron_10c_draw_training_plot();
void perceptron_10c_training(Perceptron_10c *perceptron);
void perceptron_10c_display_pattern(const Pattern_10c *pattern);
void perceptron_10c_test_perceptron(const Perceptron_10c *perceptron);
void perceptron_10c_display_weights(const Perceptron_10c *p);
void perceptron_10c_noise_pattern(const Pattern_10c *original_pattern, Pattern_10c *noisy_pattern, const float noise_percentage);
float perceptron_10c_test_pattern_generalisation(const Perceptron_10c *perceptron, char *filename, float noise_percentage, int nb_tests);
void perceptron_10c_draw_generalisation_plot(const GeneralisationResult_10c *results, const int nb_points);
void perceptron_10c_create_generalisation_graph(const Perceptron_10c *perceptron, const int nb_points, const int nb_tests_per_point);

void perceptron_10c_run(const int choice_weight_init);

#endif //PERCEPTRON_10CLASSES_H
