//
// Created by lucas on 3/13/25.
//

#include "perceptron-2classes-simple.h"


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

void perceptron_2c_simple_init_perceptron(Perceptron_2c_simple *p, const int choice_weight_init) {
    for (int i = 0; i < NB_PIXELS; i++) {
        if (choice_weight_init == 1) {
            p->weights[i] = ((float)rand() / RAND_MAX) / (NB_PIXELS);
        }
        else if (choice_weight_init == 2) {
            p->weights[i] = 10;
        }
        else if (choice_weight_init == 3) {
            p->weights[i] = -10;
        }
    }

    p->theta = -0.5f;
    p->epsilon = 0.01f;
}

int perceptron_2c_simple_load_pattern(char *filename, Pattern_2c_simple *pattern) {
    char filepath[100];
    sprintf(filepath, "../data/%s", filename);
    FILE *file = fopen(filepath, "r");
    if (file == NULL) {
        printf("Error: unable to open file %s\n", filepath);
        return 0;
    }

    int index = 0;

    for (int h = 0; h < HEIGHT; h++) {
        char line[WIDTH + 2];
        if (fgets(line, WIDTH + 2, file) == NULL) {
            printf("Erreur: format de fichier incorrect\n");
            fclose(file);
            return 0;
        }

        for (int i = 0; i < WIDTH; i++) {
            if (line[i] == '*') {
                pattern->pixels[index] = 1;
            } else {
                pattern->pixels[index] = 0;
            }
            index++;
        }
    }

    if (fscanf(file, "%d", &pattern->class) != 1) {
        printf("Error: unable to read the class\n");
        fclose(file);
        return 0;
    }

    fclose(file);
    return 1;
}

int perceptron_2c_simple_load_random_pattern(Pattern_2c_simple *pattern) {
    if (rand() % 2 == 0) {
        return perceptron_2c_simple_load_pattern("zero.txt", pattern);
    } else {
        return perceptron_2c_simple_load_pattern("un.txt", pattern);
    }
}

int perceptron_2c_simple_heaviside_activation_function(const Perceptron_2c_simple *perceptron, const float x) {
    return (x >= fabsf(perceptron->theta)) ? 1 : 0;
}

float perceptron_2c_simple_calculate_potential(const Perceptron_2c_simple *perceptron, const Pattern_2c_simple *pattern) {
    float potential_i = 0.0f;

    for (int i = 0; i < NB_PIXELS; i++) {
        potential_i += perceptron->weights[i] * (float)pattern->pixels[i];
    }

    potential_i += perceptron->theta * 1.0f;

    return potential_i;
}

int perceptron_2c_simple_neuron_propagation(const Perceptron_2c_simple *perceptron, const Pattern_2c_simple *pattern) {
    const float potential_i = perceptron_2c_simple_calculate_potential(perceptron, pattern);
    return perceptron_2c_simple_heaviside_activation_function(perceptron, potential_i);
}

void perceptron_2c_simple_learning(Perceptron_2c_simple *perceptron, const Pattern_2c_simple *pattern, const int error) {
    if (error != 0) {
        for (int i = 0; i < NB_PIXELS; i++) {
            perceptron->weights[i] += perceptron->epsilon * (float)error * (float)pattern->pixels[i];
        }
        perceptron->theta += perceptron->epsilon * (float)error * 1.0f;
    }
}

void perceptron_2c_simple_draw_training_plot() {
    FILE *script_gnuplot = fopen("../result/temp_training_plot.gp", "w");
    if (script_gnuplot == NULL) {
        fprintf(stderr, "Erreur: Unable to create the gnuplot script.\n");
        return;
    }

    fprintf(script_gnuplot, "set title 'Training curve'\n");
    fprintf(script_gnuplot, "set xlabel 'Iterations'\n");
    fprintf(script_gnuplot, "set ylabel 'Total error'\n");
    fprintf(script_gnuplot, "set grid\n");
    fprintf(script_gnuplot, "set key top left\n");
    fprintf(script_gnuplot, "plot [1:*] [0:2] '../result/training.dat' using 1:2 with line title 'Training total errors'\n");
    fclose(script_gnuplot);

    char commande[256];
#ifdef _WIN32
    sprintf(commande, "start gnuplot -persist ../result/temp_training_plot");
#else
    sprintf(commande, "gnuplot -persist ../result/temp_training_plot.gp &");
#endif

    system(commande);

    printf("Graph generated. A gnuplot window should open.\n");
}

double perceptron_2c_simple_distance_a_hyperplan(Perceptron_2c_simple *perceptron, Pattern_2c_simple *pattern) {
    float potentiel = 0.0f;
    float norme_poids = 0.0f;

    for (int i = 0; i < NB_PIXELS; i++) {
        potentiel += perceptron->weights[i] * (float)pattern->pixels[i];
        norme_poids += perceptron->weights[i] * perceptron->weights[i];
    }

    potentiel += perceptron->theta;
    norme_poids += perceptron->theta * perceptron->theta;

    return fabs(potentiel) / sqrt(norme_poids);
}

void perceptron_2c_simple_train_perceptron(Perceptron_2c_simple *perceptron) {
    Pattern_2c_simple pattern;
    int total_error;
    int iteration = 0;
    perceptron->errors[0] = 1;
    perceptron->errors[1] = 1;
    char* filepath = "../result/training.dat";
    FILE *file = fopen(filepath, "w");
    if (file == NULL) {
        printf("Error: unable to open file %s\n", filepath);
        return;
    }

    fprintf(file, "# Iterations Total_error\n");

    do {
        total_error = 0;

        perceptron_2c_simple_load_random_pattern(&pattern);
        const int output = perceptron_2c_simple_neuron_propagation(perceptron, &pattern);
        perceptron->errors[pattern.class] = pattern.class - output;
        perceptron_2c_simple_learning(perceptron, &pattern, perceptron->errors[pattern.class]);

        for (int i = 0; i < NB_CLASSES; i++) {
            total_error += abs(perceptron->errors[i]);
        }

        printf("Iteration %d, total error: %d\n", iteration, total_error);

        iteration++;
        fprintf(file, "%d %d\n", iteration, total_error);
    } while (total_error > 0 && iteration < MAX_ITERATIONS);
    fclose(file);
    printf("\nTraining results saved in ‘training.dat'\n");

    if (total_error == 0) {
        printf("Successful learning in %d iterations\n", iteration);
    } else {
        printf("Learning complete after %d itérations. Residual error: %f\n", iteration, total_error);
    }
    perceptron_2c_simple_draw_training_plot();
    perceptron_2c_simple_load_pattern("zero.txt", &pattern);
    printf("Distance à l'hyperplan 0 : %f", perceptron_2c_simple_distance_a_hyperplan(perceptron, &pattern));
    perceptron_2c_simple_load_pattern("un.txt", &pattern);
    printf("Distance à l'hyperplan 1 : %f\n", perceptron_2c_simple_distance_a_hyperplan(perceptron, &pattern));
}

void perceptron_2c_simple_display_pattern(const Pattern_2c_simple *pattern) {
    int index = 0;
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            printf("%c", pattern->pixels[index++] ? '*' : '.');
        }
        printf("\n");
    }
    printf("Class: %d\n", pattern->class);
}

void perceptron_2c_simple_test_perceptron(const Perceptron_2c_simple *perceptron) {
    Pattern_2c_simple pattern;

    printf("\n--- Test avec motif '0' ---\n");
    perceptron_2c_simple_load_pattern("zero.txt", &pattern);
    perceptron_2c_simple_display_pattern(&pattern);
    int result = perceptron_2c_simple_neuron_propagation(perceptron, &pattern);
    printf("Result: %d (expected: %d)\n", result, pattern.class);

    printf("\n--- Test avec motif '1' ---\n");
    perceptron_2c_simple_load_pattern("un.txt", &pattern);
    perceptron_2c_simple_display_pattern(&pattern);
    result = perceptron_2c_simple_neuron_propagation(perceptron, &pattern);
    printf("Result: %d (expected: %d)\n", result, pattern.class);
}

void perceptron_2c_simple_noise_pattern(const Pattern_2c_simple *original_pattern, Pattern_2c_simple *noisy_pattern, const float noise_percentage) {
    for (int i = 0; i < NB_PIXELS; i++) {
        noisy_pattern->pixels[i] = original_pattern->pixels[i];
    }
    noisy_pattern->class = original_pattern->class;

    int nb_pixels_to_reverse = (int)(NB_PIXELS * noise_percentage / 100.0);
    int current_noisy_pixel_nb = 0;
    int current_noisy_pixels[NB_PIXELS] = {0};

    while (current_noisy_pixel_nb < nb_pixels_to_reverse) {
        int pixel = rand() % (NB_PIXELS);
        if (current_noisy_pixels[pixel] == 0) {
            noisy_pattern->pixels[pixel] = 1 - noisy_pattern->pixels[pixel];
            current_noisy_pixels[pixel] = 1;
            current_noisy_pixel_nb++;
        }
    }
}

float perceptron_2c_simple_test_pattern_generalisation(const Perceptron_2c_simple *perceptron, char *filename, float noise_percentage, int nb_tests) {
    Pattern_2c_simple original_pattern, noisy_pattern;
    int errors = 0;

    perceptron_2c_simple_load_pattern(filename, &original_pattern);

    for (int i = 0; i < nb_tests; i++) {
        perceptron_2c_simple_noise_pattern(&original_pattern, &noisy_pattern, noise_percentage);

        int output = perceptron_2c_simple_neuron_propagation(perceptron, &noisy_pattern);

        if (output != original_pattern.class) {
            errors++;
        }
    }

    return (float)errors / (float)nb_tests * 100;
}

void perceptron_2c_simple_draw_generalisation_plot(const GeneralisationResult_2c_simple *results, const int nb_points) {
    FILE *fichier = fopen("../result/generalisation.dat", "w");
    if (fichier != NULL) {
        fprintf(fichier, "# Noise_percentage Zero_error_rate One_error_rate\n");
        for (int i = 0; i < nb_points; i++) {
            fprintf(fichier, "%.1f %.4f %.4f\n",
                    results[i].noise_percent,
                    results[i].zero_error_rate,
                    results[i].one_error_rate);
        }
        fclose(fichier);
        printf("\nResults saved in ‘generalisation.dat'\n");
    }

    FILE *script_gnuplot = fopen("../result/temp_plot.gp", "w");
    if (script_gnuplot == NULL) {
        fprintf(stderr, "Erreur: Unable to create the gnuplot script.\n");
        return;
    }

    fprintf(script_gnuplot, "set title 'Generalisation curves'\n");
    fprintf(script_gnuplot, "set xlabel 'Noise percentage'\n");
    fprintf(script_gnuplot, "set ylabel 'Error rate'\n");
    fprintf(script_gnuplot, "set grid\n");
    fprintf(script_gnuplot, "set key top left\n");

    fprintf(script_gnuplot, "plot '../result/generalisation.dat' using 1:2 with lines title 'Pattern 0', ");
    fprintf(script_gnuplot, "'../result/generalisation.dat' using 1:3 with lines title 'Pattern 1'\n");
    fclose(script_gnuplot);

    char commande[256];
    #ifdef _WIN32
        sprintf(commande, "start gnuplot -persist temp_plot.gp");
    #else
        sprintf(commande, "gnuplot -persist ../result/temp_plot.gp &");
    #endif

    system(commande);

    printf("Graph generated. A gnuplot window should open.\n");
}

void perceptron_2c_simple_create_generalisation_graph(const Perceptron_2c_simple *pattern, const int nb_points, const int nb_tests_per_point) {
    GeneralisationResult_2c_simple results[nb_points];

    printf("\n--- Creating generalisation curves ---\n");

    for (int i = 0; i < nb_points; i++) {
        const float noise_percentage = 100.0f * (float)i / (float)(nb_points - 1);

        const float zero_error_rate = perceptron_2c_simple_test_pattern_generalisation(pattern, "zero.txt", noise_percentage, nb_tests_per_point);
        const float one_error_rate = perceptron_2c_simple_test_pattern_generalisation(pattern, "un.txt", noise_percentage, nb_tests_per_point);

        results[i].noise_percent = noise_percentage;
        results[i].zero_error_rate = zero_error_rate;
        results[i].one_error_rate = one_error_rate;

        //printf("Noise %.1f%%: Error rate '0' = %.2f%%, Error rate '1' = %.2f%%\n", noise_percentage, zero_error_rate, one_error_rate);
    }

    printf("\nDrawing generalisation curves...\n");
    perceptron_2c_simple_draw_generalisation_plot(results, nb_points);
}

void perceptron_2c_simple_display_weights(Perceptron_2c_simple *p) {
    printf("Poids du réseau:\n");

    int indice = 0;
    for (int h = 0; h < HEIGHT; h++) {
        for (int l = 0; l < WIDTH; l++) {
            printf("%6.6f ", p->weights[indice++]);
        }
        printf("\n");
    }

    printf("Biais: %6.3f\n", p->theta);
}


void perceptron_2c_simple_run(int choice_weight_init){
    srand(time(nullptr));

    Perceptron_2c_simple perceptron;
    perceptron_2c_simple_init_perceptron(&perceptron, choice_weight_init);
    perceptron_2c_simple_display_weights(&perceptron);

    printf("Start learning\n");
    perceptron_2c_simple_train_perceptron(&perceptron);

    printf("Test");
    perceptron_2c_simple_test_perceptron(&perceptron);

    perceptron_2c_simple_create_generalisation_graph(&perceptron, 101, 100);
    perceptron_2c_simple_display_weights(&perceptron);
}
