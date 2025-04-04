//
// Created by lucas on 3/13/25.
//

#include "perceptron-10classes.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

char *filenames[NB_CLASSES] = {"zero.txt", "un.txt", "deux.txt", "trois.txt", "quatre.txt", "cinq.txt", "six.txt", "sept.txt", "huit.txt", "neuf.txt"};

void perceptron_10c_init_perceptron(Perceptron_10c *p, const int choice_weight_init) {
    for (int i = 0; i < NB_CLASSES; i++) {
        for (int j = 0; j < NB_PIXELS; j++) {
            if (choice_weight_init == 1) {
                p->weights[i][j] = ((float)rand() / RAND_MAX) / (NB_PIXELS);
            }
            else if (choice_weight_init == 2) {
                p->weights[i][j] = 10;
            }
            else if (choice_weight_init == 3) {
                p->weights[i][j] = -10;
            }
        }
        p->weights[i][NB_PIXELS] = -0.5;
    }
    p->epsilon = 0.01;
}

int perceptron_10c_load_pattern(char *filename, Pattern_10c *pattern) {
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

    for (int i = 0; i < NB_CLASSES; i++) {
        pattern->desired_output[i] = 0;
    }
    pattern->desired_output[pattern->class] = 1;

    fclose(file);
    return 1;
}

int perceptron_10c_load_random_pattern(Pattern_10c *pattern) {
    //char *filenames[NB_CLASSES] = {"zero.txt", "un.txt", "deux.txt", "trois.txt", "quatre.txt", "cinq.txt", "six.txt", "sept.txt", "huit.txt", "neuf.txt"};
    const int choice = rand() % NB_CLASSES;

    printf("Choice: %d\n", choice);

    return perceptron_10c_load_pattern(filenames[choice], pattern);

    /*
    switch (choice) {
        case 0:
            return perceptron_10c_load_pattern("zero.txt", pattern);
        case 1:
            return perceptron_10c_load_pattern("un.txt", pattern);
        case 2:
            return perceptron_10c_load_pattern("deux.txt", pattern);
        case 3:
            return perceptron_10c_load_pattern("trois.txt", pattern);
        case 4:
            return perceptron_10c_load_pattern("quatre.txt", pattern);
        case 5:
            return perceptron_10c_load_pattern("cinq.txt", pattern);
        case 6:
            return perceptron_10c_load_pattern("six.txt", pattern);
        case 7:
            return perceptron_10c_load_pattern("sept.txt", pattern);
        case 8:
            return perceptron_10c_load_pattern("huit.txt", pattern);
        case 9:
            return perceptron_10c_load_pattern("neuf.txt", pattern);
        default:
            printf("Error: unable to select a pattern\n");
            return 0;
    }
    */
}

int perceptron_10c_activation_function(const double potentials[NB_CLASSES]) {
    int predicted_class = 0;
    for (int i = 0; i < NB_CLASSES; i++) {
        if (potentials[i] > potentials[predicted_class]) {
            predicted_class = i;
        }
    }
    return predicted_class;
}

void perceptron_10c_calculate_potential(const Perceptron_10c *perceptron, const Pattern_10c *pattern, double potentials[NB_CLASSES]) {
    for (int i = 0; i < NB_CLASSES; i++) {
        double potential_i = 0.0f;

        for (int j = 0; j < NB_PIXELS; j++) {
            potential_i += perceptron->weights[i][j] * pattern->pixels[j];
        }

        potential_i += perceptron->weights[i][NB_PIXELS] * 1.0f;
        potentials[i] = potential_i;
    }
}


int perceptron_10c_neuron_propagation(const Perceptron_10c *perceptron, const Pattern_10c *pattern) {
    double potentials[NB_CLASSES];
    perceptron_10c_calculate_potential(perceptron, pattern, potentials);
    return perceptron_10c_activation_function(potentials);
}


double perceptron_10c_learning(Perceptron_10c *perceptron, const Pattern_10c *pattern, const double potentials[NB_CLASSES]) {
    double total_error = 0;
    for (int i = 0; i < NB_CLASSES; i++) {
        double error = pattern->desired_output[i] - potentials[i];
        total_error += fabs(error);
        if (error != 0) {
            for (int j = 0; j < NB_PIXELS; j++) {
                perceptron->weights[i][j] += perceptron->epsilon * error * pattern->pixels[j];
            }
            perceptron->weights[i][NB_PIXELS] += perceptron->epsilon * error * 1.0f;
        }
    }
    return total_error;
}

void perceptron_10c_draw_training_plot() {
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
    //fprintf(script_gnuplot, "set xtics 1,2\n");
    fprintf(script_gnuplot, "plot [1:*] [0:2] '../result/training.dat' using 1:2 with line title 'Training total errors'\n");
    fprintf(script_gnuplot, "pause -1 'Appuyez sur une touche pour continuer...'\n");
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


void perceptron_10c_training(Perceptron_10c *perceptron) {
    Pattern_10c pattern;
    double potentials[NB_CLASSES];
    //OutputNeuron outputs[NB_CLASSES];
    double total_error;
    int iteration = 0;
    for (int i = 0; i < NB_CLASSES; i++) {
        perceptron->errors[i] = 1;
    }
    char* filepath = "../result/training.dat";
    FILE *file = fopen(filepath, "w");
    if (file == NULL) {
        printf("Error: unable to open file %s\n", filepath);
        return;
    }

    fprintf(file, "# Iterations Total_error\n");

    do {
        perceptron_10c_load_random_pattern(&pattern);
        perceptron_10c_calculate_potential(perceptron, &pattern, potentials);
        total_error = perceptron_10c_learning(perceptron, &pattern, potentials);


        printf("Iteration %d, total error: %f\n", iteration, total_error);

        iteration++;
        fprintf(file, "%d %f\n", iteration, total_error);
    } while (total_error > ALPHA && iteration < MAX_ITERATIONS);
    fclose(file);
    printf("\nTraining results saved in ‘training.dat'\n");

    if (total_error == 0) {
        printf("Successful learning in %d iterations\n", iteration);
    } else {
        printf("Learning complete after %d itérations. Residual error: %f\n", iteration, total_error);
    }
    perceptron_10c_draw_training_plot();
}

void perceptron_10c_display_pattern(const Pattern_10c *pattern) {
    int index = 0;
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            printf("%c", pattern->pixels[index++] ? '*' : '.');
        }
        printf("\n");
    }
    printf("Class: %d\n", pattern->class);
}

void perceptron_10c_test_perceptron(const Perceptron_10c *perceptron) {
    Pattern_10c pattern;

    printf("\n--- Test avec motif '0' ---\n");
    perceptron_10c_load_pattern("zero.txt", &pattern);
    perceptron_10c_display_pattern(&pattern);
    int result = perceptron_10c_neuron_propagation(perceptron, &pattern);
    printf("Result: %d (expected: %d)\n", result, pattern.class);

    printf("\n--- Test avec motif '1' ---\n");
    perceptron_10c_load_pattern("un.txt", &pattern);
    perceptron_10c_display_pattern(&pattern);
    result = perceptron_10c_neuron_propagation(perceptron, &pattern);
    printf("Result: %d (expected: %d)\n", result, pattern.class);
}

void perceptron_10c_display_weights(const Perceptron_10c *p) {
    printf("Network weights:\n");

    for (int i = 0; i < NB_CLASSES; i++) {
        printf("Classe %d:\n", i);
        for (int j = 0; j < NB_PIXELS; j++) {
            printf("%6.6f ", p->weights[i][j]);
        }
        printf("Biais: %6.3f\n", p->weights[i][NB_PIXELS]);
    }
}

void perceptron_10c_noise_pattern(const Pattern_10c *original_pattern, Pattern_10c *noisy_pattern, const float noise_percentage) {
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

float perceptron_10c_test_pattern_generalisation(const Perceptron_10c *perceptron, char *filename, float noise_percentage, int nb_tests) {
    Pattern_10c original_pattern, noisy_pattern;
    int errors = 0;

    perceptron_10c_load_pattern(filename, &original_pattern);

    for (int i = 0; i < nb_tests; i++) {
        perceptron_10c_noise_pattern(&original_pattern, &noisy_pattern, noise_percentage);

        int output = perceptron_10c_neuron_propagation(perceptron, &noisy_pattern);

        if (output != original_pattern.class) {
            errors++;
        }
    }

    return (float)errors / (float)nb_tests * 100;
}

void perceptron_10c_draw_generalisation_plot(const GeneralisationResult_10c *results, const int nb_points) {
    char *error_rate_labels[NB_CLASSES] = {"Zero_error_rate", "One_error_rate", "Two_error_rate", "Three_error_rate", "Four_error_rate", "Five_error_rate", "Six_error_rate", "Seven_error_rate", "Eight_error_rate", "Nine_error_rate"};
    FILE *fichier = fopen("../result/generalisation.dat", "w");
    if (fichier != NULL) {
        fprintf(fichier, "# Noise_percentage");
        for (int i = 0; i < NB_CLASSES; i++) {
            fprintf(fichier, " %s", error_rate_labels[i]);
        }
        fprintf(fichier, "\n");
        for (int i = 0; i < nb_points; i++) {
            fprintf(fichier, "%.1f",results[i].noise_percent);
            for (int j = 0; j < NB_CLASSES; j++) {
                fprintf(fichier, " %.4f", results[i].error_rates[j]);
            }
            fprintf(fichier, "\n");
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

    fprintf(script_gnuplot, "plot '../result/generalisation.dat' using 1:2 with lines title 'Pattern 0'");
    for (int i = 1; i < NB_CLASSES; i++) {
        fprintf(script_gnuplot, ", '../result/generalisation.dat' using 1:%d with lines title 'Pattern %d'", i+2, i);
    }
    fprintf(script_gnuplot, "\n");
    fprintf(script_gnuplot, "pause -1 'Appuyez sur une touche pour continuer...'\n");
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

void perceptron_10c_create_generalisation_graph(const Perceptron_10c *perceptron, const int nb_points, const int nb_tests_per_point) {
    GeneralisationResult_10c results[nb_points];

    printf("\n--- Creating generalisation curves ---\n");

    for (int i = 0; i < nb_points; i++) {
        const float noise_percentage = 100.0f * (float)i / (float)(nb_points - 1);
        results[i].noise_percent = noise_percentage;

        for (int j = 0; j < NB_CLASSES; j++) {
            results[i].error_rates[j] = perceptron_10c_test_pattern_generalisation(perceptron, filenames[j], noise_percentage, nb_tests_per_point);
        }

        //printf("Noise %.1f%%: Error rate '0' = %.2f%%, Error rate '1' = %.2f%%\n", noise_percentage, zero_error_rate, one_error_rate);
    }

    printf("\nDrawing generalisation curves...\n");
    perceptron_10c_draw_generalisation_plot(results, nb_points);
}



void perceptron_10c_run(const int choice_weight_init) {
    srand(time(nullptr));

    Perceptron_10c perceptron;
    perceptron_10c_init_perceptron(&perceptron, choice_weight_init);
    perceptron_10c_display_weights(&perceptron);

    printf("Start learning\n");
    perceptron_10c_training(&perceptron);

    printf("Test");
    perceptron_10c_test_perceptron(&perceptron);

    perceptron_10c_create_generalisation_graph(&perceptron, 101, 100);
    perceptron_10c_display_weights(&perceptron);
}
