//
// Created by lucas on 3/29/25.
//

#include <math.h>
#include <time.h>

#include "perceptron-mnist.h"

Mnist_Image train_image[NUM_TRAIN];
Mnist_Image test_image[NUM_TEST];

void perceptron_mnist_init(Perceptron *perceptron) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            perceptron->weights_ih[i][j] = ((float)rand() / RAND_MAX) / INPUT_SIZE; // [0, 1]
        }
        perceptron->bias_h[i] = -0.5f;
    }

    // Initialisation des poids entre couche cachée et sortie
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            perceptron->weights_ho[i][j] = ((float)rand() / RAND_MAX) / INPUT_SIZE; // [0, 1]
        }
        perceptron->bias_o[i] = -0.5f;
    }

    perceptron->learning_rate = 0.01f;
}


Mnist_Image* perceptron_mnist_load_random_pattern() {
    const int choice = rand() % NUM_TRAIN;

    return &train_image[choice];
}


float sigmoid(const float x) {
    return 1.0f / (1.0f + expf(-x));
}

float sigmoid_derivative(const float s) {
    return s * (1.0f - s);
}

void perceptron_mnist_neuron_propagation(const Perceptron *perceptron, const float input[INPUT_SIZE], float hidden_output[HIDDEN_SIZE], float output[OUTPUT_SIZE]) {
    // Calcul des sorties de la couche cachée
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        float sum = perceptron->bias_h[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += perceptron->weights_ih[i][j] * input[j];
        }
        hidden_output[i] = sigmoid(sum);
    }

    // Calcul des sorties du réseau
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float sum = perceptron->bias_o[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += perceptron->weights_ho[i][j] * hidden_output[j];
        }
        output[i] = sigmoid(sum);
    }
}

int perceptron_mnist_learning(Perceptron* perceptron, const Mnist_Image* input, const float hidden_output[HIDDEN_SIZE], const float output[OUTPUT_SIZE]) {

    float target[OUTPUT_SIZE] = {0};
    target[input->label] = 1.0;

    // Calcul de l'erreur sur la couche de sortie
    float output_delta[OUTPUT_SIZE];
    float erreur_totale = 0.0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float error = target[i] - output[i];
        erreur_totale += fabs(error);
        output_delta[i] = error * sigmoid_derivative(output[i]); // f'(x) = f(x) * (1 - f(x))
    }

    // Calcul de l'erreur sur la couche cachée
    float hidden_delta[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        float error = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            error += output_delta[j] * perceptron->weights_ho[j][i];
        }
        hidden_delta[i] = error * sigmoid_derivative(hidden_output[i]); // f'(x) = f(x) * (1 - f(x))
    }

    // Mise à jour des poids entre couche cachée et sortie
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        perceptron->bias_o[i] += perceptron->learning_rate * output_delta[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            perceptron->weights_ho[i][j] += perceptron->learning_rate * output_delta[i] * hidden_output[j];
        }
    }

    // Mise à jour des poids entre entrée et couche cachée
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        perceptron->bias_h[i] += perceptron->learning_rate * hidden_delta[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            perceptron->weights_ih[i][j] += perceptron->learning_rate * hidden_delta[i] * input->image[j];
        }
    }

    // Retour: 1 si erreur significative, 0 sinon
    return (erreur_totale > 0.1) ? 1 : 0;
}

/*
int perceptron_mnist_backpropagation(Perceptron *perceptron, Mnist_Image* image) {

    // Création du vecteur de sortie désirée (one-hot)
    float target[OUTPUT_SIZE] = {0};
    target[image->label] = 1.0;

    // Forward pass
    float hidden_output[HIDDEN_SIZE];
    float output[OUTPUT_SIZE];
    perceptron_mnist_neuron_propagation(perceptron, image->image, hidden_output, output);

    // Calcul de l'erreur sur la couche de sortie
    float output_delta[OUTPUT_SIZE];
    float erreur_totale = 0.0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float error = target[i] - output[i];
        erreur_totale += fabs(error);
        output_delta[i] = error * output[i] * (1.0 - output[i]); // f'(x) = f(x) * (1 - f(x))
    }

    // Calcul de l'erreur sur la couche cachée
    float hidden_delta[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        float error = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            error += output_delta[j] * perceptron->weights_ho[j][i];
        }
        hidden_delta[i] = error * hidden_output[i] * (1.0 - hidden_output[i]); // f'(x) = f(x) * (1 - f(x))
    }

    // Mise à jour des poids entre couche cachée et sortie
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        perceptron->bias_o[i] += perceptron->learning_rate * output_delta[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            perceptron->weights_ho[i][j] += perceptron->learning_rate * output_delta[i] * hidden_output[j];
        }
    }

    // Mise à jour des poids entre entrée et couche cachée
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        perceptron->bias_h[i] += perceptron->learning_rate * hidden_delta[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            perceptron->weights_ih[i][j] += perceptron->learning_rate * hidden_delta[i] * image_input[j];
        }
    }

    // Retour: 1 si erreur significative, 0 sinon
    return (erreur_totale > 0.1) ? 1 : 0;
}
*/


float perceptron_mnist_testing(Perceptron *perceptron, int nb_test_images) {
    int correct = 0;
    int test_size = (nb_test_images < NUM_TEST) ? nb_test_images : NUM_TEST;
    int used_indexes[NUM_TEST] = {0}; // Pour éviter de réutiliser les mêmes images

    for (int i = 0; i < test_size; i++) {
        //MnistImage *image = obtenir_image_test(i);

        int index;
        do {
            index = rand() % NUM_TEST;
        } while (used_indexes[index] && i < NUM_TEST);

        used_indexes[index] = 1;

        Mnist_Image* image = &test_image[index];


        // Forward pass
        float hidden_out[HIDDEN_SIZE];
        float output[OUTPUT_SIZE];
        perceptron_mnist_neuron_propagation(perceptron, image->image, hidden_out, output);

        // Trouver la classe prédite (indice du neurone avec la plus grande activation)
        int prediction = 0;
        float max_activation = output[0];
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (output[j] > max_activation) {
                max_activation = output[j];
                prediction = j;
            }
        }

        if (prediction == image->label) {
            correct++;
        }
    }

    return (float)correct / (float)test_size;
}

void perceptron_mnist_draw_training_plot_precision() {
    FILE *script_gnuplot = fopen("../result/temp_training_plot_precision.gp", "w");
    if (script_gnuplot == NULL) {
        fprintf(stderr, "Erreur: Unable to create the gnuplot script.\n");
        return;
    }

    fprintf(script_gnuplot, "set title 'Training curve'\n");
    fprintf(script_gnuplot, "set xlabel 'Epoch'\n");
    fprintf(script_gnuplot, "set ylabel 'Precision'\n");
    fprintf(script_gnuplot, "set grid\n");
    fprintf(script_gnuplot, "set key top left\n");
    //fprintf(script_gnuplot, "set xtics 1,2\n");
    fprintf(script_gnuplot, "plot [1:*] [0:1] '../result/training.dat' using 1:2 with line title 'Training precision'\n");
    fprintf(script_gnuplot, "pause -1 'Appuyez sur une touche pour continuer...'\n");
    fclose(script_gnuplot);

    char commande[256];

    sprintf(commande, "gnuplot -persist ../result/temp_training_plot_precision.gp &");

    system(commande);

    printf("Graph generated. A gnuplot window should open.\n");
}

void perceptron_mnist_draw_training_plot_error() {
    FILE *script_gnuplot = fopen("../result/temp_training_plot_error.gp", "w");
    if (script_gnuplot == NULL) {
        fprintf(stderr, "Erreur: Unable to create the gnuplot script.\n");
        return;
    }

    fprintf(script_gnuplot, "set title 'Training curve'\n");
    fprintf(script_gnuplot, "set xlabel 'Epoch'\n");
    fprintf(script_gnuplot, "set ylabel 'Error'\n");
    fprintf(script_gnuplot, "set grid\n");
    fprintf(script_gnuplot, "set key top left\n");
    //fprintf(script_gnuplot, "set xtics 1,2\n");
    fprintf(script_gnuplot, "plot [1:*] [0:*] '../result/training.dat' using 1:3 with line title 'Training errors'\n");
    fprintf(script_gnuplot, "pause -1 'Appuyez sur une touche pour continuer...'\n");
    fclose(script_gnuplot);

    char commande[256];

    sprintf(commande, "gnuplot -persist ../result/temp_training_plot_error.gp &");

    system(commande);

    printf("Graph generated. A gnuplot window should open.\n");
}


void perceptron_mnist_training(Perceptron* perceptron, const int epochs) {
    int epoch = 1;
    float precision = 0;

    char* filepath = "../result/training.dat";
    FILE *file = fopen(filepath, "w");
    if (file == NULL) {
        printf("Error: unable to open file %s\n", filepath);
        return;
    }
    fprintf(file, "# Epoch Precision Nb_error\n");

    do {
        int errors = 0;
        for (int i = 0; i < NB_ITERATIONS_IN_EPOCH; i++) {
            float output[OUTPUT_SIZE];
            float hidden_output[HIDDEN_SIZE];
            Mnist_Image* image = perceptron_mnist_load_random_pattern();
            perceptron_mnist_neuron_propagation(perceptron, image->image, hidden_output, output);
            errors += perceptron_mnist_learning(perceptron, image, hidden_output, output);
        }
        // Test de précision à la fin de chaque époque
        precision = perceptron_mnist_testing(perceptron, 1500); // Tester sur 1000 images
        printf("Epoch %d/%d terminée, Erreurs: %d/%d, Précision test: %.2f%%\n",
               epoch, epochs, errors, NB_ITERATIONS_IN_EPOCH,precision * 100);

        fprintf(file, "%d %f %d\n", epoch, precision, errors);
        // Si précision supérieure à 96%, on peut arrêter
        if (precision >= TARGETED_PRECISION) {
            printf("Objectif de précision atteint, arrêt de l'entraînement.\n");
        }
        epoch++;
    }while (epoch <= epochs && precision < TARGETED_PRECISION);
    fclose(file);
    perceptron_mnist_draw_training_plot_precision();
    perceptron_mnist_draw_training_plot_error();
}

float perceptron_mnist_testing_all(const Perceptron *perceptron) {
    int correct = 0;

    for (int i = 0; i < NUM_TEST; i++) {
        Mnist_Image* image = &test_image[i];

        // Forward pass
        float hidden_out[HIDDEN_SIZE];
        float output[OUTPUT_SIZE];
        perceptron_mnist_neuron_propagation(perceptron, image->image, hidden_out, output);

        // Trouver la classe prédite (indice du neurone avec la plus grande activation)
        int prediction = 0;
        float max_activation = output[0];
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (output[j] > max_activation) {
                max_activation = output[j];
                prediction = j;
            }
        }

        if (prediction == image->label) {
            correct++;
        }
    }

    return (float)correct / (float)NUM_TEST;
}

void perceptron_mnist_run() {
    srand(time(nullptr));

    load_mnist(train_image, test_image);

    Perceptron perceptron;

    perceptron_mnist_init(&perceptron);

    clock_t start = clock();
    perceptron_mnist_training(&perceptron, 200);
    clock_t end = clock();

    long time = (end - start) / CLOCKS_PER_SEC;

    printf("\nTraining time: %ld min %ld s\n", time/60, time%60);

    float precision = perceptron_mnist_testing_all(&perceptron);

    printf("\nTest on all test images : %f%%", precision);
}