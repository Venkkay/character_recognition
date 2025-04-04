
#include "src/perceptron-2classes-simple.h"
#include "src/perceptron-2classes-wh.h"
#include "src/perceptron-10classes.h"
#include "src/perceptron-mnist.h"

int main(void) {

    int choice_exercise = 0;
    int choice_weight_init = 0;

    printf("=== Main menu ===\n");
    printf("1. Exercise 1 - Simple Perceptron with 2 outputs\n");
    printf("2. Exercise 2 - Widrow-Hoff rule perceptron with 2 outputs\n");
    printf("3. Exercise 3 - Widrow-Hoff rule perceptron with 10 outputs\n");
    printf("4. Exercise 4 - MNIST classification\n");
    printf("0. Quit\n");
    printf("Your choice: ");
    scanf("%d", &choice_exercise);

    switch(choice_exercise) {
        case 0:
            return 0;
        case 1:
        case 2:
        case 3:
            printf("\n=== Choice of weight initialisation ===\n");
            printf("1. Random between 0 and 1\n");
            printf("2. Large positive numbers\n");
            printf("3. Large negative numbers\n");
            printf("Your choice: ");
            scanf("%d", &choice_weight_init);

            if (choice_weight_init < 1 || choice_weight_init > 3) {
                printf("Choix d'initialisation invalide. Utilisation du mode par défaut (aléatoire entre 0 et 1).\n");
                choice_weight_init = 1;
            }

            if (choice_exercise == 1) {
                perceptron_2c_simple_run(choice_weight_init);
            }
            else if (choice_exercise == 2) {
                perceptron_2c_wh_run(choice_weight_init);
            }
            else {
                perceptron_10c_run(choice_weight_init);
            }
            break;
        case 4:
            perceptron_mnist_run();
        break;

        default:
            printf("Invalid choice!\n");
        return 1;
    }

    return 0;
}
