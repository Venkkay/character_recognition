/*
Takafumi Hoiruchi. 2018.
https://github.com/takafumihoriuchi/MNIST_for_C
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

// set appropriate path for data
#define TRAIN_IMAGE "../data/train-images.idx3-ubyte"
#define TRAIN_LABEL "../data/train-labels.idx1-ubyte"
#define TEST_IMAGE "../data/t10k-images.idx3-ubyte"
#define TEST_LABEL "../data/t10k-labels.idx1-ubyte"

#define SIZE 784 // 28*28
#define NUM_TRAIN 60000
#define NUM_TEST 10000
#define LEN_INFO_IMAGE 4
#define LEN_INFO_LABEL 2

#define MAX_IMAGESIZE 1280
#define MAX_BRIGHTNESS 255
#define MAX_FILENAME 256
#define MAX_NUM_OF_IMAGES 1

typedef struct {
    float image[SIZE];
    int label;
} Mnist_Image;

// double train_image[NUM_TRAIN][SIZE];
// double test_image[NUM_TEST][SIZE];
// int  train_label[NUM_TRAIN];
// int test_label[NUM_TEST];


void FlipLong(unsigned char * ptr);


void read_mnist_char(char *file_path, int num_data, int len_info, int arr_n, unsigned char data_char[][arr_n], int info_arr[]);


void image_char2double(int num_data, unsigned char data_image_char[][SIZE], double data_image[][SIZE]);


void label_char2int(int num_data, unsigned char data_label_char[][1], int data_label[]);

void image_char_2_Mnist_Image(int num_data, unsigned char data_image_char[][SIZE], unsigned char data_label_char[][1], Mnist_Image data_image[]);


void load_mnist(Mnist_Image* train_image, Mnist_Image* test_image);

// name: path for saving image (ex: "./images/sample.pgm")
void save_image(int n, char name[]);


// save mnist image (call for each image)
// store train_image[][] into image[][][]
void save_mnist_pgm(double data_image[][SIZE], int index);
