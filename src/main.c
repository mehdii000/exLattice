#include "../include/neural_network.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

#define MNIST_IMAGE_SIZE 784
#define MNIST_TRAIN_COUNT 60000
#define BATCH_SIZE 32
#define LEARNING_RATE 0.001f
#define MAX_EPOCHS 10

// Helper to reverse Big-Endian integers from MNIST files
uint32_t reverse_int(uint32_t i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
    return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + c4;
}

int main(void) {
    srand(time(NULL));

    // 1. Load MNIST Images
    FILE* img_file = fopen("data/train-images.idx3-ubyte", "rb");
    if (!img_file) { 
        perror("Failed to open images. Did you run the wget commands in Step 1?"); 
        return 1; 
    }
    
    uint32_t magic, count, rows, cols;
    fread(&magic, 4, 1, img_file);
    fread(&count, 4, 1, img_file);
    fread(&rows, 4, 1, img_file);
    fread(&cols, 4, 1, img_file);
    count = reverse_int(count);

    printf("Loading %u images...\n", count);
    float* all_images = malloc(count * MNIST_IMAGE_SIZE * sizeof(float));
    unsigned char* img_buffer = malloc(MNIST_IMAGE_SIZE);
    for (uint32_t i = 0; i < count; i++) {
        fread(img_buffer, 1, MNIST_IMAGE_SIZE, img_file);
        for (int j = 0; j < MNIST_IMAGE_SIZE; j++) {
            all_images[i * MNIST_IMAGE_SIZE + j] = img_buffer[j] / 255.0f;
        }
    }
    fclose(img_file);

    // 2. Load MNIST Labels
    FILE* lbl_file = fopen("data/train-labels.idx1-ubyte", "rb");
    if (!lbl_file) { perror("Failed to open labels"); return 1; }
    fread(&magic, 4, 1, lbl_file);
    fread(&count, 4, 1, lbl_file);
    count = reverse_int(count);

    uint8_t* all_labels = malloc(count);
    fread(all_labels, 1, count, lbl_file);
    fclose(lbl_file);

    // 3. Setup Network
    NeuralNetwork* network = init_neural_network();
    append_layer(network, 784); // 28x28 pixels
    append_layer(network, 128); // Hidden layer
    append_layer(network, 10);  // 10 digits (0-9)
    nn_init_weights(network);

    printf("Started Training on %u MNIST images...\n", count);

    uint32_t* indices = malloc(count * sizeof(uint32_t));
    for (uint32_t i = 0; i < count; i++) indices[i] = i;

    // 4. Training Loop
    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        // Shuffle indices
        for (uint32_t i = count - 1; i > 0; i--) {
            uint32_t j = rand() % (i + 1);
            uint32_t temp = indices[i]; indices[i] = indices[j]; indices[j] = temp;
        }

        float total_loss = 0;
        for (uint32_t i = 0; i < count; i += BATCH_SIZE) {
            uint32_t current_batch_size = (i + BATCH_SIZE > count) ? (count - i) : BATCH_SIZE;
            
            for (uint32_t b = 0; b < current_batch_size; b++) {
                uint32_t idx = indices[i + b];
                float* input = &all_images[idx * MNIST_IMAGE_SIZE];
                
                // One-Hot target
                float target[10] = {0};
                target[all_labels[idx]] = 1.0f;

                nn_forward_pass(network, input);
                total_loss += nn_compute_loss(network, target);
                nn_backward_pass(network, network->layers[network->layer_count - 1].neurons, target);
            }
            nn_update_weights(network, LEARNING_RATE, current_batch_size);
            
            if (i % 3200 == 0) {
                printf("Epoch %d, Progress: %.1f%%, Avg Loss: %f\r", epoch, (float)i/count*100, total_loss / (i + current_batch_size));
                fflush(stdout);
            }
        }
        printf("\nEpoch %d completed. Average Loss: %f\n", epoch, total_loss / count);
    }

    // Cleanup
    free(all_images); free(all_labels); free(indices); free(img_buffer);
    free_neural_network(network);
    return 0;
}