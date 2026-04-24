#include "../include/neural_network.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>

float relu(float x) {
    return x > 0 ? x : 0;
}

float relu_derivative(float y) {
    return y > 0 ? 1 : 0;
}

void nn_init_weights(NeuralNetwork *network) {
    if (!network) return;
    // He initialization for ReLU: scale = sqrt(2/n_in)
    for (size_t i = 1; i < network->layer_count; i++) {
        float scale = sqrtf(2.0f / network->layers[i - 1].neuron_count);
        for (size_t j = 0; j < network->layers[i].neuron_count; j++) {
            for (size_t k = 0; k < network->layers[i - 1].neuron_count; k++) {
                // Initialize weights with small random values scaled by He factor
                network->layers[i].weights[j * network->layers[i - 1].neuron_count + k] = 
                    ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
            }
            network->layers[i].biases[j] = 0.01f; // Small positive bias to help ReLU
        }
    }
}

void nn_forward_pass(NeuralNetwork *network, float* input) {
    if (!network || !input) return;
    // Copy the input to the first layer
    for (size_t i = 0; i < network->layers[0].neuron_count; i++) {
        network->layers[0].neurons[i] = input[i];
    }
    // Loop the network layers starting from the second layer
    for (size_t i = 1; i < network->layer_count; i++) {
        // Loop the neurons of the current layer
        for (size_t j = 0; j < network->layers[i].neuron_count; j++) {
            // Calculate the weighted sum of the previous layer's neurons
            float weighted_sum = 0.0f;
            for (size_t k = 0; k < network->layers[i - 1].neuron_count; k++) {
                weighted_sum += network->layers[i].weights[j * network->layers[i - 1].neuron_count + k] * network->layers[i - 1].neurons[k];
            }
            // Add the bias
            weighted_sum += network->layers[i].biases[j];
            // Store the pre-activation value (z)
            network->layers[i].pre_activations[j] = weighted_sum;
            // Apply the activation function
            network->layers[i].neurons[j] = relu(weighted_sum);
        }
    }
}

// Now for my nemesis, the backward pass, help only slightly from AI
void nn_backward_pass(NeuralNetwork *network, float *output, float *target) {

    // deltaOutput = dC/dz for the last layer
    float* deltaOutput = malloc(network->layers[network->layer_count - 1].neuron_count * sizeof(float));
    for (size_t i = 0; i < network->layers[network->layer_count - 1].neuron_count; i++) {
        deltaOutput[i] = 2 * (output[i] - target[i]) * relu_derivative(network->layers[network->layer_count - 1].pre_activations[i]);
        // dC/dW and dC/db for the last layer
        for (size_t j = 0; j < network->layers[network->layer_count - 2].neuron_count; j++) {
            network->layers[network->layer_count - 1].d_weights[i * network->layers[network->layer_count - 2].neuron_count + j] += deltaOutput[i] * network->layers[network->layer_count - 2].neurons[j];
        }
        network->layers[network->layer_count - 1].d_biases[i] += deltaOutput[i];
    }
    
    // For each layer from the L-1 to the second layer, calculate the delta error propagated from deltaOutput going backward
    for (size_t i = network->layer_count - 2; i > 0; i--) {
        float* deltaError = malloc(network->layers[i].neuron_count * sizeof(float));
        for (size_t j = 0; j < network->layers[i].neuron_count; j++) {
            deltaError[j] = 0.0f;
            for (size_t k = 0; k < network->layers[i + 1].neuron_count; k++) {
                deltaError[j] += network->layers[i + 1].weights[k * network->layers[i].neuron_count + j] * deltaOutput[k];
            }
            deltaError[j] *= relu_derivative(network->layers[i].pre_activations[j]);
        }

        // dC/dW and dC/db for the current layer
        for (size_t j = 0; j < network->layers[i].neuron_count; j++) {
            for (size_t k = 0; k < network->layers[i - 1].neuron_count; k++) {
                network->layers[i].d_weights[j * network->layers[i - 1].neuron_count + k] += deltaError[j] * network->layers[i - 1].neurons[k];
            }
            network->layers[i].d_biases[j] += deltaError[j];
        }

        free(deltaOutput);
        deltaOutput = deltaError;
    }
    free(deltaOutput);
}

void nn_update_weights(NeuralNetwork *network, float learning_rate, size_t batch_size) {
    for (size_t i = 1; i < network->layer_count; i++) {
        for (size_t j = 0; j < network->layers[i].neuron_count; j++) {
            for (size_t k = 0; k < network->layers[i - 1].neuron_count; k++) {
                network->layers[i].weights[j * network->layers[i - 1].neuron_count + k] -= learning_rate * network->layers[i].d_weights[j * network->layers[i - 1].neuron_count + k] / batch_size;
                network->layers[i].d_weights[j * network->layers[i - 1].neuron_count + k] = 0.0f;
            }
            network->layers[i].biases[j] -= learning_rate * network->layers[i].d_biases[j] / batch_size;
            network->layers[i].d_biases[j] = 0.0f;
        }
    }
}

float nn_compute_loss(NeuralNetwork *network, float *target) {
    float loss = 0.0f;
    // Mean squared error
    for (size_t i = 0; i < network->layers[network->layer_count - 1].neuron_count; i++) {
        loss += (network->layers[network->layer_count - 1].neurons[i] - target[i]) * (network->layers[network->layer_count - 1].neurons[i] - target[i]);
    }
    return loss;
}