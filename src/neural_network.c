#include "../include/neural_network.h"
#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <stdbool.h>

Layer* init_layer(size_t neuron_count, size_t prev_layer_neuron_count) {
    Layer* layer = malloc(sizeof(Layer));
    if (!layer) {
        fprintf(stderr, "Failed to allocate memory for layer\n");
        return NULL;
    }
    layer->neuron_count = neuron_count;
    layer->neurons = calloc(neuron_count, sizeof(float));
    layer->pre_activations = calloc(neuron_count, sizeof(float));
    
    if (prev_layer_neuron_count > 0) {
        layer->weights = calloc(neuron_count * prev_layer_neuron_count, sizeof(float));
        layer->biases = calloc(neuron_count, sizeof(float));
        layer->d_weights = calloc(neuron_count * prev_layer_neuron_count, sizeof(float));
        layer->d_biases = calloc(neuron_count, sizeof(float));
    } else {
        // input layer has no weights, biases, or gradients
        layer->weights = NULL;
        layer->biases = NULL;
        layer->d_weights = NULL;
        layer->d_biases = NULL;
    }
    
    if (!layer->neurons || !layer->pre_activations) {
        free_layer(layer);
        return NULL;
    }
    return layer;
}

void free_layer(Layer* layer) {
    if (!layer) return;
    free(layer->neurons);
    free(layer->pre_activations);
    free(layer->weights);
    free(layer->biases);
    free(layer->d_weights);
    free(layer->d_biases);
    free(layer);
}

NeuralNetwork* init_neural_network() {
    NeuralNetwork* network = malloc(sizeof(NeuralNetwork));
    if (!network) {
        return NULL;
    }
    network->layer_count = 0;
    network->layers = NULL;
    return network;
}

bool append_layer(NeuralNetwork* network, size_t neuron_count) {
    size_t prev_layer_neuron_count = (network->layer_count == 0) ? 0 : network->layers[network->layer_count - 1].neuron_count;
    Layer* new_layer = init_layer(neuron_count, prev_layer_neuron_count);
    if (!new_layer) {
        return false;
    }
    Layer* temp = realloc(network->layers, (network->layer_count + 1) * sizeof(Layer));
    if (!temp) {
        free_layer(new_layer);
        return false;
    }
    network->layers = temp;
    network->layers[network->layer_count] = *new_layer;
    free(new_layer); // Free the container, but neurons are now pointed to by the array element
    network->layer_count++;
    return true;
}

void print_neural_network(NeuralNetwork* network) {
    if (!network) return;
    printf("\nNeural Network:\n");
    for (size_t i = 0; i < network->layer_count; i++) {
        printf("Layer %zu: %zu neurons\n", i, network->layers[i].neuron_count);
        for (size_t j = 0; j < network->layers[i].neuron_count; j++) {
            printf("  Neuron %zu: %f\n", j, network->layers[i].neurons[j]);
            if (network->layers[i].weights) {
                printf("    Weights: [ ");
                for (size_t k = 0; k < network->layers[i - 1].neuron_count; k++) {
                    printf("%f ", network->layers[i].weights[j * network->layers[i - 1].neuron_count + k]);
                }
                printf("]\n");
            }
            if (network->layers[i].biases) {
                printf("    Bias: %f\n", network->layers[i].biases[j]);
            }
        }
    }
}

bool free_neural_network(NeuralNetwork* network) {
    if (!network) return true;
    for (size_t i = 0; i < network->layer_count; i++) {
        free(network->layers[i].neurons);
        free(network->layers[i].pre_activations);
        free(network->layers[i].weights);
        free(network->layers[i].biases);
        free(network->layers[i].d_weights);
        free(network->layers[i].d_biases);
    }
    free(network->layers);
    free(network);
    return true;
}