#pragma once
#include <stddef.h>
#include <stdbool.h>

typedef struct {
    size_t neuron_count;
    float* neurons;
    float* pre_activations;
    float* weights;
    float* biases;

    // Gradients
    float* d_weights;
    float* d_biases;
    
} Layer;

Layer* init_layer(size_t neuron_count, size_t prev_layer_neuron_count);
void free_layer(Layer* layer);

typedef struct {
    size_t layer_count;
    Layer* layers;
} NeuralNetwork;

NeuralNetwork* init_neural_network();
bool append_layer(NeuralNetwork* network, size_t neuron_count);
void print_neural_network(NeuralNetwork* network);
bool free_neural_network(NeuralNetwork* network);

// activation functions
float relu(float x);
float relu_derivative(float y);

void nn_init_weights(NeuralNetwork* network);

void nn_forward_pass(NeuralNetwork* network, float* input);
void nn_backward_pass(NeuralNetwork* network, float* output, float* target);
void nn_update_weights(NeuralNetwork* network, float learning_rate, size_t batch_size);
float nn_compute_loss(NeuralNetwork* network, float* target);