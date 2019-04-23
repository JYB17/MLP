#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "Layer.h"
#include "Network.h"

//#define XOR
#define MNIST

#define CROSS_ENTROPY
//#define MSE

Network::Network(){
	n_layers = 0;
	layers = NULL;
}

void Network::Alloc(int _n_layers, int *nodes){
	n_layers = _n_layers;

	layers = new Layer*[n_layers];
	assert(layers!=NULL);
	for(int i=0; i<n_layers; i++){
		layers[i] = new Layer;
		assert(layers[i]!=NULL);
		layers[i]->Alloc(nodes[i], nodes[i+1]);
	}
}

void Network::ForwardNetwork(float *_input){
#ifdef	XOR
	layers[0]->Forward(_input, 0);

        for(int i=1; i<n_layers; i++){
                layers[i]->Forward(layers[i-1]->getOutput(), 0);
        }
#endif	// XOR

#ifdef	MNIST
	layers[0]->Forward(_input, 2);

	for(int i=1; i<n_layers-1; i++){
		layers[i]->Forward(layers[i-1]->getOutput(), 2);
	}

#ifdef	CROSS_ENTROPY
	layers[n_layers-1]->Forward(layers[n_layers-2]->getOutput(), 3);
#endif	// CROSS_ENTROPY

#ifdef	MSE
	layers[n_layers-1]->Forward(layers[n_layers-2]->getOutput(), 0);
#endif	// MSE

#endif	// MNIST

}

float Network::LossNetwork(float *label){
#ifdef	CROSS_ENTROPY	
	float loss = layers[n_layers-1]->CrossEntropyLoss(label);
#endif	// CROSS_ENTROPY

#ifdef	MSE
	float loss = layers[n_layers-1]->MSELoss(label);
#endif	// MSE

	return loss;
}

void Network::BackPropagationNetwork(float *label){
#ifdef	XOR
	for(int i=n_layers-1; i>=0; i--){
                if(i==n_layers-1)
                        layers[i]->dMSELoss(label);
                else
                        layers[i+1]->dLoss(layers[i]->getdLoss());

                layers[i]->BackPropagation(0);
        }
#endif	// XOR

#ifdef	MNIST
	for(int i=n_layers-1; i>=0; i--){
		if(i==n_layers-1){

#ifdef	CROSS_ENTROPY			
			layers[i]->dCrossEntropyLossdSoftmax(label);
			layers[i]->BackPropagationSoftmax();
#endif	// CROSS_ENTROPY

#ifdef	MSE
			layers[i]->dMSELoss(label);
			layers[i]->BackPropagation(0);
#endif	// MSE

		}
		else{
			layers[i+1]->dLoss(layers[i]->getdLoss());
			layers[i]->BackPropagation(2);
		}
	}
#endif	// MNIST
}

void Network::updateWeightNetwork(float learning_rate, float lambda){
#ifdef XOR
	for(int i=0; i<n_layers; i++){
		layers[i]->updateWeight(learning_rate, lambda, 0);
	}
#endif	// XOR

#ifdef MNIST
	for(int i=0; i<n_layers; i++){
		layers[i]->updateWeight(learning_rate, lambda, 1);
	}
#endif	// MNIST
}

void Network::updateWeightMomentumNetwork(float learning_rate, float mu, float lambda){
	for(int i=0; i<n_layers; i++){
		layers[i]->updateWeightMomentum(learning_rate, mu, lambda, 1);
	}
}
