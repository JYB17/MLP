#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cfloat>

#include "Layer.h"

Layer::Layer(){
	input = NULL;
	output = NULL;
	weight = NULL;
	gradient = NULL;
	d_loss = NULL;
	d_actFun = NULL;
	d_loss_actFun = NULL;

	momentum_vector = NULL;

	dim_input = 0;
	dim_output = 0;
	dim_weight = 0;
}

void Layer::Alloc(int _dim_input, int _dim_output){
	dim_input = _dim_input;
	dim_output = _dim_output;
	dim_weight = dim_output*(dim_input+1);

	input = NULL;

	output = new float[dim_output];
	assert(output!=NULL);

	weight = new float[dim_weight];
	assert(weight!=NULL);
	float range = 0.1/(float)sqrt(dim_input+1.0);
	for(int i=0; i<dim_weight; i++){
		weight[i] = rand()/(float)RAND_MAX * 2*range - range;		// initiate weight by random number between [-0.1, 0.1]
	}

	gradient = new float[dim_weight];
	assert(gradient!=NULL);
	memset(gradient, 0, dim_weight*sizeof(float));

	momentum_vector = new float[dim_weight];
	assert(momentum_vector!=NULL);
	memset(momentum_vector, 0, dim_weight*sizeof(float));

	d_loss = new float[dim_output];
	assert(d_loss!=NULL);

	d_actFun = new float[dim_output];
	assert(d_actFun!=NULL);

	d_loss_actFun = new float[dim_output];
	assert(d_loss_actFun!=NULL);
}

float Layer::Sigmoid(float net){
	return 1.0/(1.0+(float)exp(-net));
}

float Layer::Tanh(float net){
	return (1.0-(float)exp(-net))/(1.0+(float)exp(-net));
}

float Layer::ReLU(float net){
	if(net>0.0)	return net;
	else		return 0.0;
}

void Layer::Forward(float *_input, int actFun){
	input = _input;

	for(int i=0; i<dim_output; i++){
		float net = 0.0;
		float *in_weight = weight+i*(dim_input+1);
		for(int j=0; j<dim_input; j++)						
			net += _input[j]*in_weight[j];						//multiplyMatrix
		net += in_weight[dim_input];				// bias

		output[i] = net;
	}
	ApplyActivationFunction(actFun);
}

void Layer::ApplyActivationFunction(int actFun){
	float sum = 0.F;

	switch(actFun){
		
	case AF_Sigmoid:
		for(int i=0; i<dim_output; i++)
			output[i] = Sigmoid(output[i]);
		break;

	case AF_Tanh:
		for(int i=0; i<dim_output; i++)
			output[i] = Tanh(output[i]);
		break;

	case AF_ReLU:
		for(int i=0; i<dim_output; i++)
			output[i] = ReLU(output[i]);
		break;

	case AF_Softmax:
		for(int i=0; i<dim_output; i++){
			output[i] = (float)exp(output[i]);

			if(!(output[i] >= -FLT_MAX && output[i] <= FLT_MAX)){
				printf("Overflow in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
				exit(-1);
			}
			sum += output[i];
		}

		if(sum > 0.F){		// for safety
			for(int i=0; i<dim_output; i++)
				output[i] /= (sum+0.000001);
		}
		break;

	}
	
}

float Layer::MSELoss(float *label){
	float loss = 0.0;
	for(int i=0; i<dim_output; i++){
		loss += 0.5*(output[i]-label[i])*(output[i]-label[i]);
	}
	loss /= (float)dim_output;

	return loss;
}

float Layer::CrossEntropyLoss(float *label){
	float loss = 0.0;
	for(int i=0; i<dim_output; i++){
		if(label[i] == 1)
			loss += -(float)log(output[i]);
	}

	return loss;
}

void Layer::dMSELoss(float *label){
	for(int i=0; i<dim_output; i++){
		d_loss[i] = (output[i]-label[i])/(float)dim_output;
	}
}

void Layer::dCrossEntropyLossdSoftmax(float *label){
	for(int i=0; i<dim_output; i++){
		d_loss[i] = -1.0/(output[i]+0.000001);
		d_actFun[i] = output[i]*(label[i]-output[i]);
	}
}

void Layer::BackPropagation(int actFun){
	dActivationFunction(actFun);

	for(int i=0; i<dim_output; i++)
		d_loss_actFun[i] = d_loss[i]*d_actFun[i];

	for(int i=0; i<dim_output; i++){
		for(int j=0; j<dim_input; j++)
			gradient[(dim_input+1)*i+j] += d_loss_actFun[i]*input[j];
		gradient[(dim_input+1)*i+dim_input] += d_loss_actFun[i];
	}
}

void Layer::BackPropagationSoftmax(){
	for(int i=0; i<dim_output; i++)
		d_loss_actFun[i] = d_loss[i]*d_actFun[i];

	for(int i=0; i<dim_output; i++){
		for(int j=0; j<dim_input; j++)
			gradient[(dim_input+1)*i+j] += d_loss_actFun[i]*input[j];
		gradient[(dim_input+1)*i+dim_input] += d_loss_actFun[i];
	}
}

void Layer::dActivationFunction(int actFun){
	switch(actFun){

	case AF_Sigmoid:
		for(int i=0; i<dim_output; i++)
			d_actFun[i] = output[i]*(1.0-output[i]);
		break;

	case AF_Tanh:
		for(int i=0; i<dim_output; i++)
			d_actFun[i] = (1.0-output[i])*(1.0+output[i]);
		break;

	case AF_ReLU:
		for(int i=0; i<dim_output; i++){
			if(output[i]!=0.0)	d_actFun[i] = 1.0;
			else			d_actFun[i] = 0.0;
		}
		break;
/*
	case AF_Softmax:
		for(int i=0; i<dim_output; i++)
			d_actFun[i] = d_actFun[i];
		break;

*/
	}
}

void Layer::dLoss(float *prev_d_loss){
	for(int i=0; i<dim_input; i++){
		prev_d_loss[i] = 0.F;
		for(int o=0; o<dim_output; o++)
			prev_d_loss[i] += weight[(dim_input+1)*o+i]*d_loss_actFun[o];
	}
}

void Layer::regularize(float learning_rate, float lambda){
	for(int o=0; o<dim_output; o++){
		for(int i=0; i<dim_input+1; i++){
			if(weight[o*(dim_input+1)+i] > (learning_rate*lambda))
				weight[o*(dim_input+1)+i] -= learning_rate*lambda;
			else if(weight[o*(dim_input+1)+i] < (-learning_rate*lambda))
				weight[o*(dim_input+1)+i] += learning_rate*lambda;
			else
				weight[o*(dim_input+1)+i] = 0;
		}
	}
}

void Layer::updateWeight(float learning_rate, float lambda, int reg){
	for(int o=0; o<dim_output; o++){
		for(int i=0; i<dim_input+1; i++){
			weight[o*(dim_input+1)+i] -= learning_rate*gradient[o*(dim_input+1)+i];

			gradient[o*(dim_input+1)+i] = 0.F;		// reset gradient
		}
	}
	if(reg==1)
		regularize(learning_rate, lambda);
}

void Layer::updateWeightMomentum(float learning_rate, float mu, float lambda, int reg){
	for(int o=0; o<dim_output; o++){
		for(int i=0; i<dim_input+1; i++){
			momentum_vector[o*(dim_input+1)+i] = mu*momentum_vector[o*(dim_input+1)+i] + learning_rate*gradient[o*(dim_input+1)+i];
			weight[o*(dim_input+1)+i] -= momentum_vector[o*(dim_input+1)+i];

			gradient[o*(dim_input+1)+i] = 0.F;		// reset gradient
		}
	}
	if(reg=1)
		regularize(learning_rate, lambda);
}


