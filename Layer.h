#include <iostream>
#include <string>

using namespace std;

class Layer{
	float *input;
	float *output;
	float *weight;
	float *gradient;
	float *d_loss;
	float *d_actFun;
	float *d_loss_actFun;

	float *momentum_vector;	

	int dim_input;
	int dim_output;
	int dim_weight;

public:
	enum activationFunction { AF_Sigmoid, AF_Tanh, AF_ReLU, AF_Softmax };

	Layer();
	void Alloc(int _dim_input, int _dim_output);
	float Sigmoid(float net);
	float Tanh(float net);
	float ReLU(float net);
	void Forward(float *_input, int actFun);
	void ApplyActivationFunction(int actFun);
	float MSELoss(float *label);
	float CrossEntropyLoss(float *label);
	void dMSELoss(float *label);
	void dCrossEntropyLossdSoftmax(float *label);
	void BackPropagation(int actFun);
	void BackPropagationSoftmax();
	void dActivationFunction(int actFun);
	void dLoss(float *prev_d_loss);
	void regularize(float learning_rate, float lambda);
	void updateWeight(float learning_rate, float lambda, int reg);
	void updateWeightMomentum(float learning_rate, float mu, float lambda, int reg);

	float* getInput()		{return input;}
	float* getOutput()		{return output;}
	float* getWeight()		{return weight;}
	float* getGradient()	{return gradient;}
	float* getdLoss()		{return d_loss;}
};
