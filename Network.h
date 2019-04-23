#include <iostream>
#include <string>

using namespace std;

class Network{
	int n_layers;
	Layer **layers;

public:
	Network();
	void Alloc(int _n_layers, int *nodes);
	void ForwardNetwork(float *_input);
	float LossNetwork(float *label);
	void BackPropagationNetwork(float *label);
	void updateWeightNetwork(float learning_rate, float lambda);
	void updateWeightMomentumNetwork(float learning_rate, float mu, float lambda);

	float* getOutputNetwork()	{return layers[n_layers-1]->getOutput();}
};
