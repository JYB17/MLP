#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <fstream>
#include <ctime>

#include "Layer.h"
#include "Network.h"

void testXOR();
int reverseInt(int n);
void readMNIST_images(string filename, float **images);
void readMNIST_labels(string filename, float **labels);
int checkRight(float *label, float *output);
void testMNIST();

int main(){

	srand((unsigned int)time(NULL));

//	testXOR();

	testMNIST();

	return 0;
}

void testXOR(){
	float inputs[4][2] = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
	float labels[] = {0, 1, 1, 0};

	int n_inputs = 4;

	int nodes[] = {2, 4, 1};

	srand((unsigned int)time(NULL));

	Network net;
	net.Alloc(2, nodes);

	int epochs = 1000000;
	float learning_rate = 0.1;
	float lambda = 0.0005;

	cout << "Training begins... \n";

	for(int i=0; i<epochs; i++){
		float loss = 0.0;
		for(int j=0; j<n_inputs; j++){
			net.ForwardNetwork(inputs[j]);
			loss = net.LossNetwork(&labels[j]);
			net.BackPropagationNetwork(&labels[j]);
		}
		net.updateWeightNetwork(learning_rate, lambda);

		if(i==0 || (i+1)%10000 == 0){
			cout << "\n*********************************\n";
			cout << i+1 << "th epochs: \n";
			cout << " - loss: " << loss << "\n";
			cout << "*********************************\n";
		}
	}
	cout << "Training done! \n\n";

	for(int i=0; i<n_inputs; i++){
		net.ForwardNetwork(inputs[i]);

		float *output = net.getOutputNetwork();

		cout << inputs[i][0] << " " << inputs[i][1] << "   " << output[0] << "\n";
	}
}

int reverseInt(int n){
	unsigned char c1, c2, c3, c4;

	c1 = n&255;
	c2 = (n>>8)&255;
	c3 = (n>>16)&255;
	c4 = (n>>24)&255;

	return ((int)c1<<24) + ((int)c2<<16) + ((int)c3<<8) + c4;
}

void readMNIST_images(string filename, float **images){
	ifstream infile(filename.c_str(), ios::binary);

	if (infile.is_open()){
		int n_magic=0;
		int n_images=0, n_rows=0, n_cols=0;

		infile.read((char*)&n_magic,sizeof(n_magic));
		n_magic= reverseInt(n_magic);

		if(n_magic!=2051)	throw runtime_error("Wrong magic number(images)!");

		infile.read((char*)&n_images,sizeof(n_images));
		n_images= reverseInt(n_images);

		infile.read((char*)&n_rows,sizeof(n_rows));
		n_rows= reverseInt(n_rows);

		infile.read((char*)&n_cols,sizeof(n_cols));
		n_cols= reverseInt(n_cols);

		cout << "magic: " << n_magic << ", number: " << n_images << ", rows: " << n_rows << ", cols: " << n_cols << "\n";

		for(int i=0;i<n_images;i++){
			int n_imageSize = n_rows*n_cols;
			for(int j=0; j<n_imageSize; j++){
				unsigned char temp=0;
				infile.read((char*)&temp,sizeof(temp));
				images[i][j]= (float)temp/255.0;
			}
		}
	}
	else{
		throw runtime_error("No file named \"" + filename +"\"!\n");
	}
}

void readMNIST_labels(string filename, float **labels){
	ifstream infile(filename.c_str(), ios::binary);

	if(infile.is_open()){
		int n_magic=0;
		int n_labels=0;

		infile.read((char*)&n_magic, sizeof(n_magic));
		n_magic = reverseInt(n_magic);

		if(n_magic!=2049)	throw runtime_error("Wrong magic number(labels)!");

		infile.read((char*)&n_labels, sizeof(n_labels));
		n_labels = reverseInt(n_labels);

		float *desired = new float[n_labels];
		assert(desired!=NULL);

		cout << "magic: " << n_magic << ", number: " << n_labels << "\n";

		for(int i=0; i<n_labels; i++){
			unsigned char temp=0;
			infile.read((char*)&temp, sizeof(temp));
			desired[i] = (float)temp;
		}

		int n_output = 10;

		for(int i=0; i<n_labels; i++){
			for(int j=0; j<n_output; j++){
				if(desired[i]==j)	labels[i][j] = 1.0;
				else	labels[i][j] = 0.0;
			}
		}
	}
	else{
		throw runtime_error("No file named \"" + filename +"\"!\n");
	}
}

int checkRight(float *label, float *output){
	float max = -1.0;
	int maxIndex = -1;
	int rightAnswer = -1;

	for(int i=0; i<10; i++){
		if(output[i]>=max){
			max = output[i];
			maxIndex = i;
		}
		if(label[i]==1.0){
			rightAnswer = i;
		}
	}
	
	if(rightAnswer==maxIndex) return 1;
	else return 0;
}

void testMNIST(){
	int nodes1[] = {784, 300, 100, 10};
	int nodes2[] = {784, 500, 150, 10};
	int n_trainvalid = 60000, n_test = 10000;
	int n_train = 55000, n_valid = 5000;
	int n_imageSize = 784;

	float **train_inputs = new float*[n_trainvalid];
	assert(train_inputs!=NULL);
	for(int i=0; i<n_trainvalid; i++){
		train_inputs[i] = new float[n_imageSize];
		assert(train_inputs[i]!=NULL);
	}

	float **train_labels = new float*[n_trainvalid];
	assert(train_labels!=NULL);
	for(int i=0; i<n_trainvalid; i++){
		train_labels[i] = new float[n_imageSize];
		assert(train_labels[i]!=NULL);
	}

	readMNIST_images("train-images-idx3-ubyte", train_inputs);
	readMNIST_labels("train-labels-idx1-ubyte", train_labels);

	srand((unsigned int)time(NULL));

	Network net;
	net.Alloc(3, nodes2);

	int steps = 20000;
	float learning_rate = 0.01;
	int batchsize = 20;
	float mu = 0.9;
	float lambda = 0.0005;

	int *index = new int[n_train];
	assert(index!=NULL);
	for(int i=0; i<n_train; i++){
		index[i] = i;
	}

	cout << "Training begins... \n\n";
	
	int k=0;
	clock_t start_time = clock();
	for(int i=0; i<steps; i++){
		float loss=0.0;
		for(int j=k; j<k+batchsize; j++){
			int m = index[j];
			net.ForwardNetwork(train_inputs[m]);
			loss = net.LossNetwork(train_labels[m]);
			net.BackPropagationNetwork(train_labels[m]);
		}
		net.updateWeightNetwork(learning_rate, lambda);
//		net.updateWeightMomentumNetwork(learning_rate, mu, lambda);

		int correct = 0;
		float valid_accuracy = 0.0;
		if(i==0 || (i+1)%1000 == 0){
			cout << "\n***************************** \n";
			cout << i+1 << "th step: \n";
			cout << " - loss: " << loss << "\n";
			for(int j=n_train; j<n_trainvalid; j++){
				net.ForwardNetwork(train_inputs[j]);
				float *output = net.getOutputNetwork();
				correct += checkRight(train_labels[j], output);
			}
			valid_accuracy = (float)correct/n_valid * 100.0;
			cout << " - valid accuracy: " << valid_accuracy << "% \n";
			cout << "***************************** \n";				
		}

		k += batchsize;
		if((k+batchsize)>n_train)	k=0;

		if(k==0){
			srand((unsigned int)time(NULL));
			for (int i=0; i<n_train; i++) {
				int temp = index[i];
				int p = rand() % n_train;
				index[i] = index[p];
				index[p] = temp;
			}	
		}
	}
	clock_t end_time = clock();

	float train_time = (float)(end_time-start_time)/CLOCKS_PER_SEC;
	int hours = (int)train_time/3600;
	int mins = (int)train_time%3600/60;
	int secs = (int)train_time%3600%60;
	
	cout << "\n\nTraining done!\n";

	cout << "\nTime spent for training: " << hours << " : " << mins << " : " << secs  << "\n\n";

	float **test_inputs = new float*[n_test];
	assert(test_inputs!=NULL);
	for(int i=0; i<n_test; i++){
		test_inputs[i] = new float[n_imageSize];
		assert(test_inputs[i]!=NULL);
	}	

	float **test_labels = new float*[n_test];
	assert(test_labels!=NULL);
	for(int i=0; i<n_test; i++){
		test_labels[i] = new float[n_imageSize];
		assert(test_labels[i]!=NULL);
	}
	
	readMNIST_images("t10k-images-idx3-ubyte", test_inputs);
	readMNIST_labels("t10k-labels-idx1-ubyte", test_labels);
	
	int correct = 0;
	for(int i=0; i<n_test; i++){
		net.ForwardNetwork(test_inputs[i]);
		float *output = net.getOutputNetwork();
		correct += checkRight(test_labels[i], output);
	}
	float test_accuracy = (float)correct/n_test *100.0;
	cout << "\nTest accuracy: " << test_accuracy << "% \n\n";
}

