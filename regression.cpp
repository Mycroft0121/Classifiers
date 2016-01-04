#include<stdio.h>
#include<fstream>
#include<math.h>
#include<stdlib.h>
#include<iostream>
#include<string>
#include<string.h>
#include<algorithm>
#include<map>
#include<set>

using namespace std;

int objectCount = 0;
int dimensionCount = 0;
int trainCount = 0;
int testCount = 0;
int validationCount = 0;

int *objects;
int *dimensions;
float *frequencies;
int *trainSet;
int *testSet;
int *validationSet;
float *testObj;
int *Y;
float *AX;
float *optimiser;
float *predictionsV;
float *predictionsT;
float *predictionsO;
int *classT;
int validationLambda[20];

int classSet[21];
int maxDimension;
float precision = 0;
float recall = 0;
float F1 = 0;
int maxIterations = 40;
int maxTrainDimension = 0;
int numberOfClasses = 20;
float commonThreshold = 0;

map<int, int> termCount;
vector<string>classNames(20);
map<int, string> testResults;
map<int, string> actualTestClass;
map<int, string> classMap;
map<int, vector<int> >trainDimensionMap;
map<int, vector<float> >trainFrequencyMap;
vector<string> featureSpace;
map<int, int> rLabel;
//float **X;
float **X1;
float **X2;
float **X3;
float **X4;
float **X5;
float **X6;



// Count number of lines in file
int countLines(string filename){
	int count = 0;
	string line;
	ifstream myfile;
	myfile.open(filename);
	while(getline(myfile, line))
		count++;
	myfile.close();
	return count;
}


// Write cluster output to file
void writeOutput(string fileName){
	string line;
	ofstream myfile;
	myfile.open(fileName);
	map<int, string>::iterator it_testSet;
	
	if(myfile){
		for(it_testSet = testResults.begin(); it_testSet != testResults.end(); it_testSet++){	
			string temp = to_string(rLabel[it_testSet->first + 1]) + "," + (it_testSet->second);
			myfile << temp << endl;
		}
	}
	myfile.close();
}


// Get classID for a object
int getClassId(int objectID){

	int low = 0;
	int high = 20; 	
	while (low <= high){

		int mid = (low + high)/2;

		if (classSet[mid] == objectID)
			return mid;
		else if(classSet[mid] < objectID)
			low = mid;
		else
			high = mid;	
		if (high - low <= 1){
			if(classSet[low] > objectID)
				return low;
			else
				return high;
		}
	}
}


// Normalize frequencies of objects
void normalizeFrequencies(){
	
	for(int i = 0; i < objectCount; i++){
		float temp = 0;

		for(int j = ( i==0 ? 0 : objects[i-1] + 1); j <= objects[i]; j++){
			temp = temp + frequencies[j]*frequencies[j];
		}
		temp = sqrt(temp);			

		for(int j = ( i==0 ? 0 : objects[i-1] + 1); j <= objects[i]; j++){
			frequencies[j] = frequencies[j]/temp;
		}
	}
}

void allocateX(){

	int allocate = 0;
/*
        X = (float**)malloc(20*sizeof(float*));
        for(int i = 0; i < 20; i++){
                X[i] = (float*)malloc((maxDimension)*sizeof(float));
        }
        for(int i = 0; i < 20; i++)
                for (int j = 0; j < maxDimension; j++)
                        X[i][j] = allocate;
*/

	// X1
        X1 = (float**)malloc(20*sizeof(float*));
        for(int i = 0; i < 20; i++){
                X1[i] = (float*)malloc((maxDimension)*sizeof(float));
        }
        for(int i = 0; i < 20; i++)
                for (int j = 0; j < maxDimension; j++)
                        X1[i][j] = allocate;

	// X2
        X2 = (float**)malloc(20*sizeof(float*));
        for(int i = 0; i < 20; i++){
                X2[i] = (float*)malloc((maxDimension)*sizeof(float));
        }
        for(int i = 0; i < 20; i++)
                for (int j = 0; j < maxDimension; j++)
                        X2[i][j] = allocate;

	// X3
        X3 = (float**)malloc(20*sizeof(float*));
        for(int i = 0; i < 20; i++){
                X3[i] = (float*)malloc((maxDimension)*sizeof(float));
        }
        for(int i = 0; i < 20; i++)
                for (int j = 0; j < maxDimension; j++)
                        X3[i][j] = allocate;

	// X4
        X4 = (float**)malloc(20*sizeof(float*));
        for(int i = 0; i < 20; i++){
                X4[i] = (float*)malloc((maxDimension)*sizeof(float));
        }
        for(int i = 0; i < 20; i++)
                for (int j = 0; j < maxDimension; j++)
                        X4[i][j] = allocate;

	// X5
        X5 = (float**)malloc(20*sizeof(float*));
        for(int i = 0; i < 20; i++){
                X5[i] = (float*)malloc((maxDimension)*sizeof(float));
        }
        for(int i = 0; i < 20; i++)
                for (int j = 0; j < maxDimension; j++)
                        X5[i][j] = allocate;

	// X6
        X6 = (float**)malloc(20*sizeof(float*));
        for(int i = 0; i < 20; i++){
                X6[i] = (float*)malloc((maxDimension)*sizeof(float));
        }
        for(int i = 0; i < 20; i++)
                for (int j = 0; j < maxDimension; j++)
                        X6[i][j] = allocate;

}

// Find cosine Similarity between two objects
float cosine(float *centroid, int object){

	float tempNumerator = 0;

	for(int i = (object==0 ? 0 : objects[object-1]+1); i <= objects[object]; i++){
		tempNumerator = tempNumerator + frequencies[i]*centroid[dimensions[i]];
	}
	return tempNumerator;
}


void sortPredictionsV(){
        // Sort the predictions in decreasing order and the corresponding testSet
        for(int i = 0; i < validationCount; i++){
                for(int j = i; j < validationCount; j++){
                        if(predictionsV[j] > predictionsV[i]){
                                float temp = predictionsV[j];
                                predictionsV[j] = predictionsV[i];
                                predictionsV[i] = temp;

                                int temp1 = validationSet[j];
                                validationSet[j] = validationSet[i];
                                validationSet[i] = temp1;
                        }
                }
		//cout << predictionsV[i] << " " << validationSet[i] << endl;
        }
}


float evaluationV(int i){
	
	int threshold = 0;
	commonThreshold = 0;
	float f1 = 0;
	while (threshold < validationCount){
		int tp = 0;
		int tn = 0;
		int fn = 0;
		int fp = 0;
		for(int j = 0; j < validationCount; j++){
			int classID = getClassId(validationSet[j]);
			if (j <= threshold){
				if(classID  == i)
					tp++;
				else
					fp++;
			}
			else{
				if(classID == i)
					fn++;
				else
					tn++;
				}

		}
		float tempThreshold = predictionsV[threshold];
		float tempf1 = (2 * tp) / float((2 * tp) + fp + fn);
		float recall = tp / float(tp + fn);			
		float precision = tp / float(tp + fp);
		float F1_new = (2 * precision * recall) / (precision + recall);			
		if (f1 < tempf1){
			f1 = tempf1;
			commonThreshold = tempThreshold;
		}
//		cout << F1_new << " " << precision << " " << recall << " " << f1 << " " << tempf1 << " " << tp << " " << fp << " " << fn << " " << endl;
		threshold++;
		}
//		cout << i << " " << f1 << endl;
		return f1;
	
}

float validation (float **X, int classID, int objectID){
	float AX = 0;
	for(int i = (objectID == 0 ? 0 : objects[objectID-1]+1); i <= objects[objectID]; i++){
		AX += frequencies[i] * X[classID][dimensions[i]];
//		cout << frequencies[i] << " " << X[classID][dimensions[i]] << endl;
	}
	return AX;
}

float testingF (float **X, int classID, int objectID){
	float AX = 0;
	for(int i = (objectID == 0 ? 0 : objects[objectID-1]+1); i <= objects[objectID]; i++){
		AX += frequencies[i] * X[classID][dimensions[i]];
	}
	return AX;
}

void sortPredictionsT(){
        // Sort the predictions in decreasing order and the corresponding testSet
        for(int i = 0; i < testCount; i++){
                for(int j = i; j < testCount; j++){
                        if(predictionsT[j] > predictionsT[i]){
                                float temp = predictionsT[j];
                                predictionsT[j] = predictionsT[i];
                                predictionsT[i] = temp;

                                int temp1 = testSet[j];
                                testSet[j] = testSet[i];
                                testSet[i] = temp1;

				int temp2 = classT[j];
				classT[j] = classT[i];
				classT[i] = temp2;
                        }
                }
//		cout << predictionsT[i] << " " << testSet[i] << endl;
        }
}


float evaluationT(int classId, float thresholdT){
	
	int tp = 0;
	int tn = 0;
	int fn = 0;
	int fp = 0;
	for(int i = 0; i < testCount; i++){
		int classID = getClassId(testSet[i]);
		if(predictionsT[i] > thresholdT){
			if (classID == classId)
				tp++;
			else
				fp++;
		}
		else{
			if(classID == classId)
				fn++;
			else
				tn++;
		}
	}

	float recall = tp / float(tp + fn);			
	float precision = tp / float(tp + fp);
	float F1_new = (2 * precision * recall) / (precision + recall);			
	cout << classId << " " << F1_new << endl;

/*
	int threshold = 0;
	float f1 = 0;
	while (thresholdT < testCount){
		int tp = 0;
		int tn = 0;
		int fn = 0;
		int fp = 0;
		for(int j = 0; j < testCount; j++){
			int classID = getClassId(testSet[j]);
			if (j <= threshold){
				if(classID  == i)
					tp++;
				else
					fp++;
			}
			else{
				if(classID == i)
					fn++;
				else
					tn++;
				}

		}

		float tempf1 = (2 * tp) / float((2 * tp) + fp + fn);
		float recall = tp / float(tp + fn);			
		float precision = tp / float(tp + fp);
		float F1_new = (2 * precision * recall) / (precision + recall);			
		if (f1 < tempf1)
			f1 = tempf1;
//		cout << F1_new << " " << precision << " " << recall << " " << f1 << " " << tempf1 << " " << tp << " " << fp << " " << fn << " " << endl;
		threshold++;
		}
		cout << i << " " << f1 << endl;
		return f1;
*/
	
}

void regression(float lambda, float** X){

//	AX = (float*)malloc(sizeof(float)*trainCount);
//	memset(AX, 0, sizeof(AX[0])*trainCount);


	for(int i = 0; i < 20; i++){

		optimiser = (float*)malloc(sizeof(float)*objectCount);
		memset(optimiser, 0, sizeof(optimiser[0])*objectCount);

                Y = (int*)malloc((objectCount)*sizeof(int));
		memset(Y, 0, sizeof(Y[0])*objectCount);

        	for(int j = 0; j < trainCount; j++){                
			if(getClassId(trainSet[j]) == i)
                                Y[trainSet[j]] = 1;                      

                        for(int k = (trainSet[j]==0 ? 0 : objects[trainSet[j]-1]+1); k <= objects[trainSet[j]]; k++)
                              optimiser[trainSet[j]] += frequencies[k] * X[i][dimensions[k]];  
                }

//		for(int j = 0; j < trainCount; j++)
//			cout << optimiser[trainSet[j]] << endl;

		vector<int>::iterator it_v;
		vector<float>::iterator it_v2;
		map<int, vector<int> >::iterator it_m;	
		map<int, vector<float> >::iterator it_m2;

		float FX = 0;
//		maxIterations = 1;
                for(int j = 0; j < maxIterations; j++){    

/*
                        AX = (float*)malloc(sizeof(float)*trainCount);
                        memset(AX, 0, sizeof(AX[0])*trainCount);
                        for(int k = 0; k < trainCount; k++){
//                              cout << trainSet[k] << endl;
                                for(int m = (trainSet[k] == 0 ? 0 : objects[trainSet[k]-1]+1); m <= objects[trainSet[k]]; m++){
                                        AX[k] += frequencies[m]*X[i][dimensions[m]];
//                                      cout << AX[k] << " " << frequencies[m] << " " << X[i][dimensions[m]] << endl;
                                }
//                              cout << endl;

                        }
*/
                        float AXY = 0;
                        float XX = 0;
/*                        float FX = 0;
                        for(int k = 0; k < trainCount; k++){
                                AXY += (AX[k] - Y[trainSet[k]]) * (AX[k] - Y[trainSet[k]]);
                        }
			cout << "AXY = " << AXY << endl;
                        for(it_m = trainDimensionMap.begin(); it_m != trainDimensionMap.end(); it_m++)
                                XX += X[i][it_m->first] * X[i][it_m->first];

			cout << "XX = " << XX << endl;
//                      FX = sqrt(AXY) + (1)*sqrt(XX);
			FX = AXY + (1) * XX;
                        cout << i << " " << j << endl;
                        cout << "FX value = " << FX << endl;
*/
             		it_m2 = trainFrequencyMap.begin();
			for(it_m = trainDimensionMap.begin(); it_m != trainDimensionMap.end(); it_m++){
                                float num1 = 0;
                                float num2 = 0;
                                float den1 = 0;
				float newX = 0;
				
				it_v2 = it_m2->second.begin();
				for(it_v = it_m->second.begin(); it_v != it_m->second.end(); it_v++){
				/*	int dimension = 0;
					int flag = 0;
					for(int k = (*it_v == 0 ? 0 : objects[*it_v - 1] + 1); k <= objects[*it_v]; k++){
						if (dimensions[k] == it_m->first){
							dimension = k;	
							flag  = 1;
							break;
						}
					}
					if(flag != 1) 
						cout << "ERROR\n";
					cout << it_m->first << " " << it_m2->first << " " << *it_v2 << " " << frequencies[dimension] << endl;
					
					//cout << *it_s << " " << frequencies[dimension] << endl;
					num1 += Y[*it_v] * frequencies[dimension];
					num2 += frequencies[dimension] * (optimiser[*it_v] - frequencies[dimension] * X[i][it_m->first]);
					den1 += frequencies[dimension]*frequencies[dimension];	
				*/
					float frequency = *it_v2;
                                        num1 += Y[*it_v] * (frequency);
                                        num2 += (frequency) * (optimiser[*it_v] - frequency * X[i][it_m->first]);
                                        den1 += frequency * frequency;
						
					it_v2++;
				}
//				cout << it_m->first << " " << num1 << " " << num2 << " " << den1 << endl;
				newX = (num1 - num2) / (den1 + lambda);
//				cout << it_m->first << " " << newX << endl;
				newX = newX < 0 ? 0 : newX;

				it_v2 = it_m2->second.begin();
				for(it_v = it_m->second.begin(); it_v != it_m->second.end(); it_v++){
                                /*        int dimension = 0;
                                        int flag = 0;
                                        for(int k = (*it_v == 0 ? 0 : objects[*it_v - 1] + 1); k <= objects[*it_v]; k++){
                                                if (dimensions[k] == it_m->first){
                                                        dimension = k;
                                                        flag  = 1;
                                                        break;
                                                }
                                        }
					optimiser[*it_v] = optimiser[*it_v] - frequencies[dimension]*X[i][it_m->first] + frequencies[dimension]*newX;
				*/
					optimiser[*it_v] = optimiser[*it_v] - (*it_v2)*X[i][it_m->first] + (*it_v2)*newX;
					it_v2++;
				}
				it_m2++;
		
//		        	for(int j = 0; j < trainCount; j++)
//		                      cout << "\t" << trainSet[j] << " " << optimiser[trainSet[j]] << endl;
		
				X[i][it_m->first] = newX;
//				cout << endl;
//				break;
			}

		        AX = (float*)malloc(sizeof(float)*trainCount);
		        memset(AX, 0, sizeof(AX[0])*trainCount);
	                for(int k = 0; k < trainCount; k++){
//				cout << trainSet[k] << endl;
        	                for(int m = (trainSet[k] == 0 ? 0 : objects[trainSet[k]-1]+1); m <= objects[trainSet[k]]; m++){
                	                AX[k] += frequencies[m]*X[i][dimensions[m]];
//					cout << frequencies[m] << " " << X[i][dimensions[m]] << endl;
				}
//				cout << endl;

			}

			AXY = 0;
	                XX = 0; 
        	        float FX_new = 0;
	                for(int k = 0; k < trainCount; k++){
        	                AXY += (AX[k] - Y[trainSet[k]]) * (AX[k] - Y[trainSet[k]]);
        	        }
			for(it_m = trainDimensionMap.begin(); it_m != trainDimensionMap.end(); it_m++)
				XX += X[i][it_m->first] * X[i][it_m->first];       
			
//			cout << "AXY = " << AXY << endl;
//			cout << "XX = " << XX << endl;	
//                	FX = sqrt(AXY) + (1)*sqrt(XX);
			FX_new = AXY + (lambda)*XX;
			if (FX != 0 && FX - FX_new < 0.000001){
//				cout << FX << " " << FX_new << endl;
//				cout << j << endl;
				break;
			}
			else
				FX = FX_new;

//			cout << i << " " << j << endl;                                                                      
//	                cout << "FX value = " << FX << endl;
//			break;					
		}
//		cout << i << " " << FX << endl;
		free(Y);
		free(optimiser);
	}

}



// Main function
int main(int argc, char* argv[]){

	// Take input parameters

	if(argc < 3){
		cout << "Usage: classifier-name input-file input-rlabel-file train-file test-file class-file features-label-file feature-representation-option output-file [options]\n";
		return 1;
	}

	string inputFile = argv[1];
	string inputRlabelFile = argv[2];
	string trainFile = argv[3];
	string testFile = argv[4];
	string classFile = argv[5];
	string featureLabelFile = argv[6];
	string featureOption = argv[7];
	string outputFile = argv[8];
	string validationFile = argv[9];


	// Count objects, dimensions, traindata and testdata
	objectCount = countLines(inputRlabelFile);
	dimensionCount = countLines(inputFile);
	trainCount = countLines(trainFile);
	testCount = countLines(testFile);
	validationCount = countLines(validationFile);


	// Allocate Memory 
	objects = (int *)malloc(objectCount*sizeof(int));
	dimensions = (int *)malloc(dimensionCount*sizeof(int));
	frequencies = (float *)malloc(dimensionCount*sizeof(float));
	trainSet = (int *)malloc(trainCount*sizeof(int));
	testSet = (int *)malloc(testCount*sizeof(int));
	validationSet = (int *)malloc(validationCount*sizeof(int));


	// Create CSR
	string objectID, dimensionNO, frequency;
	int tempIndex = 0;
	ifstream myfile;
	myfile.open(inputFile);

	while(getline(myfile, objectID, ' ')){
		getline(myfile, dimensionNO, ' ');
		getline(myfile, frequency);
		
		objects[stoi(objectID)-1] = tempIndex;
		int dimension = stoi(dimensionNO) -1;
		dimensions[tempIndex] = dimension;


/*
		// if IDF then store count of each term in termCount(Map)		
		if((featureOption == "tfidf") || (featureOption == "binaryidf") || (featureOption == "sqrtidf")){
			map<int, int>::iterator it_m = termCount.find(dimension);
			
			if(it_m != termCount.end())
				termCount[dimension]++;
			else
				termCount[dimension] = 1;
		}
*/
		
		if(featureOption == "binary")
			frequencies[tempIndex] = 1;
		else if(featureOption == "sqrt")
			frequencies[tempIndex] = sqrt(stof(frequency));
		else 
			frequencies[tempIndex] = stof(frequency);

		tempIndex++;
	}
	myfile.close();

/*
	// if IDF then update frequencies
	if((featureOption == "tfidf") || (featureOption == "binaryidf") || (featureOption == "sqrtidf")){
		for(int i = 0; i < dimensionCount; i++){
			if(featureOption == "tfidf")
				frequencies[i] *= log2(objectCount / float(termCount[dimensions[i]]));
			else if (featureOption == "binaryidf")
				frequencies[i] = log2(objectCount / float(termCount[dimensions[i]]));
			else
				frequencies[i] = sqrt(frequencies[i]) * log2(objectCount / float(termCount[dimensions[i]]));

		}
	}	
*/
	// Store train data in trainSet(array)
	myfile.open(trainFile);
	tempIndex = 0;
	while(getline(myfile, objectID))
		trainSet[tempIndex++] = stoi(objectID) - 1;
	myfile.close();

	
	// Store validation data in validationSet(array)
        myfile.open(validationFile);
        tempIndex = 0;
        while(getline(myfile, objectID))
                validationSet[tempIndex++] = stoi(objectID) - 1;
        myfile.close();



	// Store test data in testSet(array)
	myfile.open(testFile);
	tempIndex = 0;
	while(getline(myfile, objectID))
		testSet[tempIndex++] = stoi(objectID) - 1;
	myfile.close();


	// Store ClassNames - names of all classes, ClassSet - number of elements in each class
	myfile.open(classFile);
	tempIndex = 1;
	int index = 0;
	string objectClass;
	string tempClass = "";	
	while(getline(myfile, objectID, ' ')){
		getline(myfile, objectClass);
		if (tempClass == ""){
			tempClass = objectClass;
			classSet[index] = tempIndex;
			classNames[index] = objectClass;
		}
		else if (tempClass == objectClass)
			classSet[index] = ++tempIndex;
		else{
			tempClass = objectClass;
			index++;
			classSet[index] = ++tempIndex;
			classNames[index] = objectClass;
		}
	}
	myfile.close();
	
	// Feature Space
	string word;
	myfile.open(featureLabelFile);
	while(getline(myfile, word))
		featureSpace.push_back(word);
	myfile.close();

	cout << featureSpace[0];

	//  Rlabel
	myfile.open(inputRlabelFile);
	string newId1;
	string newId2;
	string newWord;
	while(getline(myfile, newId1, ' ')){
		getline(myfile, newWord, ' ');
		getline(myfile, newId2);
		
		rLabel[stoi(newId1)] = stoi(newId2);
	}


	if((featureOption == "tfidf") || (featureOption == "binaryidf") || (featureOption == "sqrtidf")){

		for(int i = 0; i < trainCount; i++){
			for(int j = (trainSet[i] == 0 ? 0 : objects[trainSet[i]-1]+1); j <= objects[trainSet[i]]; j++)
                                	termCount[dimensions[j]]++;
		}
/*	
		map<int, int>::iterator it_m;
		int count = 0;
		for(it_m = termCount.begin(); it_m != termCount.end(); it_m++){
			cout << it_m->first << " " << it_m->second << endl;  
			count++;
		}

		cout << "Count" << count << endl;
*/
//        if((featureOption == "tfidf") || (featureOption == "binaryidf") || (featureOption == "sqrtidf")){
		for(int i = 0; i < objectCount; i++){
                	for(int j = (i == 0 ? 0 : objects[i-1]+1); j <= objects[i]; j++){
//				cout << j << " " << termCount[j] << endl;
				if (termCount[dimensions[j]] != 0){
	                        if(featureOption == "tfidf")
        	                        frequencies[j] *= log2(float(trainCount) / (termCount[dimensions[j]]));
                	        else if (featureOption == "binaryidf")
                        	        frequencies[j] = log2(trainCount / float(termCount[dimensions[j]]));
	                        else
        	                        frequencies[j] = sqrt(frequencies[j]) * log2(trainCount / float(termCount[dimensions[j]]));	
				}
				else
					frequencies[j] = 0;			
			}
		}
        }



	//Get maximum dimension in a document	
	maxDimension = *max_element(dimensions, dimensions+dimensionCount) + 1;	
	cout << maxDimension << endl;

	
	// Normalise documents
	normalizeFrequencies();

	// ++ Ridge Regression ++


	// Training begins

	for(int i = 0; i < trainCount; i++){
		for(int j = (trainSet[i] == 0 ? 0 : objects[trainSet[i]-1] + 1); j <= objects[trainSet[i]]; j++){
			trainDimensionMap[dimensions[j]].push_back(trainSet[i]);
//			cout << frequencies[j];
			trainFrequencyMap[dimensions[j]].push_back(frequencies[j]);
		}
	}

	float lambda[6] = {0.01, 0.05, 0.1, 0.5, 1, 10};	
	allocateX();

	cout << "Regression\n"; 
	regression(lambda[0], X1);
	regression(lambda[1], X2);
	regression(lambda[2], X3);
	regression(lambda[3], X4);
	regression(lambda[4], X5);
	regression(lambda[5], X6);

	// Training Ends ... Hussh learnt a lot !!!


	// Validation begins

	predictionsV = (float*)malloc(validationCount * sizeof(float));
	float thresholds[20];
	
	for(int i = 0; i < 20; i++){
		float F1 = 0;
		int classLambda = 0;
		float tempF1 = 0;


		// For lambda[0]
		for(int j = 0; j < validationCount; j++){
			predictionsV[j] = validation(X1, i, validationSet[j]);	
		}

//		cout << "Sorting for 0\n";
		sortPredictionsV();
		tempF1 = evaluationV(i);
		float threshold1 = commonThreshold;

		if(tempF1 > F1){
			F1 = tempF1;
			classLambda = 0;
			thresholds[i] = threshold1;
		}

//		break;
		// For lambda[1]
		for(int j = 0; j < validationCount; j++){
			predictionsV[j] = validation(X2, i, validationSet[j]);				
		}

//		cout << "Sorting for 1\n";
		sortPredictionsV();
		tempF1 = evaluationV(i);
		float threshold2 = commonThreshold;

		if(tempF1 > F1){
			F1 = tempF1;
			classLambda = 1;
			thresholds[i] = threshold2;
		}


		// For lambda[2]
		for(int j = 0; j < validationCount; j++){
			predictionsV[j] = validation(X3, i, validationSet[j]);				
		}

//		cout << "Sorting for 2\n";
		sortPredictionsV();
		tempF1 = evaluationV(i);
		float threshold3 = commonThreshold;

		if(tempF1 > F1){
			F1 = tempF1;
			classLambda = 2;
			thresholds[i] = threshold3;
		}


		// For lambda[3]		
		for(int j = 0; j < validationCount; j++){
			predictionsV[j] = validation(X4, i, validationSet[j]);				
		}

//		cout << "Sorting for 3\n";
		sortPredictionsV();
		tempF1 = evaluationV(i);
		float threshold4 = commonThreshold;

		if(tempF1 > F1){
			F1 = tempF1;
			classLambda = 3;
			thresholds[i] = threshold4;
		}


		// For lambda[4]
                for(int j = 0; j < validationCount; j++){
                        predictionsV[j] = validation(X5, i, validationSet[j]);
                }

//		cout << "Sorting for 4\n";
                sortPredictionsV();
                tempF1 = evaluationV(i);
		float threshold5 = commonThreshold;

                if(tempF1 > F1){
                        F1 = tempF1;
                        classLambda = 4;
			thresholds[i] = threshold5;
                }

		
		// For lambda[5]
		for(int j = 0; j < validationCount; j++){
			predictionsV[j] = validation(X6, i, validationSet[j]);				
		}

		sortPredictionsV();
		tempF1 = evaluationV(i);
		float threshold6 = commonThreshold;

		if(tempF1 > F1){
			F1 = tempF1;
			classLambda = 5;
			thresholds[i] = threshold6;		
		}
		cout << i << classLambda << " " << F1 << endl;
		validationLambda[i] = classLambda;
		
//		break;		
	}
	// Done with validation


	predictionsT = (float*)malloc(testCount * sizeof(float));
	predictionsO = (float*)malloc(testCount * sizeof(float));
	memset(predictionsO, 0, sizeof(predictionsO[0]) * testCount);
	classT = (int*)malloc(testCount * sizeof(int));
	float **X;

	cout << "Testing :\n";
	for(int i = 0; i < 20; i++){


		if(validationLambda[i] == 0)
			X = X1;
		else if (validationLambda[i] == 1)
			X = X2;
		else if (validationLambda[i] == 2)
			X = X3;
		else if (validationLambda[i] == 3)
			X = X4;
		else if (validationLambda[i] == 4)
			X = X5;
		else if (validationLambda[i] == 5)
			X = X6;


		for(int j = 0; j < testCount; j++){
			predictionsT[j] = testingF(X, i, testSet[j]);				

			if (predictionsO[j] < predictionsT[j]){
				classT[j] = i;
				predictionsO[j] = predictionsT[j];
			}
		}
//		cout << "Testing Results \n";
//		sortPredictionsT();
		cout<< "Class ";
		evaluationT(i, thresholds[i]);
//		break;
	}


//	string line;
	ofstream myfile1;
	myfile1.open("words.txt");
	for(int i = 0; i < 20; i++){

		if(validationLambda[i] == 0)
			X = X1;
		else if (validationLambda[i] == 1)
			X = X2;
		else if (validationLambda[i] == 2)
			X = X3;
		else if (validationLambda[i] == 3)
			X = X4;
		else if (validationLambda[i] == 4)
			X = X5;
		else if (validationLambda[i] == 5)
			X = X6;
		
		vector<pair<float, int> > top;
		for(int j = 0; j < maxDimension; j++)
			top.push_back(make_pair(X[i][j],j));
		sort(top.begin(), top.end());

		reverse(top.begin(),top.end());
		for(int j=0;j<10;j++){
//			cout << top[j].first << " " << top[j].second << endl;
			myfile1 << featureSpace[top[j].second] << " " << top[j].first << endl;
		}

		myfile1 << endl << endl;
	}
	myfile1.close();



	for(int i = 0; i < testCount; i++){
//		cout << testSet[i] << " " << classNames[classT[i]] << endl;
		testResults[testSet[i]] = classNames[classT[i]];
	}


	// Calculate Performance 
//	evaluation();

	
	// Write to output file	
	writeOutput(outputFile);
	
	
	// Free memory
	free(objects);
	free(dimensions);
	free(frequencies);
	free(trainSet);
	free(testSet);

//	free(X);
	free(X1);
	free(X2);
	free(X3);
	free(X4);
	free(X5);
	free(X6);

	return 0;
}
