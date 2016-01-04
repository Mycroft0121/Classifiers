#include<stdio.h>
#include<fstream>
#include<math.h>
#include<stdlib.h>
#include<iostream>
#include<string>
#include<string.h>
#include<algorithm>
#include<map>


using namespace std;

int objectCount = 0;
int dimensionCount = 0;
int trainCount = 0;
int testCount = 0;
int *objects;
int *dimensions;
float *frequencies;
int *trainSet;
int *testSet;
int classSet[21];
float **centroids;
float **badCentroids;
float *predictions;
int maxDimension;
float precision = 0;
float recall = 0;
float F1 = 0;

map<int, int> termCount;
vector<string>classNames(20);
map<int, string> testResults;
map<int, string> actualTestClass;
map<int, string> classMap;
map<int, int> rLabel;



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


// Normalize centroid
void normalizeCentroids(float *centroid){

	float sumOfDimensions = 0;
	for (int j = 0; j < maxDimension; j++)
		sumOfDimensions += centroid[j] * centroid[j];

	sumOfDimensions = sqrt(sumOfDimensions);
	for(int j = 0; j < maxDimension; j++)
		centroid[j] /= sumOfDimensions;
}


// Calculate cosine similarity between object and centroid
float cosine(float *centroid, int object){

	float tempNumerator = 0;

	for(int i = (object==0 ? 0 : objects[object-1]+1); i <= objects[object]; i++){
		tempNumerator = tempNumerator + frequencies[i]*centroid[dimensions[i]];
	}
	return tempNumerator;
}


// Evaluate the performance of algorithm - Precision, Recall and F1 
void evaluation(){
	int tp[20] = {0};
	int tn[20] = {0};
	int fp[20] = {0};
	int fn[20] = {0};

	for(int i = 0; i < testCount ; i++){
		int classID = getClassId(testSet[i]);
		actualTestClass[testSet[i]] = classNames[classID];

		if (testResults[testSet[i]] == actualTestClass[testSet[i]]){
			tp[classID]++;
			tn[classID]--;
			for(int j = 0; j < 20; j++)
				tn[j]++;
		}
		else{
			fp[classID]++;
			vector<string>::iterator it_v;
			it_v = find(classNames.begin(), classNames.end(), testResults[testSet[i]]);
			int pos = distance(classNames.begin(), it_v);
			fn[pos]++;
			tn[classID]--;
			tn[pos]--;
			for(int j = 0; j < 20; j++)
				tn[j]++;
		}		

	}

	float F1 = 0;

	for(int i = 0; i < 20; i++){
		cout << i << endl;
		cout << tp[i] << " " << fp[i] << endl;
		cout << fn[i] << " " << tn[i] << endl;
//		cout << endl;

		recall = tp[i]/float(tp[i] + fn[i]);
		precision = tp[i]/float(tp[i] + fp[i]);

		float F1_new = (2 * precision * recall) / (precision + recall);
		F1 += F1_new; 
		cout << " F1 - " << F1_new << endl;
//	        cout << "Recall = "<< recall << endl;
//        	cout << "Precision = " << precision << endl;

	}
//	recall /= 20;
//	precision /= 20;
//	F1 = (2 * precision * recall) / (precision + recall);
	cout << "Recall = "<< recall << endl;
	cout << "Precision = " << precision << endl;
	cout << "F1(+ve) = " << F1/20 << endl;		
}

void sortPredictions(){
        // Sort the predictions in decreasing order and the corresponding testSet
        for(int i = 0; i < testCount; i++){
                for(int j = i; j < testCount; j++){
                        if(predictions[j] > predictions[i]){
                                float temp = predictions[j];
                                predictions[j] = predictions[i];
                                predictions[i] = temp;

                                int temp1 = testSet[j];
                                testSet[j] = testSet[i];
                                testSet[i] = temp1;
                        }
                }
//		cout << predictions[i] << " " << testSet[i] << endl;
        }
}


void evaluation1(int i){
	
	int threshold = 0;
	float f1 = 0;
	while (threshold < testCount){
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
/*	string options = "";
	if (argv[9])
		options = argv[9];
*/


	// Count objects, dimensions, traindata and testdata
	objectCount = countLines(inputRlabelFile);
	dimensionCount = countLines(inputFile);
	trainCount = countLines(trainFile);
	testCount = countLines(testFile);


	// Allocate Memory 
	objects = (int *)malloc(objectCount*sizeof(int));
	dimensions = (int *)malloc(dimensionCount*sizeof(int));
	frequencies = (float *)malloc(dimensionCount*sizeof(float));
	trainSet = (int *)malloc(trainCount*sizeof(int));
	testSet = (int *)malloc(testCount*sizeof(int));


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

//	cout << index << "\t" << tempIndex << endl;
//	for(int i = 0; i <= index; i++)
//		cout << i << " " << classSet[i] << endl;

//	for (int i = 0; i < trainCount; i++)
//		cout << trainSet[i] << " ";
//	cout << endl;

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

	
	// Normalise documents
	normalizeFrequencies();


	// Allocate memory to centroids and initialise them to 0
        centroids = (float**)malloc(20*sizeof(float*));
        for(int i = 0; i < 20; i++){
                centroids[i] = (float*)malloc((maxDimension)*sizeof(float));
        }
	for(int i = 0; i < 20; i++)
		for (int j = 0; j < maxDimension; j++)
			centroids[i][j] = 0;
	
        badCentroids = (float**)malloc(20*sizeof(float*));
        for(int i = 0; i < 20; i++){
                badCentroids[i] = (float*)malloc((maxDimension)*sizeof(float));
        }
	for(int i = 0; i < 20; i++)
		for (int j = 0; j < maxDimension; j++)
			badCentroids[i][j] = 0;


	// Update Centroids for each of the classes
	int classCount[20] = {0};
	for(int i = 0; i < trainCount; i++){
		int centroid = getClassId(trainSet[i]);
		
		for(int j = (trainSet[i] == 0 ? 0 : objects[trainSet[i]-1]+1); j <= objects[trainSet[i]]; j++){
			centroids[centroid][dimensions[j]] += frequencies[j];

			for(int k = 0; k < 20; k++){
				if (centroid != k){
					badCentroids[k][dimensions[j]] += frequencies[j];
				}
			}
			
		}
		classCount[centroid]++;
	}	

	for(int i = 0; i < 20; i++){
		float sum = 0;
		for(int j = 0; j < maxDimension; j++){
			sum += centroids[i][j];
			centroids[i][j] /= float(classCount[i]); 
			badCentroids[i][j] /= float(trainCount - classCount[i]);
		}
		normalizeCentroids(centroids[i]);
		normalizeCentroids(badCentroids[i]);		
	}


//	cout <<"I am here";	
	predictions = (float*)malloc(testCount*sizeof(float));
	// For test documents, calculate similarity with centroids
	for(int i = 0; i < testCount; i++){
		float oldSimilarity = -1;
		int assignedCentroid = 0;
		for(int j = 0; j < 20; j++){
			float newSimilarity1 = cosine(centroids[j], testSet[i]);
			float newSimilarity2 = cosine(badCentroids[j], testSet[i]);
//			cout << newSimilarity1 << " " << newSimilarity2 << " "  << newSimilarity1 - newSimilarity2 << " "  << testSet[i] << " "  << j << endl;
			if ((newSimilarity1 - newSimilarity2) > oldSimilarity){
				oldSimilarity = newSimilarity1 - newSimilarity2;
				assignedCentroid = j;	
			}
		}
//		cout << testSet[i] << " " << oldSimilarity << endl;
//		predictions[i] = oldSimilarity;
		testResults[testSet[i]] = classNames[assignedCentroid];
//		break;	
	}

	
	for(int i = 0; i < 20; i++){
		for(int j = 0; j < testCount; j++){
			predictions[j] = cosine(centroids[i], testSet[j]) - cosine(badCentroids[i], testSet[j]);
//			cout << predictions[j] << " " << testSet[j] << endl;
		}
		
		sortPredictions();
		evaluation1(i);
//		cout << predictions[0] << " " << testSet[0] << "\n" << predictions[1] << " " <<  testSet[1] << endl;
//		break;
	}	
	
/*
	cout << testSet[0] << " Done";
	// Sort the predictions in decreasing order and the corresponding testSet
	for(int i = 0; i < testCount; i++){
		for(int j = 1; j < testCount; j++){
			if(predictions[j] > predictions[i]){
				float temp = predictions[j];
				predictions[j] = predictions[i];
				predictions[i] = temp;

				int temp1 = testSet[j];
				testSet[j] = testSet[i];
				testSet[i] = temp1;
			}
		}
	}

	cout << predictions[0] <<" " << testSet[0] << endl;
*/
	// Calculate Performance 
//	evaluation1();

	
	// Write to output file	
	writeOutput(outputFile);
	
	
	// Free memory
	free(objects);
	free(dimensions);
	free(frequencies);

	for(int i = 0; i < 20; i++)
		free(centroids[i]);

	free(centroids);
//	free(predictions);
	free(trainSet);
	free(testSet);

	return 0;
}
	
