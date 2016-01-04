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
float *testObj;
float *trainObj;
float *predictions;
int *testClass;

int classSet[21];
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
	
	for(int i = 0; i < 20; i++){
//		cout << i << endl;
//		cout << tp[i] << " " << fp[i] << endl;
//		cout << fn[i] << " " << tn[i] << endl;
//		cout << endl;

		recall = tp[i]/float(tp[i] + fn[i]);
		precision = tp[i]/float(tp[i] + fp[i]);
		float sum =  (2 * precision * recall) / (precision + recall);
		F1 += sum;
		cout << "F1 = " << sum << endl;
 
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


// Find cosine Similarity between two objects
float cosine(float *centroid, int object){

	float tempNumerator = 0;

	for(int i = (object==0 ? 0 : objects[object-1]+1); i <= objects[object]; i++){
		tempNumerator = tempNumerator + frequencies[i]*centroid[dimensions[i]];
	}
	return tempNumerator;
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

				int temp2 = testClass[j];
				testClass[j] = testClass[i];
				testClass[i] = temp2;
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


// Find nearest neighbour of an object
float nearestneighbor(int objectID, int K, int classID){
	
	float similarity = 0;
	map<float, int> neighbors;
	int neighbor[trainCount];
	float similarityN[trainCount];

	for(int i = 0; i < trainCount; i++){
		similarity = cosine(testObj, trainSet[i]);                		
		similarityN[i] = similarity;
//		neighbors[similarity] = trainSet[i];
		neighbor[i] = 1;					
	}


	float sum[20] = {0};
	int count[20] = {0};

	for(int i = 0; i < K; i++){
		const int N = sizeof(similarityN)/sizeof(int);
		float max = *max_element(similarityN, (similarityN+N));
	        int index = distance(similarityN, max_element(similarityN, (similarityN+N)));
		int classID = getClassId(trainSet[index]);
//		cout << max << " " << index << endl;
		count[classID]++;
		sum[classID] += max;
		similarityN[index] = 0;
	}

	float PSimilarity = sum[classID];
	float NSimilarity = 0; 
	for(int i = 0; i < 20; i++){
		if (i != classID)
			NSimilarity += sum[i];
	}
	
	return (PSimilarity - NSimilarity);
/*	
	for(int i = 0; i < 20; i++)
		cout << count[i] << " " ;
	cout << endl;

	for(int i = 0; i < 20; i++)
		cout << sum[i] << " ";
	cout << endl;
*/

/*
	int max = *max_element(count, (count+20));
	int index = distance(count, max_element(count, (count+20)));

	if (max <= K/2){
		count[index] = 0;
		int max2 = *max_element(count, (count+20));
		int index2 = distance(count, max_element(count, (count+20)));
//		cout << max2 << " " << index2 << " " << sum[index2] << " Yo\n"; 
		while (max2 == max){
//			cout << max2 << endl;
			if (sum[index] < sum[index2]){
				max = max2;
				index = index2;
			}
				count[index2] = 0;
		                max2 = *max_element(count, (count+20));
		                index2 = distance(count, max_element(count, (count+20)));
		}
	}

	float kSimilarity[20];
	float pSimilarity = 0;
	float nSimilarity = 0;
	for(int i = 0; i < 20; i++){
		pSimilarity = sum[i];
		nSimilarity = 0;
		for(int j = 0; j < 20; j++)
			if (j != i)
				nSimilarity += sum[j];
		kSimilarity[i] = pSimilarity - nSimilarity;
		
	}
//	index = distance(kSimilarity, max_element(kSimilarity, (kSimilarity + 20)));

//	cout << index << endl;
	testResults[objectID] = classNames[index];

//	cout << objectID << " " << index << " " << sum[index] << endl;
*/
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
	int options = stoi(argv[9]);


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


		if(featureOption == "binary")
			frequencies[tempIndex] = 1;
		else if(featureOption == "sqrt")
			frequencies[tempIndex] = sqrt(stof(frequency));
		else 
			frequencies[tempIndex] = stof(frequency);

		tempIndex++;
	}
	myfile.close();
	

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


	// KNN
	testObj = (float*)malloc(maxDimension * sizeof(float));
	trainObj = (float*)malloc(maxDimension * sizeof(float));
	predictions = (float*)malloc(testCount * sizeof(float));
	testClass = (int*)malloc(testCount *sizeof(int));
	memset(testClass, 0, sizeof(testClass[0])*testCount);
	memset(predictions, 0, sizeof(predictions[0]) * testCount);


	for(int k = 0; k < 20; k++){	
		for(int i = 0; i < testCount; i++){
		
        	        memset(testObj, 0, sizeof(testObj[0])*maxDimension);
			for(int j = (objects[testSet[i]] == 0 ? 0 : objects[testSet[i]-1]+1); j <= objects[testSet[i]]; j++)
				testObj[dimensions[j]] = frequencies[j];

			float tempPred = predictions[i];
			predictions[i] = nearestneighbor(testSet[i], options, k); 
			if(predictions[i] > tempPred)
				testClass[i] = k;
//			break;
		}
		sortPredictions();
		evaluation1(k);
//		break;
	}

	for(int i = 0; i < testCount; i++){
		testResults[testSet[i]] = classNames[testClass[i]];
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
	free(testObj);
	free(trainObj);

	return 0;
}
