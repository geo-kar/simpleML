#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\nonfree\nonfree.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\ml\ml.hpp>
#include <iostream>
#include "dirent.h"
#include <vector>
#include <string>


using namespace cv;
using namespace std;

std::vector<string> getFiles(char* folder) {
	vector<string> files;
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir(folder)) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			files.push_back(ent->d_name);
			printf("%s\n", ent->d_name);
		}
		closedir(dir);
	}
	else {
		/* could not open directory */
		perror("");
	}
	return files;
}

#define NumFolders 10
char* folders[NumFolders] = {
	"c:\\imagedb\\cannon",
	"c:\\imagedb\\chair",
	"c:\\imagedb\\crocodile",
	"c:\\imagedb\\elephant",
	"c:\\imagedb\\flamingo",
	"c:\\imagedb\\helicopter",
	"c:\\imagedb\\Motorbikes",
	"c:\\imagedb\\scissors",
	"c:\\imagedb\\strawberry",
	"c:\\imagedb\\sunflower"

};

char* testfolders[NumFolders] = {
	"c:\\imagedb_test\\cannon",
	"c:\\imagedb_test\\chair",
	"c:\\imagedb_test\\crocodile",
	"c:\\imagedb_test\\elephant",
	"c:\\imagedb_test\\flamingo",
	"c:\\imagedb_test\\helicopter",
	"c:\\imagedb_test\\Motorbikes",
	"c:\\imagedb_test\\scissors",
	"c:\\imagedb_test\\strawberry",
	"c:\\imagedb_test\\sunflower"

};

char clsNames[NumFolders][15] = {
	"cannon",
	"chair",
	"crocodile",
	"elephant",
	"flamingo",
	"helicopter",
	"Motorbikes",
	"scissors",
	"strawberry",
	"sunflower"
};

void CreateVocabulary(char** databasePath) {

	SiftFeatureDetector detector = SiftFeatureDetector();
	SiftDescriptorExtractor descriptor = SiftDescriptorExtractor();
	vector<KeyPoint> keypoints;
	BOWKMeansTrainer trainer(100);

	////////////////////////////////////////////////////////////////
	//
	// create vocabulary
	//
	////////////////////////////////////////////////////////////////

	for (int r = 0; r < NumFolders; r++) {
		string databasePathString_ = string(databasePath[r]);
		vector<string> files_ = getFiles(databasePath[r]);

		for (int i = 2; i < files_.size(); i++) {
			string imagePath = databasePathString_ + "\\" + files_[i];
			Mat image = imread(imagePath);
			detector.detect(image, keypoints);
			Mat descriptors;
			descriptor.compute(image, keypoints, descriptors);
			trainer.add(descriptors);
			keypoints.clear();
		}
	}
	///////////////////////////////////////////////////////////
	/// store file: vocab.xml
	///
	///////////////////////////////////////////////////////////
	cv::Mat vocabulary = trainer.cluster();
	cv::FileStorage file("vocab.xml", cv::FileStorage::WRITE);
	file << "vocab" << vocabulary;
	file.release();

	////////////////////////////////////////////////////////////////
}

void train(char** databasePath) {	

	SiftFeatureDetector detector = SiftFeatureDetector();
	SiftDescriptorExtractor descriptor = SiftDescriptorExtractor();
	vector<KeyPoint> keypoints;
	BOWKMeansTrainer trainer(128);

	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");
	cv::Ptr<cv::DescriptorExtractor> extractor = new cv::SiftDescriptorExtractor();
	cv::BOWImgDescriptorExtractor dextract(extractor, matcher);

	cv::Mat vocabulary;
	cv::FileStorage file("vocab.xml", cv::FileStorage::READ);
	file["vocab"] >> vocabulary;
	file.release();

	dextract.setVocabulary(vocabulary);

	for (int r = 0; r < NumFolders; r++) {
		Mat alldescs;
		Mat alllabels;

		vector<string> files;
		string databasePathString;

		for (int k = 0; k < NumFolders; k++) {
			databasePathString = string(databasePath[k]);
			files = getFiles(databasePath[k]);

			for (int i = 2; i < files.size(); i++) {
				string imagePath = databasePathString + "\\" + files[i];
				Mat image = imread(imagePath);
				detector.detect(image, keypoints);
				Mat descriptors;
				descriptor.compute(image, keypoints, descriptors);
				cv::Mat desc;
				dextract.compute(image, keypoints, desc);
				alldescs.push_back(desc);
				if (k == r)
					alllabels.push_back(1);
				else
					alllabels.push_back(0);
				keypoints.clear();
			}
		}		

		CvSVM svm;
		CvSVMParams params;
		//SVM type is defined as n-class classification n>=2, allows imperfect separation of classes
		params.svm_type = CvSVM::C_SVC;
		// No mapping is done, linear discrimination (or regression) is done in the original feature space.
		params.kernel_type = CvSVM::LINEAR;
		//Define the termination criterion for SVM algorithm.
		//Here stop the algorithm after the achieved algorithm-dependent accuracy becomes lower than epsilon
		//or run for maximum 100 iterations
		params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

		svm.train_auto(alldescs, alllabels, Mat(), Mat(), params);
		char SVM[15];
		sprintf(SVM, "%s%d%s", "svm", r, ".xml");
		svm.save(SVM);
	}
}

////// test function NOT!! in use///////////
// test(string path) {
//	Mat test_img = imread(path);
//
//	cv::Mat vocabulary;
//	cv::FileStorage file("vocab.xml", cv::FileStorage::READ);
//	file["vocab"] >> vocabulary;
//	file.release();
//
//	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");
//	SiftFeatureDetector detector = SiftFeatureDetector();
//	cv::Ptr<cv::DescriptorExtractor> descriptor = new cv::SiftDescriptorExtractor();
//	cv::BOWImgDescriptorExtractor dextract(descriptor, matcher);
//	dextract.setVocabulary(vocabulary);
//
//	vector<KeyPoint> keypoints;
//	detector.detect(test_img, keypoints);
//	Mat descriptors;
//	descriptor->compute(test_img, keypoints, descriptors);
//	cv::Mat desc;
//	dextract.compute(test_img, keypoints, desc);
//
//	CvSVM svm;
//	float prediction[NumFolders];
//	char svmname[15];
//	int cls;
//	float Min = 100;
//	for (int i = 0; i < NumFolders; i++) {
//		sprintf(svmname, "%s%d%s", "svm", i, ".xml");
//		svm.load(svmname);
//		prediction[i] = svm.predict(desc, true);
//		cout << prediction[i] << endl;
//		if (Min > abs(1 - prediction[i])) {
//			Min = abs(1 - prediction[i]);
//			cls = i;
//		}
//	}
//	//cout << clsNames[cls] << endl;
//	return cls;
//
//
//	//svm.load("svm0.xml");
//	
//
///*	float prediction = svm.predict(desc, true);
//	cout << prediction << endl;
//
//	svm.load("svm1.xml");
//	prediction = svm.predict(desc, true);
//	cout << prediction << endl;
//
//	svm.load("svm2.xml");
//	prediction = svm.predict(desc, true);
//	cout << prediction << endl;*/	
//
//	//svm.load("svm3.xml");
//	//prediction = svm.predict(desc, true);
//	//cout << prediction << endl;
//
//	//svm.load("svm4.xml");
//	//prediction = svm.predict(desc, true);
//	//cout << prediction << endl;
//
//	//svm.load("svm5.xml");
//	//prediction = svm.predict(desc, true);
//	//cout << prediction << endl;
//
//	//svm.load("svm6.xml");
//	//prediction = svm.predict(desc, true);
//	//cout << prediction << endl;
//
//	//svm.load("svm7.xml");
//	//prediction = svm.predict(desc, true);
//	//cout << prediction << endl;
//
//	//svm.load("svm8.xml");
//	//prediction = svm.predict(desc, true);
//	//cout << prediction << endl;
//
//	//svm.load("svm9.xml");
//	//prediction = svm.predict(desc, true);
//	//cout << prediction << endl;
//
//	//svm.load("svm10.xml");
//	//prediction = svm.predict(desc, true);
//	//cout << prediction << endl;
//
//	//system("PAUSE");
//}

int main(int argc, char** argv) {

	////////// put comments on the other to create vocabulary////////////////
	//CreateVocabulary(folders);
	////////////////////////////////////////////////////////////////////////
    //////// same for train/////////////////////////////////////////////
	//train(folders);	
	//////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////
	// testing, put comments before createVocabulary or train/////////
	///////////////////////.//////////////////////////////////////////


	string databasePathString;
	SiftFeatureDetector detector = SiftFeatureDetector();

	cv::Mat vocabulary;
	cv::FileStorage file("vocab.xml", cv::FileStorage::READ);
	file["vocab"] >> vocabulary;
	file.release();

	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");

	cv::Ptr<cv::DescriptorExtractor> descriptor = new cv::SiftDescriptorExtractor();
	cv::BOWImgDescriptorExtractor dextract(descriptor, matcher);
	dextract.setVocabulary(vocabulary);
	vector<KeyPoint> keypoints;

	int sumcounter = 0;  /// counts the number of test files
	int sucscounter = 0; //counts the the number of succesfuls tests
	for (int a = 0 ; a < NumFolders; a++) {
		databasePathString = string(testfolders[a]);
		vector<string> files1 = getFiles(testfolders[a]);
		cout << files1.size() -2 << endl;
		//system("pause");
		for (int j = 2; j< files1.size();j++) {
			string path = databasePathString + "\\" + files1[j];
			//int clas = test(path);

			Mat test_img = imread(path);

			
			detector.detect(test_img, keypoints);
			Mat descriptors;
			descriptor->compute(test_img, keypoints, descriptors);
			cv::Mat desc;
			dextract.compute(test_img, keypoints, desc);

			CvSVM svm;
			float prediction[NumFolders];
			char svmname[15];
			int cls;
			float Min = 100;
			for (int i = 0; i < NumFolders; i++) {
				sprintf(svmname, "%s%d%s", "svm", i, ".xml");
				svm.load(svmname);
				prediction[i] = svm.predict(desc, true);
				//cout << prediction[i] << endl;
				if (Min ==  1) {
					Min = abs(prediction[i]);
					cls = i;
				}
			}



			sumcounter = sumcounter + 1;
			if (cls == a){
				sucscounter = sucscounter + 1;
			}

			keypoints.clear();
		}
	}
	float succrate = ((float)sucscounter / (float)sumcounter )* 100.0f;
	cout << "Success rate = " << succrate << "%" << endl;

	


	system("pause");
	return 0;
	

}
