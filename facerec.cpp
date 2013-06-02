#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "facerec.h"
#include <iostream>
#include <fstream>
#include <sstream>
using namespace cv;
using namespace std;

FaceRec::FaceRec()
{
    model = createLBPHFaceRecognizer();
    haar_cascade.load("/home/kevin/Downloads/opencv-2.4.5/data/haarcascades/haarcascade_frontalface_alt.xml");
}
void FaceRec::read_csv(const string& filename, vector<Mat>& images,
                            vector<int>& labels, char separator)
{
    std::ifstream file(filename.c_str(), ifstream::in);
    if(!file){
        string error_message = "No valid input file was given, please check the filename";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while(getline(file, line)){
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()){
            images.push_back(imread(path,0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

int FaceRec::train(string path)
{
    vector<Mat> trainImages;
    vector<int> labels;
    try{
        read_csv(path, trainImages, labels, ';');
    }catch(cv::Exception& e){
        cerr <<"Error opening file\"" << path << "\". reason: " << e.msg << endl;      
        exit(1);
    }
    if(trainImages.size() <=0){
        string error_message = "No pictures readed, please check your input director";
        CV_Error(CV_StsError,error_message);
        exit(1);
    }
    model->train(trainImages,labels);
    return 0;
}

vector<int>FaceRec::predict(string path,vector<int>& labels)
{
    vector<Mat> testImages;
    vector<int> predictlabels;
    labels.clear();
    try{
        // labels is precaculated and stored with the testImages
        read_csv(path, testImages, labels,';');
    }catch(cv::Exception& e){
        cerr <<"Error opening file\"" << path << "\". reason: " << e.msg << endl;      
        exit(1);
    }
    if(testImages.size() <=0){
        string error_message = "No pictures readed, please check your input director";
        CV_Error(CV_StsError,error_message);
        exit(1);
    }
    int label = 0;
    for(int i = 0; i < testImages.size(); i++){
        Mat original = testImages[i].clone();
        Mat gray;
        vector< Rect_<int> > faces;
        label = model->predict(testImages[i]);
        predictlabels.push_back(label);
    }
    return predictlabels;
}
