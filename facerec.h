#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

using namespace cv;
using namespace std;
class FaceRec{
private:
    Ptr<FaceRecognizer> model;    
    
    void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator);
//    vector<int> labels;
    CascadeClassifier haar_cascade;
public: 
    FaceRec();  
    int train(string path);
    vector<int> predict(string path, vector<int>& labels);
}; 
