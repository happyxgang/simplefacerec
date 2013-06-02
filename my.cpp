#include "./facerec.cpp"
#include <iostream>
#include <vector>
using namespace std;
int main(){
    FaceRec fr;
    fr.train("/home/kevin/workspace/opencv/csv.ext");
    vector<int> labels, predictLabels; 
    predictLabels = fr.predict("/home/kevin/workspace/opencv/test.ext", labels); 
    for(int i = 0; i < labels.size(); i++){
        cout << "Actual  result: " << labels[i] << endl;
        cout << "Predict result: " << predictLabels[i] << endl;    
    }
    return 0;
}
