#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>


using namespace std;
using namespace cv;
using namespace chrono;

void binarization(Mat, int);
Mat makeEvidence(Mat,Mat);
Mat makeXORdifference(Mat,Mat);

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Loading error!" << endl;
        return -1;
    }

    Mat img1 = imread(argv[1], IMREAD_COLOR);
    Mat display = img1.clone();
    //Mat toclone = imread(argv[1], IMREAD_COLOR);
    //Mat display = toclone.clone();
    Mat img2 = imread(argv[2], IMREAD_GRAYSCALE);
    int th = 150;

    if (img1.empty() || img2.empty()) {
        cerr << "Error in loading images!" << endl;
        return -1;
    }

    cvtColor(img1,img1,COLOR_BGR2GRAY);

    auto start_time = steady_clock::now();
    binarization(img1, th);
    binarization(img2, th);
    // Xor and evidence of the object
    Mat xor_result = makeXORdifference(img1, img2);
    Mat result = makeEvidence(xor_result, display);
    auto end_time = steady_clock::now();

    duration<double> elapsed_seconds = end_time - start_time;
    cout << "EXECUTION TIME: " << fixed << setprecision(6) << elapsed_seconds.count() << "s\n";

    if(imwrite("outputs/evidence_result_sequential.jpg", result) && imwrite("outputs/XOR_sequential.jpg",xor_result))
        cout << "\nImage outputs correctly saved!" << endl;
    else
        return -1;

    return 0;
}

void binarization(Mat src, int th){

    for (int i = 0; i < src.rows; i++){
        for (int j = 0; j < src.cols; j++){
                int pixel = src.at<uchar>(i, j);
                if (pixel >= th)
                    src.at<uchar>(i, j) = 255;
                else
                    src.at<uchar>(i, j) = 0;
                }
            }
}

Mat makeEvidence(Mat xor_result,Mat color_src){

    Mat evidence;
    cvtColor(xor_result, evidence, COLOR_GRAY2BGR);
    for (int i = 0; i < xor_result.rows; i++) {
        for (int j = 0; j < xor_result.cols; j++) {
            if (xor_result.at<uchar>(i, j) > 0) {
                evidence.at<Vec3b>(i, j) = color_src.at<Vec3b>(i, j);
            }
        }
    }

    return evidence;
}

Mat makeXORdifference(Mat img1,Mat img2){
  
    Mat xor_result = Mat::zeros(img1.size(), CV_8UC1);

    for (int i = 0; i < img1.rows; i++) {
        for (int j = 0; j < img1.cols; j++) {
            uchar pixel1 = img1.at<uchar>(i, j);
            uchar pixel2 = img2.at<uchar>(i, j);
            xor_result.at<uchar>(i, j) = pixel1 ^ pixel2;
        }
    }

    return xor_result;
}
