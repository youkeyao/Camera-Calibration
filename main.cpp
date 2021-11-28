#include <dirent.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Size board_size = Size(10, 4);
Size square_size = Size(1, 1);
string type = "normal";

string chessDir = "../chessboard/";
string imageDir = "../images/";
string videoDir = "../videos/";
string undistDir = "../undistort/";
string cornersDir = "../corners/";

void findFiles(string path, vector<string> &files) {
    struct dirent* d_ent = NULL;
    DIR* dir = opendir(path.c_str());
    while ((d_ent = readdir(dir)) != NULL) {
        if (strcmp(d_ent->d_name, "..") != 0 && strcmp(d_ent->d_name, ".") != 0)
            files.push_back(d_ent->d_name);
    }
}

int findCorners(cv::Mat &image, vector<vector<Point2f>> &corners_Seq, string imageFileName) {
    vector<Point2f> corners;
    Mat imageGray;
    cvtColor(image, imageGray, cv::COLOR_RGB2GRAY);
    bool isfound = findChessboardCorners(image, board_size, corners, cv::CALIB_CB_ADAPTIVE_THRESH|cv::CALIB_CB_NORMALIZE_IMAGE|cv::CALIB_CB_FAST_CHECK);
    if (!isfound) {
        return 0;
    }
    else {
        cornerSubPix(imageGray, corners, Size(5, 5), Size(-1, -1), TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.001));
        Mat imageTemp = image.clone();
        drawChessboardCorners(imageTemp, board_size, corners, 1);
        imwrite(cornersDir + imageFileName, imageTemp);
        corners_Seq.push_back(corners);
        return corners.size();
    }
}

void initObjPoints(int num, vector<vector<Point3f>> &object_Points) {
    for (int t = 0; t < num; t++) {
        vector<Point3f> tempPointSet;
        for (int i = 0; i < board_size.height; i++) {
            for (int j = 0; j < board_size.width; j++) {
                Point3f tempPoint;
                tempPoint.y = i * square_size.height;
                tempPoint.x = j * square_size.width;
                tempPoint.z = 0;
                tempPointSet.push_back(tempPoint);
            }
        }
        object_Points.push_back(tempPointSet);
    }
}

void calibrate_camera(vector<vector<Point3f>> &object_Points, vector<vector<Point2f>> &corners_Seq, Size image_size, Mat& intrinsic_matrix, Mat& distortion_coeffs, vector<cv::Vec3d> &rotation_vectors, vector<cv::Vec3d> &translation_vectors) {
    int flags = 0;
    flags |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
    flags |= cv::fisheye::CALIB_CHECK_COND;
    flags |= cv::fisheye::CALIB_FIX_SKEW;
    flags |= cv::fisheye::CALIB_FIX_K2 ;
    flags |= cv::fisheye::CALIB_FIX_K3 ;
    flags |= cv::fisheye::CALIB_FIX_K4;

    if (type == "normal") {
        calibrateCamera(object_Points, corners_Seq, image_size, intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors);
    }
    else if (type == "fisheye") {
        fisheye::calibrate(object_Points, corners_Seq, image_size, intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors, flags, cv::TermCriteria(3, 20, 1e-6));
    }
}

void storeResult(Mat& intrinsic_matrix, Mat& distortion_coeffs, vector<cv::Vec3d> &rotation_vectors, vector<cv::Vec3d> &translation_vectors) {
    cv::FileStorage fs("../intrinsics.yml", cv::FileStorage::WRITE);
    if (fs.isOpened()) {
        fs << "M1" << intrinsic_matrix << "D1" << distortion_coeffs;
        fs.release();
    }
    else
        cout << "Error: can not save the intrinsic parameters" << endl;
}

void undistort_image(float scale, vector<Mat> &image_Seq, vector<string> &chess_names, Mat& intrinsic_matrix, Mat& distortion_coeffs, vector<cv::Vec3d> &rotation_vectors, vector<cv::Vec3d> &translation_vectors, Mat &mapx, Mat &mapy) {
    Size image_size = image_Seq[0].size();
    mapx = Mat(image_size,CV_32FC1);
    mapy = Mat(image_size,CV_32FC1);
    cv::Mat intrinsic_matrix2 = intrinsic_matrix.clone();
    intrinsic_matrix2.at<double>(0,0) = intrinsic_matrix2.at<double>(0,0) / scale;
    intrinsic_matrix2.at<double>(1,1) = intrinsic_matrix2.at<double>(1,1) / scale;
    if (type == "normal") {
        initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, Mat(), intrinsic_matrix2, image_size, CV_16SC2, mapx, mapy);
    }
    else if (type == "fisheye") {
        fisheye::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, Mat::eye(3, 3, CV_32F), intrinsic_matrix2, image_size, CV_16SC2, mapx, mapy);
    }

    FileStorage fs;
    fs.open("../undistort.yml", cv::FileStorage::WRITE);
    if(fs.isOpened()) {
        fs << "Mapx" << mapx << "Mapy" << mapy << "New M1" << intrinsic_matrix2;
        fs.release();
    }
    else
        cout << "Error: can not save the remap parameters\n";
    for (int i = 0 ; i < chess_names.size(); i++) {
        cout << "Undistorting image " << chess_names[i] << "..." << endl;
        Mat t = image_Seq[i].clone();
        cv::remap(image_Seq[i], t, mapx, mapy, INTER_LINEAR, BORDER_CONSTANT);		
        imwrite(undistDir + chess_names[i], t);
    }
}

int main(int argv, char* argc[]) {
    for (int i = 0; i < argv; i ++) {
        if (strcmp(argc[i], "-m") == 0) {
            board_size.width = atoi(argc[++i]);
        }
        else if (strcmp(argc[i], "-n") == 0) {
            board_size.height = atoi(argc[++i]);
        }
        else if (strcmp(argc[i], "-type") == 0) {
            type = argc[++i];
            if (type != "normal" && type != "fisheye") {
                cout << "Unknown type!" << endl;
                return -1;
            }
        }
    }

    cv::Mat intrinsic_matrix;
    cv::Mat distortion_coeffs;
    vector<cv::Vec3d> rotation_vectors;
    vector<cv::Vec3d> translation_vectors;
    cv::Mat mapx;
    cv::Mat mapy;

    vector<vector<Point2f>> corners_Seq;
    vector<vector<Point3f>> object_Points;
    vector<Mat> image_Seq;
    vector<string> chess_names;
    vector<string> image_names;
    vector<string> video_names;

    cout << "Finding files..." << endl;
    findFiles(chessDir, chess_names);
    for(int i = 0; i < chess_names.size(); i++) {
        cv::Mat image = imread(chessDir + chess_names[i], 1);
        if (image.empty()) {
            cout << "can not find image " << chess_names[i] << endl;
            chess_names.erase(chess_names.begin()+i);
            i --;
            continue;
        }
        int corner_size = findCorners(image, corners_Seq, chess_names[i]);
        if (corner_size <= 0) {
            cout << "Image " << chess_names[i] << " can not find chessboard corners!" << endl;
            chess_names.erase(chess_names.begin()+i);
            i --;
            continue;
        }
        image_Seq.push_back(image);
    }

    if (image_Seq.size() == 0) {
        cout << "No chessboard image!" << endl;
        return -1;
    }
    else {
        initObjPoints(image_Seq.size(), object_Points);
    }

    cout << "Calibrating..." << endl;
    calibrate_camera(object_Points, corners_Seq, image_Seq[0].size(), intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors);

    cout << "Storing calibration results..." << endl;
    storeResult(intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors);

    undistort_image(1, image_Seq, chess_names, intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors, mapx, mapy);

    findFiles(imageDir, image_names);
    for(int i = 0; i < image_names.size(); i++) {
        cv::Mat image = imread(imageDir + image_names[i], 1);
        if (image.empty()) {
            cout << "can not find image " << image_names[i] << endl;
            continue;
        }
        cout << "Undistorting image " << image_names[i] << "..." << endl;
        Mat t = image.clone();
        cv::remap(image, t, mapx, mapy, INTER_LINEAR, BORDER_CONSTANT);		
        imwrite(undistDir + image_names[i], t);
    }

    findFiles(videoDir, video_names);
    for(int i = 0; i < video_names.size(); i++) {
        VideoCapture capture;
        capture.open(videoDir + video_names[i]);
        if (!capture.isOpened()) {
            cout << "cannot open video " << video_names[i] << endl;
        }

        int frame_len = capture.get(CAP_PROP_FRAME_COUNT), count = 0;
        Mat frame, t;
        Size size = Size(capture.get(CAP_PROP_FRAME_WIDTH), capture.get(CAP_PROP_FRAME_HEIGHT));
        VideoWriter writer;
        writer.open(undistDir + video_names[i], capture.get(CAP_PROP_FOURCC), capture.get(CAP_PROP_FPS), size, true);
        while (capture.read(frame)) {
            cout << "\rUndistorting video " << video_names[i] << "..." << (double)100 * count / frame_len << "%";
            cv::remap(frame, t, mapx, mapy, INTER_LINEAR, BORDER_CONSTANT);
            writer.write(t);
            count ++;
        }
        cout << endl;
        writer.release();
        capture.release();
    }
	return 0;
}
