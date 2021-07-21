#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<iostream>

#include<vector>

using namespace cv;
using namespace std;

Mat image_size_double(Mat input_image);
int get_floor_data(Mat input_image);
Mat image_matching(Mat input_image1, Mat input_image2);
Mat create_floorless_map(Mat input_image, int threshold);

Mat line_detection(Mat input_image);

int main() {

	Mat src, global;
	src = imread("maps/3.5m/3.5m 2.pgm", IMREAD_GRAYSCALE);
	global = imread("maps/global_map.pgm", IMREAD_GRAYSCALE);

	// Get Floor data to extract from the map.
	int global_floor_data = get_floor_data(global);
	int src_floor_data = get_floor_data(src);

	// Create new map image without floor.
	Mat src_floorless, global_floorless;
	src_floorless = create_floorless_map(src, src_floor_data);
	global_floorless = create_floorless_map(global, global_floor_data + 10);
	
	Mat global_line, src_line;
	global_line = line_detection(global_floorless);
	src_line = line_detection(src_floorless);

	imshow("global", global_line);
	imshow("src", src_line);

	waitKey(0);
	return 0;
}


Mat image_size_double(Mat input_image) {
	int input_image_rows = input_image.rows;
	int input_image_cols = input_image.cols;
	Mat double_result = Mat::zeros(input_image_rows * 2, input_image_cols * 2, CV_8UC1);
	for(int j=0; j<input_image_rows; j++)
		for (int i = 0; i < input_image_cols; i++) {
			double_result.at<uchar>(j + (input_image_rows / 2),
				i + (input_image_cols / 2)) = input_image.at<uchar>(j, i);
		}
	return double_result;
}
int get_floor_data(Mat input_image) {
	Mat img_hist;
	MatND histogram;
	const int* channel_numbers = { 0 };
	float channel_range[] = { 0.0, 255.0 };
	const float* channel_ranges = channel_range;
	int number_bins = 255;

	calcHist(&input_image, 1, channel_numbers, Mat(), histogram,
		1, &number_bins, &channel_ranges);

	float hist_max = 0;
	int floor_data = 0;

	for (int i = 1; i < number_bins; i++) {
		if (histogram.at<float>(i) > hist_max)
			hist_max = histogram.at<float>(i);
	}
	for (int i = 1; i < number_bins; i++) {
		if (histogram.at<float>(i) == hist_max)
			floor_data = i;
	}
	return floor_data;
}
Mat image_matching(Mat input_image1, Mat input_image2) {
	Ptr<Feature2D> feature = ORB::create();
	vector<KeyPoint> keypoints1, keypoints2;
	Mat desc1, desc2;
	feature->detectAndCompute(input_image1, Mat(), keypoints1, desc1);
	feature->detectAndCompute(input_image2, Mat(), keypoints2, desc2);
	Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_HAMMING);
	vector<DMatch> matches;
	matcher->match(desc1, desc2, matches);
	Mat dst;
	drawMatches(input_image1, keypoints1, input_image2, keypoints2, matches, dst);
	imshow("dst", dst);

	waitKey();
	destroyAllWindows();

	return dst;
}
Mat create_floorless_map(Mat input_image, int threshold) {
	Mat result_img = Mat::zeros(input_image.rows, input_image.cols, CV_8UC1);
	for (int j = 0; j < input_image.rows; j++)
		for (int i = 0; i < input_image.cols; i++) {
			if (input_image.at<uchar>(j, i) <= threshold + 10)
				result_img.at<uchar>(j, i) = 0;
			else
				result_img.at<uchar>(j, i) = input_image.at<uchar>(j, i);
		}
	return result_img;
}
Mat line_detection(Mat input_image) {

	Mat edge_img;
	Canny(input_image, edge_img, 240, 255);
	int houghP_min_len = 30;
	int houghP_min_cross = 30;

	// Below codes include linesP and houghP means probabilistic of line detection.

	// linesP variable contain position information about line's end point.
	vector<Vec4i> linesP;

	// (src, dst, rho'default:1', theta(for all direction), minimum cross,
	// minimum length, 
	// max_length in a one line(low value can make numerous lines)
	HoughLinesP(edge_img, linesP, 1, (CV_PI / 180), houghP_min_len, houghP_min_cross, 1000);

	// img_houghP would be dst image that has line data in RGB color scale.
	Mat img_houghP;
	input_image.copyTo(img_houghP);
	cvtColor(img_houghP, img_houghP, COLOR_GRAY2BGR);


	// 첫번째 직선 정보는 Group1로 지정.
	// 두번째 직선부터 첫번째 직선과의 거리 측정 후 일정 값 이상이면 Group2.
	double line_length = 0.0;
	vector<Vec4i> group1;		vector<Vec4i> group2;

	for (size_t i = 0; i < linesP.size(); i++)
	{
		Vec4i l = linesP[i];
		if (i == 0)
			group1.push_back(l);
		else {

		}
		cout << "l[0] : " << "\t" << l[0];
		cout << "\t" << "\t" "l[1] : " << "\t" << l[1];
		cout << "\t" << "\t" "l[2] : " << "\t" << l[2];
		cout << "\t" << "\t" "l[3] : " << "\t" << l[3] << endl;
		double temp_line_length = sqrt(pow(abs(l[0] - l[2]), 2) + pow(abs(l[1] - l[3]), 2));
		if (temp_line_length > line_length)
			line_length = temp_line_length;
		line(img_houghP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, 8);
	}

	imshow("aa", img_houghP);
	waitKey(0);

	/*
	// Below codes include lines, Houghlines means oridinary line detection.
	vector<Vec2f> lines;
	HoughLines(edge_img, lines, 1, CV_PI / 180, 30);
	Mat img_hough;
	input_image.copyTo(img_hough);
	cvtColor(img_hough, img_hough, COLOR_GRAY2BGR);

	// img_lane would be black image.
	Mat img_lane;
	threshold(edge_img, img_lane, 10, 255, THRESH_MASK);

	for (size_t i = 0; i < lines.size(); i++) {
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(img_hough, pt1, pt2, Scalar(0, 0, 255), 2, 8);
		line(img_lane, pt1, pt2, Scalar::all(255), 1, 8);
	}

	imshow("hough", img_hough);
	imshow("lane", img_lane);
	waitKey(0);
	*/

	return img_houghP;
}