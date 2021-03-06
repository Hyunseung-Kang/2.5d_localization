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
Vec4i get_line_point(Vec2d input_line, Mat image);
Mat line_detection(Mat input_image);
double line2point_dist(Point line_point1, Point line_point2, Point point);
Vec2d get_avg_line(vector<Vec4i> group);

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
	
	Mat global_line_img, src_line_img;
	global_line_img = line_detection(global_floorless);
	src_line_img = line_detection(src_floorless);
	imshow("global", global_line_img);
	imshow("src", src_line_img);


	Mat rotate_img = src;
	Point2f center((src.cols - 1) / 2.0, (src.rows - 1) / 2.0);
	rotate_img = getRotationMatrix2D(center, 45, 1.0);
	Mat dst;
	warpAffine(src_line_img, dst, rotate_img, src.size());
	imshow("dst", dst);


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


	// ?????? ???? ?????? Group1?? ????.
	// ?????? ???????? ?????? ???????? ???? ???? ?? ???? ?? ???????? Group2.
	double line_length = 0.0;
	vector<Vec4i> group1;		vector<Vec4i> group2;


	//  ???? ?????? ?? ???? ???? ???????????? ??????????.
	// ???? ?????? ???????? ?????? ??????.
	for (size_t i = 0; i < linesP.size(); i++)
	{
		Vec4i l = linesP[i];
		cout << "vec4i : " << l << endl;
		double dist = 0.0;
		if (i == 0)
			group1.push_back(l);
		else {
			//dist = line2point_dist((l[0], l[1]), (l[2], l[3]), (linesP[0][0], linesP[0][1]));
			Point pt1(l[0], l[1]);
			Point pt2(l[2], l[3]);
			Point pt3(linesP[0][0], linesP[0][1]);
			dist = line2point_dist(pt1, pt2, pt3);
			if (dist < 5)
				group1.push_back(l);
			else
				group2.push_back(l);
		}
		line(img_houghP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, 8);
	}
	Vec2d line1, line2;
	line1 = get_avg_line(group1);		line2 = get_avg_line(group2);

	Vec4i line1_pt = get_line_point(line1, input_image);
	Vec4i line2_pt = get_line_point(line2, input_image);

	line(img_houghP, Point(line1_pt[0], line1_pt[1]), Point(line1_pt[2], line1_pt[3]), Scalar(0, 255, 0), 1, 8);
	line(img_houghP, Point(line2_pt[0], line2_pt[1]), Point(line2_pt[2], line2_pt[3]), Scalar(0, 255, 0), 1, 8);
	// Group 1, 2 ?? ??????????.  ?????? ?????? ???? ???????? ?????? ????????.
	cout << "Group1 size: " << group1.size() << endl;
	cout << "Group2 size: " << group2.size() << endl;
	
	imshow("aa", img_houghP);
	return img_houghP;
}

Vec4i get_line_point(Vec2d input_line, Mat image){
	// (??????, ????)
	Vec4i pt;	// (x1, y1)  (x2, y2)

	// y?????? ?????? ???? ???? ????
	if ((input_line[1] < image.rows) && (input_line[1] > 0)) {
		// ???? ?????? ???????? ?????? ???????? ????
		pt[0] = 0;	pt[1] = input_line[1];
		if (((input_line[0] * image.cols) + input_line[1]) < 0) {
			pt[2] = input_line[1] * (-1) / input_line[0];	pt[3] = 0;
		}
		// ???? ?????? ???????? ?????? ???????? ????
		else if (((input_line[0] * image.cols) + input_line[1]) > image.rows) {
			pt[2] = (image.rows - input_line[1] )/ input_line[0];	pt[3] = image.rows;
		}
		// ?????? ?????? ????
		else {
			pt[2] = image.cols;	pt[3] = (image.cols * input_line[0] + input_line[1]);
		}
	}
	// y?????? ?????? ???? ???? ????
	else {
		// ???? ?????? ?????? ???? ???? ????
		if (((input_line[0] * image.cols) + input_line[1] > 0) && ((input_line[0] * image.cols) + input_line[1] < image.rows)) {
			pt[2] = image.cols;	pt[3] = (input_line[0] * image.cols) + input_line[1];
			if ((input_line[1] * (-1) / input_line[0]) > 0 && (input_line[1] * (-1) / input_line[0]) < image.cols) {
				pt[0] = (input_line[1] * (-1) / input_line[0]);	pt[1] = 0;
			}
			else {
				pt[0] = (image.rows - input_line[1]) / input_line[0];	pt[1] = image.rows;
			}
		}
		else {
			pt[0] = (input_line[1] * (-1) / input_line[0]);	pt[1] = 0;
			pt[2] = (image.rows - input_line[1]) / input_line[0];	pt[3] = image.rows;
		}
	}
	return pt;
}



double line2point_dist(Point line_point1, Point line_point2, Point point) {

	int x1 = line_point1.x;	int y1 = line_point1.y;
	int x2 = line_point2.x;	int y2 = line_point2.y;
	int new_x = point.x;		int new_y = point.y;

	double upper = abs(((y2 - y1) * new_x) + ((x1 - x2) * new_y) + (x1 * y1 + x2 * y1 - x1 * y2 - x1 * y1));
	double lower = sqrt(pow(y2 - y1, 2) + pow(x1 - x2, 2));
	double distance = abs( upper / lower);

	return distance;
}

Vec2d get_avg_line(vector<Vec4i> group) {
	// ???? ?????? buff?? ???????? ?????? ???????? ????
	// ???????? ???? ???????? y?????? ????????.
	// (??????, ????)

	double diff = 0.0;	//??????
	double y_intercept = 0.0;

	for (int i = 0; i < group.size(); i++) {
		int x1 = group[i][0];		int y1 = group[i][1];
		int x2 = group[i][2];		int y2 = group[i][3];

		double diff_temp = (double)(y2 - y1) / (x2 - x1);
		diff = diff + diff_temp;
		y_intercept = y_intercept + (diff_temp * (x1) * (-1) + y1);
	}

	double diff_avg = double(diff) / (group.size());
	double intercept_avg = double(y_intercept) / (group.size());

	return Vec2d(diff_avg, intercept_avg);
}