#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<iostream>

#include<vector>

using namespace cv;
using namespace std;

Mat image_size_double(Mat input_image);
int get_floor_data(Mat input_image);

int main() {

	Mat src, global;
	src = imread("maps/3.5m/3.5m 3.pgm", IMREAD_GRAYSCALE);
	global = imread("maps/global_map.pgm", IMREAD_GRAYSCALE);
	//imshow("temp_src", src);
	//imshow("global_src", global);
	
	// check for image size.  Global image have to larger than src image.
	cout << "global rows: " << global.rows << endl;
	cout << "global cols: " << global.cols << endl;
	cout << "src rows: " << src.rows << endl;
	cout << "src cols: " << src.cols << endl;

	// Get Floor data to extract from the map.
	int global_floor_data = get_floor_data(global);
	int src_floor_data = get_floor_data(src);

	// Create new map image without floor.
	Mat global_without_floor = Mat::zeros(global.rows, global.cols, CV_8UC1);
	for (int j = 0; j < global.rows; j++)
		for (int i = 0; i < global.cols; i++) {
			if (global.at<uchar>(j, i) <= global_floor_data+10)
				global_without_floor.at<uchar>(j, i) = 0;
			else
				global_without_floor.at<uchar>(j, i) = global.at<uchar>(j, i);
		}
	//imshow("global_without_floor", global_without_floor);

	// Create large global map for image matching.
	// If global map has less width or height compared with src, it can occur error.
	Mat large_global_map = image_size_double(global_without_floor);
	//imshow("double_global", large_global_map);

	Mat filtered_src = Mat::zeros(src.rows, src.cols, CV_8UC1);
	for (int j = 0; j < src.rows; j++)
		for (int i = 0; i < src.cols; i++) {
			if (src.at<uchar>(j, i) <= src_floor_data+10)
				filtered_src.at<uchar>(j, i) = 0;
			else
				filtered_src.at<uchar>(j, i) = src.at<uchar>(j, i);
		}

	// GaussianBlur mask is unsuitable because it can disturb height information.
	// GaussianBlur(filtered_src, filtered_src, Size(3, 3), 0);
	//imshow("src", src);
	//imshow("Filtered_src", filtered_src);

	double minVal, maxVal;
	Point minLoc, maxLoc, matchLoc;
	Mat final;
	for (int i = 0; i < 6; i++)
	{
		Mat img_out;
		large_global_map.copyTo(img_out);

		int Matching_method = i;
		matchTemplate(img_out, filtered_src, final, i);

		normalize(final, final, 0, 1, NORM_MINMAX, -1, Mat());
		minMaxLoc(final, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
		if (Matching_method == 0 || Matching_method == 1) {
			matchLoc = minLoc;
		}
		else
			matchLoc = maxLoc;


		cvtColor(img_out, img_out, COLOR_GRAY2BGR);
		rectangle(img_out, matchLoc, Point(matchLoc.x + filtered_src.cols, matchLoc.y + filtered_src.rows),
			Scalar(255, 0, 255), 1);

		circle(img_out, matchLoc, 3, Scalar(0, 0, 255), 1);



		namedWindow("global", WINDOW_NORMAL);
		resizeWindow("global", large_global_map.cols, large_global_map.rows);
		imshow("global", img_out);
		imshow("templ", filtered_src);
		waitKey(0);

	}
	

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