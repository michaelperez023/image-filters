/**
 * @file main.cpp
 * Computer Vision Assignment 2
 * Dr. Abid
 * @author Michael Perez
 */

#include "opencv2/imgproc.hpp"
#include <iostream>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

// Global variables
const string image_name = "bicycle.bmp";
const int windowSize = 7; // Must be 3, 5, or 7
Mat image, editedImage, grayImage, imageX, imageY, sobel;
double red = 0.0, green = 0.0, blue = 0.0;
double sobmin, sobmax;
char key;

Mat applyKernel(Mat image, double kernelFilter[3][3])
{
	Mat result;
	image.copyTo(result);

	for (int y = 1; y < image.rows - 2; y++)
	{
		for (int x = 1; x < image.cols - 2; x++)
		{
			red = 0.0, green = 0.0, blue = 0.0;
			for (int k = -1; k <= 1; k++)
			{
				for (int j = -1; j <= 1; j++)
				{
					red += image.at<Vec3b>(y + j, x + k)[0] * kernelFilter[j + 1][k + 1];
					green += image.at<Vec3b>(y + j, x + k)[1] * kernelFilter[j + 1][k + 1];
					blue += image.at<Vec3b>(y + j, x + k)[2] * kernelFilter[j + 1][k + 1];
				}
			}

			if (red > 255)
				red = 255;
			else if (red < 0)
				red = 0;
			if (green > 255)
				green = 255;
			else if (green < 0)
				green = 0;
			if (blue > 255)
				blue = 255;
			else if (blue < 0)
				blue = 0;

			result.at<Vec3b>(y, x)[0] = red;
			result.at<Vec3b>(y, x)[1] = green;
			result.at<Vec3b>(y, x)[2] = blue;
		}
	}
	return result;
}

Mat applyKernel(Mat image, double kernelFilter[5][5])
{
	Mat result;
	image.copyTo(result);

	for (int y = 2; y < image.rows - 3; y++)
	{
		for (int x = 2; x < image.cols - 3; x++)
		{
			red = 0.0, green = 0.0, blue = 0.0;
			for (int k = -1; k <= 3; k++)
			{
				for (int j = -1; j <= 3; j++)
				{
					red += image.at<Vec3b>(y + j, x + k)[0] * kernelFilter[j + 1][k + 1];
					green += image.at<Vec3b>(y + j, x + k)[1] * kernelFilter[j + 1][k + 1];
					blue += image.at<Vec3b>(y + j, x + k)[2] * kernelFilter[j + 1][k + 1];
				}
			}

			if (red > 255)
				red = 255;
			else if (red < 0)
				red = 0;
			if (green > 255)
				green = 255;
			else if (green < 0)
				green = 0;
			if (blue > 255)
				blue = 255;
			else if (blue < 0)
				blue = 0;

			result.at<Vec3b>(y, x)[0] = red;
			result.at<Vec3b>(y, x)[1] = green;
			result.at<Vec3b>(y, x)[2] = blue;
		}
	}
	return result;
}

Mat applyKernel(Mat image, double kernelFilter[7][7])
{
	Mat result;
	image.copyTo(result);

	for (int y = 3; y < image.rows - 4; y++)
	{
		for (int x = 3; x < image.cols - 4; x++)
		{
			red = 0.0, green = 0.0, blue = 0.0;
			for (int k = -1; k <= 5; k++)
			{
				for (int j = -1; j <= 5; j++)
				{
					red += image.at<Vec3b>(y + j, x + k)[0] * kernelFilter[j + 1][k + 1];
					green += image.at<Vec3b>(y + j, x + k)[1] * kernelFilter[j + 1][k + 1];
					blue += image.at<Vec3b>(y + j, x + k)[2] * kernelFilter[j + 1][k + 1];
				}
			}

			if (red > 255)
				red = 255;
			else if (red < 0)
				red = 0;
			if (green > 255)
				green = 255;
			else if (green < 0)
				green = 0;
			if (blue > 255)
				blue = 255;
			else if (blue < 0)
				blue = 0;

			result.at<Vec3b>(y, x)[0] = red;
			result.at<Vec3b>(y, x)[1] = green;
			result.at<Vec3b>(y, x)[2] = blue;
		}
	}
	return result;
}

/**
 * @function main
 */
int main(int argc, char** argv)
{
	// Display windows
	image = imread(image_name);
	imshow("Image", image);

	if (image.empty())
	{
		cout << "The image could not be read. Exiting Program. ";
		return -1;
	}

	// Main loop
	while (1)
	{
		key = waitKey(10); // Scan for key press every 10 ms

		if (key == '1') // Apply the box filter that I coded myself
		{
			if (windowSize == 3)
			{
				double kernel[3][3] = {{1 / 9.0, 1 / 9.0, 1 / 9.0},
									{1 / 9.0, 1 / 9.0, 1 / 9.0},
									{1 / 9.0, 1 / 9.0, 1 / 9.0}};

				editedImage = applyKernel(image, kernel);
				imshow("Edited Image", editedImage);
			}
			else if (windowSize == 5)
			{
				double kernel[5][5] = {{1 / 25.0, 1 / 25.0, 1 / 25.0, 1 / 25.0, 1 / 25.0},
									{1 / 25.0, 1 / 25.0, 1 / 25.0, 1 / 25.0, 1 / 25.0},
									{1 / 25.0, 1 / 25.0, 1 / 25.0, 1 / 25.0, 1 / 25.0},
									{1 / 25.0, 1 / 25.0, 1 / 25.0, 1 / 25.0, 1 / 25.0},
									{1 / 25.0, 1 / 25.0, 1 / 25.0, 1 / 25.0, 1 / 25.0}};
				editedImage = applyKernel(image, kernel);
				imshow("Edited Image", editedImage);
			}
			else if (windowSize == 7)
			{
				double kernel[7][7] = {{1 / 49.0, 1 / 49.0, 1 / 49.0, 1 / 49.0, 1 / 49.0,  1 / 49.0,  1 / 49.0},
									{1 / 49.0, 1 / 49.0, 1 / 49.0, 1 / 49.0, 1 / 49.0,  1 / 49.0,  1 / 49.0},
									{1 / 49.0, 1 / 49.0, 1 / 49.0, 1 / 49.0, 1 / 49.0,  1 / 49.0,  1 / 49.0},
									{1 / 49.0, 1 / 49.0, 1 / 49.0, 1 / 49.0, 1 / 49.0,  1 / 49.0,  1 / 49.0},
									{1 / 49.0, 1 / 49.0, 1 / 49.0, 1 / 49.0, 1 / 49.0,  1 / 49.0,  1 / 49.0},
									{1 / 49.0, 1 / 49.0, 1 / 49.0, 1 / 49.0, 1 / 49.0,  1 / 49.0,  1 / 49.0},
									{1 / 49.0, 1 / 49.0, 1 / 49.0, 1 / 49.0, 1 / 49.0,  1 / 49.0,  1 / 49.0}};
				editedImage = applyKernel(image, kernel);
				imshow("Edited Image", editedImage);
			}
		}

		if (key == '2') // Apply the OpenCV box filter
		{
			boxFilter(image, editedImage, -1, Size(windowSize,windowSize), Point(-1,-1), true, BORDER_DEFAULT);
			imshow("Edited Image", editedImage);
		}

		if (key == '3') // Apply the Sobel filter towards x-axis edges that I coded myself to image
		{
			if (windowSize == 3)
			{
				double kernel[3][3] = {{-1, -2, -1},
									{0, 0, 0},
									{1, 2, 1}};

				editedImage = applyKernel(image, kernel);
				imshow("Edited Image", editedImage);
			}
			else if (windowSize == 5)
			{
				double kernel[5][5] = {{-2, -3, -4, -3, -2},
									{-1, -2, -3, -2, -1},
									{0, 0, 0, 0, 0},
									{1, 2, 3, 2, 1},
									{2, 3, 4, 3, 2}};
				editedImage = applyKernel(image, kernel);
				imshow("Edited Image", editedImage);
			}
			else if (windowSize == 7)
			{
				double kernel[7][7] = {{-3, -4, -5, -6, -5, -4, -3},
									{-2, -3, -4, -5, -4, -3, -2},
									{-1, -2, -3, -4, -3, -2, -1},
									{0, 0, 0, 0, 0, 0, 0},
									{1, 2, 3, 4, 3, 2, 1},
									{2, 3, 4, 5, 4, 3, 2},
									{3, 4, 5, 6, 5, 4, 3}};
				editedImage = applyKernel(image, kernel);
				imshow("Edited Image", editedImage);
			}
		}

		if (key == '4') // Apply the Sobel filter towards y-axis edges that I coded myself to image
		{
			if (windowSize == 3)
			{
				double kernel[3][3] = {
				{-1, 0, 1},
				{-2, 0, 2},
				{-1, 0, 1}};

				editedImage = applyKernel(image, kernel);
				imshow("Edited Image", editedImage);
			}
			else if (windowSize == 5)
			{
				double kernel[5][5] = {
									{2, 1, 0, -1, -2},
									{3, 2, 0, -2, -3},
									{4, 3, 0, -3, -4},
									{3, 2, 0, -2, -3},
									{2, 1, 0, -1, -2}
				};
				editedImage = applyKernel(image, kernel);
				imshow("Edited Image", editedImage);
			}
			else if (windowSize == 7)
			{
				double kernel[7][7] = {
									{3, 2, 1, 0, -1, -2, -3},
									{4, 3, 2, 0, -2, -3, -4},
									{5, 4, 3, 0, -3, -4, -5},
									{6, 5, 4, 0, -4, -5, -6},
									{5, 4, 3, 0, -3, -4, -5},
									{4, 3, 2, 0, -2, -3, - 4},
									{3, 2, 1, 0, -1, -2, -3}
				};
				editedImage = applyKernel(image, kernel);
				imshow("Edited Image", editedImage);
			}
		}

		if (key == '5') // Apply the Sobel filter towards x=axis and y-axis edges that I coded myself to image
		{
			if (windowSize == 3)
			{
				image.copyTo(editedImage);
				double kernel1[3][3] = {{-1, -2, -1},
										{0, 0, 0},
										{1, 2, 1}};

				double kernel2[3][3] = {{-1, 0, 1},
										{-2, 0, 2},
										{-1, 0, 1}};

				imageX = applyKernel(image, kernel1);
				imageY = applyKernel(image, kernel2);

				for (int y = 1; y < image.rows - 2; y++)
				{
					for (int x = 1; x < image.cols - 2; x++)
					{
						red = sqrt(pow(imageX.at<Vec3b>(y, x)[0], 2) + pow(imageY.at<Vec3b>(y, x)[0], 2));
						green = sqrt(pow(imageX.at<Vec3b>(y, x)[1], 2) + pow(imageY.at<Vec3b>(y, x)[1], 2));
						blue = sqrt(pow(imageX.at<Vec3b>(y, x)[2], 2) + pow(imageY.at<Vec3b>(y, x)[2], 2));

						if (red > 255)
							red = 255;
						else if (red < 0)
							red = 0;
						if (green > 255)
							green = 255;
						else if (green < 0)
							green = 0;
						if (blue > 255)
							blue = 255;
						else if (blue < 0)
							blue = 0;

						editedImage.at<Vec3b>(y, x)[0] = red;
						editedImage.at<Vec3b>(y, x)[1] = green;
						editedImage.at<Vec3b>(y, x)[2] = blue;
					}
				}
				imshow("Edited Image", editedImage);
			}
			else if (windowSize == 5)
			{
				image.copyTo(editedImage);
				double kernel1[5][5] = {{-2, -3, -4, -3, -2},
									{-1, -2, -3, -2, -1},
									{0, 0, 0, 0, 0},
									{1, 2, 3, 2, 1},
									{2, 3, 4, 3, 2}};

				double kernel2[5][5] = {{2, 1, 0, -1, -2},
									{3, 2, 0, -2, -3},
									{4, 3, 0, -3, -4},
									{3, 2, 0, -2, -3},
									{2, 1, 0, -1, -2}};

				imageX = applyKernel(image, kernel1);
				imageY = applyKernel(image, kernel2);

				for (int y = 1; y < image.rows - 2; y++)
				{
					for (int x = 1; x < image.cols - 2; x++)
					{
						red = sqrt(pow(imageX.at<Vec3b>(y, x)[0], 2) + pow(imageY.at<Vec3b>(y, x)[0], 2));
						green = sqrt(pow(imageX.at<Vec3b>(y, x)[1], 2) + pow(imageY.at<Vec3b>(y, x)[1], 2));
						blue = sqrt(pow(imageX.at<Vec3b>(y, x)[2], 2) + pow(imageY.at<Vec3b>(y, x)[2], 2));

						if (red > 255)
							red = 255;
						else if (red < 0)
							red = 0;
						if (green > 255)
							green = 255;
						else if (green < 0)
							green = 0;
						if (blue > 255)
							blue = 255;
						else if (blue < 0)
							blue = 0;

						editedImage.at<Vec3b>(y, x)[0] = red;
						editedImage.at<Vec3b>(y, x)[1] = green;
						editedImage.at<Vec3b>(y, x)[2] = blue;
					}
				}
				imshow("Edited Image", editedImage);
			}
			else if (windowSize == 7)
			{
				image.copyTo(editedImage);
				double kernel1[7][7] = { {-3, -4, -5, -6, -5, -4, -3},
									{-2, -3, -4, -5, -4, -3, -2},
									{-1, -2, -3, -4, -3, -2, -1},
									{0, 0, 0, 0, 0, 0, 0},
									{1, 2, 3, 4, 3, 2, 1},
									{2, 3, 4, 5, 4, 3, 2},
									{3, 4, 5, 6, 5, 4, 3} };
				
				double kernel2[7][7] = {{3, 2, 1, 0, -1, -2, -3},
									{4, 3, 2, 0, -2, -3, -4},
									{5, 4, 3, 0, -3, -4, -5},
									{6, 5, 4, 0, -4, -5, -6},
									{5, 4, 3, 0, -3, -4, -5},
									{4, 3, 2, 0, -2, -3, -4},
									{3, 2, 1, 0, -1, -2, -3}};

				imageX = applyKernel(image, kernel1);
				imageY = applyKernel(image, kernel2);

				for (int y = 1; y < image.rows - 2; y++)
				{
					for (int x = 1; x < image.cols - 2; x++)
					{
						red = sqrt(pow(imageX.at<Vec3b>(y, x)[0], 2) + pow(imageY.at<Vec3b>(y, x)[0], 2));
						green = sqrt(pow(imageX.at<Vec3b>(y, x)[1], 2) + pow(imageY.at<Vec3b>(y, x)[1], 2));
						blue = sqrt(pow(imageX.at<Vec3b>(y, x)[2], 2) + pow(imageY.at<Vec3b>(y, x)[2], 2));

						if (red > 255)
							red = 255;
						else if (red < 0)
							red = 0;
						if (green > 255)
							green = 255;
						else if (green < 0)
							green = 0;
						if (blue > 255)
							blue = 255;
						else if (blue < 0)
							blue = 0;

						editedImage.at<Vec3b>(y, x)[0] = red;
						editedImage.at<Vec3b>(y, x)[1] = green;
						editedImage.at<Vec3b>(y, x)[2] = blue;
					}
				}
				imshow("Edited Image", editedImage);
			}
		}

		if (key == '6')  // Apply the Sobel filter towards x=axis and y-axis edges (OpenCV) to image
		{
			// Convert the image to grayscale
			cvtColor(image, grayImage, COLOR_BGR2GRAY);

			// Generate grad_x and grad_y
			Mat grad_x, grad_y;

			// Gradient X
			Sobel(grayImage, imageX, CV_16S, 1, 0, windowSize, 10, 10, BORDER_DEFAULT);

			// Gradient Y
			Sobel(grayImage, imageY, CV_16S, 0, 1, windowSize, 10, 10, BORDER_DEFAULT);

			sobel = abs(imageX) + abs(imageY);
			
			minMaxLoc(sobel, &sobmin, &sobmax);
			sobel.convertTo(editedImage, CV_8U, 255. / sobmax, 0);

			imshow("Edited Image", editedImage);
		}

		if (key == '7') // Apply Gaussian Filter (OpenCV) to image
		{
			GaussianBlur(image, editedImage, Size(windowSize, windowSize), 0, 0, BORDER_DEFAULT);
			imshow("Edited Image", editedImage);
		}
	}
	return 0;
}
