// filters.h
//
// The seven image filters of the program. A box filter, Sobel derivative
// filters and the Sobel gradient magnitude are written from scratch; the
// others call OpenCV (cv::boxFilter, cv::Sobel with cv::magnitude, and
// cv::GaussianBlur) so that the results can be compared.
//
// Michael Perez, CAP 4410 Computer Vision, Assignment 2, Florida Polytechnic
// University (fall 2019). Rewritten as a library, with fixes and tests, in 2026.

#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/core.hpp>

#include <vector>

namespace filters
{
	// Window sizes accepted by the functions below: odd, from 3 to kMaxSize (the
	// largest Sobel aperture OpenCV supports). The program offers 3, 5 and 7.
	constexpr int kMaxSize = 31;
	bool IsValidSize(int size);

	// One-dimensional Sobel kernels of odd length `size`, unnormalised, as
	// cv::getDerivKernels returns them. The smoothing kernel is a row of
	// Pascal's triangle (1 2 1, 1 4 6 4 1, 1 6 15 20 15 6 1); the derivative
	// kernel is the central difference -1 0 1 convolved with the smoothing
	// kernel of length size - 2 (-1 0 1, -1 -2 0 2 1, -1 -4 -5 0 5 4 1).
	std::vector<double> SmoothingKernel(int size);
	std::vector<double> DerivativeKernel(int size);

	// size x size kernels (CV_64F). SobelKernelX(size)(r, c) is
	// smoothing[r] * derivative[c]: it differentiates along x and responds to
	// vertical edges. SobelKernelY is its transpose.
	cv::Mat BoxKernel(int size);  // every weight is 1 / size^2
	cv::Mat SobelKernelX(int size);
	cv::Mat SobelKernelY(int size);

	// The index that pixel `p` of a row or column of `length` pixels reads
	// when the image is mirrored at its edges without repeating the edge pixel
	// (gfedcb|abcdefgh|gfedcba, OpenCV's BORDER_REFLECT_101 and BORDER_DEFAULT).
	// `p` may lie more than `length` pixels outside.
	int ReflectIndex(int p, int length);

	// Correlates every channel of `image` (8-bit, 1 to 4 channels) with an odd,
	// square CV_64F `kernel`, as cv::filter2D does:
	//   result(y, x) = sum over j, k of kernel(j, k) * image(y + j - h, x + k - h)
	// with h = kernel.rows / 2, so the window is centred on (x, y). Pixels
	// outside the image are mirrored with ReflectIndex, so every pixel is
	// filtered. Returns a CV_64F image with the image's channels; the values
	// are neither rounded nor clipped. Throws std::invalid_argument for other
	// images or kernels.
	cv::Mat ApplyKernel(const cv::Mat& image, const cv::Mat& kernel);

	// The filters written from scratch. BoxFilter rounds to the nearest integer
	// and returns an 8-bit image, as cv::boxFilter does. SobelX and SobelY return
	// the signed derivatives along x and y (CV_64F), the same values that
	// cv::Sobel(image, out, CV_64F, 1, 0, size) and (..., 0, 1, size) give.
	// Magnitude returns sqrt(gx^2 + gy^2) for each pixel and channel.
	cv::Mat BoxFilter(const cv::Mat& image, int size);
	cv::Mat SobelX(const cv::Mat& image, int size);
	cv::Mat SobelY(const cv::Mat& image, int size);
	cv::Mat Magnitude(const cv::Mat& gx, const cv::Mat& gy);

	// The largest response of the size x size Sobel kernel to a step edge of
	// height 1: 4, 48 and 640 for sizes 3, 5 and 7.
	double StepResponse(int size);

	// Converts a derivative or magnitude (CV_64F) of a size x size Sobel filter
	// to an 8-bit image for display: |value| * StepResponse(3) / StepResponse(size),
	// rounded and clipped to 255. The 3x3 result is the plain Sobel response, and
	// a step edge of height h gives 4h at every window size.
	cv::Mat GradientToDisplay(const cv::Mat& gradient, int size);

	// The seven filters of the program, numbered by the key that applies them.
	enum class Filter
	{
		OwnBox = 1,       // box filter written from scratch
		OpenCVBox,        // cv::boxFilter
		OwnSobelY,        // derivative along y (horizontal edges), from scratch
		OwnSobelX,        // derivative along x (vertical edges), from scratch
		OwnMagnitude,     // gradient magnitude from the two above
		OpenCVMagnitude,  // cv::Sobel along x and y, then cv::magnitude
		OpenCVGaussian,   // cv::GaussianBlur with sigma 0, for which OpenCV uses
		                  // the fixed kernels (1 2 1)/4, (1 4 6 4 1)/16 and
		                  // (2 7 14 18 14 7 2)/64 at sizes 3, 5 and 7
	};

	struct FilterInfo
	{
		Filter filter;
		char key;            // '1' to '7'
		const char* name;    // printed when the filter is applied
		const char* label;   // short caption for the grid image
		const char* suffix;  // used in output file names
	};

	// The seven filters, in key order
	const std::vector<FilterInfo>& AllFilters();

	// Applies one filter to an 8-bit image with 1 to 4 channels and returns an
	// 8-bit image of the same size and channels: the smoothed image for keys 1,
	// 2 and 7, GradientToDisplay of the derivative or magnitude for keys 3 to 6.
	// Throws std::invalid_argument for an empty image, other image types or an
	// invalid size.
	cv::Mat Apply(Filter filter, const cv::Mat& image, int size);

	// The input and the seven results (in key order) side by side with
	// captions: the input and the smoothing filters (keys 1, 2 and 7) in the
	// first row, the Sobel filters (keys 3 to 6) in the second. Images are
	// enlarged by DisplayScale() so that small ones stay readable.
	int DisplayScale(const cv::Size& image_size);
	cv::Mat MakeGrid(const cv::Mat& image, const std::vector<cv::Mat>& results, int size);
}

#endif // FILTERS_H
