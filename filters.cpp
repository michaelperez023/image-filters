// filters.cpp
//
// Implementation of filters.h.
//
// Michael Perez, CAP 4410 Computer Vision, Assignment 2, Florida Polytechnic
// University (fall 2019). Rewritten as a library, with fixes and tests, in 2026.

#include "filters.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <string>

namespace filters
{
	namespace
	{
		// Grid images: small inputs are enlarged until their longer side is
		// close to this many pixels.
		constexpr int kGridTileSide = 300;

		void CheckSize(int size)
		{
			if (!IsValidSize(size))
				throw std::invalid_argument("the window size must be odd, from 3 to " +
				                            std::to_string(kMaxSize) + ", not " + std::to_string(size));
		}

		void CheckImage(const cv::Mat& image)
		{
			const int channels = image.channels();
			if (image.empty() || image.depth() != CV_8U || (channels != 1 && channels != 3 && channels != 4))
				throw std::invalid_argument("expected a non-empty 8-bit image with 1, 3 or 4 channels");
		}

		// Row `length - 1` of Pascal's triangle: 1; 1 1; 1 2 1; 1 3 3 1; ...
		std::vector<double> Binomial(int length)
		{
			std::vector<double> row(length, 0.0);
			row[0] = 1.0;
			for (int n = 1; n < length; ++n)
				for (int i = n; i > 0; --i)
					row[i] += row[i - 1];
			return row;
		}

		// The outer product column * row as a CV_64F matrix
		cv::Mat Outer(const std::vector<double>& column, const std::vector<double>& row)
		{
			cv::Mat kernel(static_cast<int>(column.size()), static_cast<int>(row.size()), CV_64F);
			for (int r = 0; r < kernel.rows; ++r)
				for (int c = 0; c < kernel.cols; ++c)
					kernel.at<double>(r, c) = column[r] * row[c];
			return kernel;
		}

		// Mirrored source index for each position from -half to length + half - 1
		std::vector<int> ReflectTable(int length, int half)
		{
			std::vector<int> table(length + 2 * half);
			for (int p = -half; p < length + half; ++p)
				table[p + half] = ReflectIndex(p, length);
			return table;
		}

		// Converts CV_64F values to 8-bit: each value (or its absolute value)
		// times `numerator` divided by `denominator`, rounded to the nearest
		// integer (halves to even, as cvRound does) and clipped to 0..255.
		// cv::Mat::convertTo is not used because it scales CV_64F values in
		// single precision, which can round values close to a half the wrong way.
		cv::Mat To8Bit(const cv::Mat& values, bool absolute, double numerator, double denominator)
		{
			cv::Mat result(values.size(), CV_MAKETYPE(CV_8U, values.channels()));
			const int count = values.cols * values.channels();
			for (int y = 0; y < values.rows; ++y)
			{
				const double* in = values.ptr<double>(y);
				uchar* out = result.ptr<uchar>(y);
				for (int i = 0; i < count; ++i)
					out[i] = cv::saturate_cast<uchar>((absolute ? std::abs(in[i]) : in[i]) * numerator / denominator);
			}
			return result;
		}

		cv::Mat ToBgr(const cv::Mat& image)
		{
			cv::Mat bgr;
			if (image.channels() == 1)
				cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
			else if (image.channels() == 4)
				cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
			else
				bgr = image;
			return bgr;
		}
	}

	bool IsValidSize(int size)
	{
		return size >= 3 && size <= kMaxSize && size % 2 == 1;
	}

	std::vector<double> SmoothingKernel(int size)
	{
		CheckSize(size);
		return Binomial(size);
	}

	std::vector<double> DerivativeKernel(int size)
	{
		CheckSize(size);
		const std::vector<double> smoothing = Binomial(size - 2);
		std::vector<double> kernel(size, 0.0);
		for (int i = 0; i < size - 2; ++i)
		{
			kernel[i] -= smoothing[i];
			kernel[i + 2] += smoothing[i];
		}
		return kernel;
	}

	cv::Mat BoxKernel(int size)
	{
		CheckSize(size);
		return cv::Mat(size, size, CV_64F, cv::Scalar(1.0 / (size * size)));
	}

	cv::Mat SobelKernelX(int size)
	{
		return Outer(SmoothingKernel(size), DerivativeKernel(size));
	}

	cv::Mat SobelKernelY(int size)
	{
		return Outer(DerivativeKernel(size), SmoothingKernel(size));
	}

	int ReflectIndex(int p, int length)
	{
		if (length <= 1)
			return 0;
		while (p < 0 || p >= length)
			p = p < 0 ? -p : 2 * (length - 1) - p;
		return p;
	}

	cv::Mat ApplyKernel(const cv::Mat& image, const cv::Mat& kernel)
	{
		CheckImage(image);
		if (kernel.type() != CV_64F || kernel.rows != kernel.cols || kernel.rows % 2 != 1)
			throw std::invalid_argument("expected an odd, square CV_64F kernel");

		const int size = kernel.rows;
		const int half = size / 2;
		const int channels = image.channels();
		const std::vector<int> rows = ReflectTable(image.rows, half);
		const std::vector<int> cols = ReflectTable(image.cols, half);

		cv::Mat result(image.size(), CV_MAKETYPE(CV_64F, channels), cv::Scalar::all(0.0));
		for (int y = 0; y < image.rows; ++y)
		{
			double* out = result.ptr<double>(y);
			// Kernel row j covers image row y + j - half, which is rows[y + j]
			for (int j = 0; j < size; ++j)
			{
				const uchar* in = image.ptr<uchar>(rows[y + j]);
				const double* weights = kernel.ptr<double>(j);
				for (int x = 0; x < image.cols; ++x)
				{
					double* sum = out + x * channels;
					for (int k = 0; k < size; ++k)
					{
						const uchar* pixel = in + cols[x + k] * channels;
						for (int c = 0; c < channels; ++c)
							sum[c] += weights[k] * pixel[c];
					}
				}
			}
		}
		return result;
	}

	cv::Mat BoxFilter(const cv::Mat& image, int size)
	{
		return To8Bit(ApplyKernel(image, BoxKernel(size)), false, 1.0, 1.0);
	}

	cv::Mat SobelX(const cv::Mat& image, int size)
	{
		return ApplyKernel(image, SobelKernelX(size));
	}

	cv::Mat SobelY(const cv::Mat& image, int size)
	{
		return ApplyKernel(image, SobelKernelY(size));
	}

	cv::Mat Magnitude(const cv::Mat& gx, const cv::Mat& gy)
	{
		if (gx.empty() || gx.depth() != CV_64F || gx.type() != gy.type() || gx.size() != gy.size())
			throw std::invalid_argument("expected two CV_64F images of the same size and type");

		cv::Mat result(gx.size(), gx.type());
		const int values = gx.cols * gx.channels();
		for (int y = 0; y < gx.rows; ++y)
		{
			const double* dx = gx.ptr<double>(y);
			const double* dy = gy.ptr<double>(y);
			double* out = result.ptr<double>(y);
			for (int i = 0; i < values; ++i)
				out[i] = std::sqrt(dx[i] * dx[i] + dy[i] * dy[i]);
		}
		return result;
	}

	double StepResponse(int size)
	{
		// At a step edge the derivative kernel's positive weights fall on one
		// side and its negative weights on the other, and the smoothing kernel
		// sums along the edge.
		const std::vector<double> smoothing = SmoothingKernel(size);
		const std::vector<double> derivative = DerivativeKernel(size);
		double rise = 0.0, total = 0.0;
		for (int i = 0; i < size; ++i)
		{
			rise += std::max(derivative[i], 0.0);
			total += smoothing[i];
		}
		return rise * total;
	}

	cv::Mat GradientToDisplay(const cv::Mat& gradient, int size)
	{
		if (gradient.empty() || gradient.depth() != CV_64F)
			throw std::invalid_argument("expected a CV_64F image");
		return To8Bit(gradient, true, StepResponse(3), StepResponse(size));
	}

	const std::vector<FilterInfo>& AllFilters()
	{
		static const std::vector<FilterInfo> filters = {
			{Filter::OwnBox, '1', "box filter (own)", "1: box, own", "1_box"},
			{Filter::OpenCVBox, '2', "box filter (OpenCV cv::boxFilter)", "2: box, OpenCV", "2_box_opencv"},
			{Filter::OwnSobelY, '3', "Sobel d/dy, horizontal edges (own)", "3: Sobel d/dy, own", "3_sobel_y"},
			{Filter::OwnSobelX, '4', "Sobel d/dx, vertical edges (own)", "4: Sobel d/dx, own", "4_sobel_x"},
			{Filter::OwnMagnitude, '5', "Sobel gradient magnitude (own)", "5: magnitude, own", "5_magnitude"},
			{Filter::OpenCVMagnitude, '6', "Sobel gradient magnitude (OpenCV cv::Sobel, cv::magnitude)",
			 "6: magnitude, OpenCV", "6_magnitude_opencv"},
			{Filter::OpenCVGaussian, '7', "Gaussian blur (OpenCV cv::GaussianBlur)", "7: Gaussian, OpenCV",
			 "7_gaussian_opencv"},
		};
		return filters;
	}

	cv::Mat Apply(Filter filter, const cv::Mat& image, int size)
	{
		CheckImage(image);
		CheckSize(size);

		cv::Mat result;
		switch (filter)
		{
		case Filter::OwnBox:
			return BoxFilter(image, size);
		case Filter::OpenCVBox:
			cv::boxFilter(image, result, -1, cv::Size(size, size), cv::Point(-1, -1), true, cv::BORDER_DEFAULT);
			return result;
		case Filter::OwnSobelY:
			return GradientToDisplay(SobelY(image, size), size);
		case Filter::OwnSobelX:
			return GradientToDisplay(SobelX(image, size), size);
		case Filter::OwnMagnitude:
			return GradientToDisplay(Magnitude(SobelX(image, size), SobelY(image, size)), size);
		case Filter::OpenCVMagnitude:
		{
			cv::Mat gx, gy, magnitude;
			cv::Sobel(image, gx, CV_64F, 1, 0, size, 1.0, 0.0, cv::BORDER_DEFAULT);
			cv::Sobel(image, gy, CV_64F, 0, 1, size, 1.0, 0.0, cv::BORDER_DEFAULT);
			cv::magnitude(gx, gy, magnitude);
			return GradientToDisplay(magnitude, size);
		}
		case Filter::OpenCVGaussian:
			cv::GaussianBlur(image, result, cv::Size(size, size), 0.0, 0.0, cv::BORDER_DEFAULT);
			return result;
		}
		throw std::invalid_argument("unknown filter");
	}

	int DisplayScale(const cv::Size& image_size)
	{
		const int longer = std::max(image_size.width, image_size.height);
		return longer > 0 ? std::max(1, kGridTileSide / longer) : 1;
	}

	cv::Mat MakeGrid(const cv::Mat& image, const std::vector<cv::Mat>& results, int size)
	{
		CheckImage(image);
		CheckSize(size);
		if (results.size() != AllFilters().size())
			throw std::invalid_argument("expected one result for each of the seven filters");
		for (const cv::Mat& result : results)
			if (result.size() != image.size() || result.type() != image.type())
				throw std::invalid_argument("the results must have the size and type of the image");

		// Cells in reading order: -1 is the input, 0 to 6 are keys 1 to 7
		constexpr int kColumns = 4;
		const int order[] = {-1, 0, 1, 6, 2, 3, 4, 5};
		const std::string input_label = "input, window " + std::to_string(size) + "x" + std::to_string(size);

		const int scale = DisplayScale(image.size());
		const cv::Size tile(image.cols * scale, image.rows * scale);
		const int gap = 8;

		// The captions grow with the tiles, so that they stay readable when the
		// grid is shown scaled down
		const int font = cv::FONT_HERSHEY_SIMPLEX;
		const double font_scale = std::clamp(tile.width / 530.0, 0.45, 1.5);
		const int thickness = font_scale >= 0.7 ? 2 : 1;
		int baseline = 0;
		const int text_height = cv::getTextSize("Ag", font, font_scale, thickness, &baseline).height;
		const int caption_height = text_height + baseline + 10;

		int cell_width = tile.width;
		for (int index : order)
		{
			const std::string label = index < 0 ? input_label : AllFilters()[index].label;
			int label_baseline = 0;
			const cv::Size text = cv::getTextSize(label, font, font_scale, thickness, &label_baseline);
			cell_width = std::max(cell_width, text.width + 8);
		}
		const int cell_height = caption_height + tile.height;
		const int rows = static_cast<int>(std::size(order)) / kColumns;

		cv::Mat grid(gap + rows * (cell_height + gap), gap + kColumns * (cell_width + gap), CV_8UC3,
		             cv::Scalar::all(255));
		for (int i = 0; i < static_cast<int>(std::size(order)); ++i)
		{
			const int index = order[i];
			const int x = gap + (i % kColumns) * (cell_width + gap);
			const int y = gap + (i / kColumns) * (cell_height + gap);
			const std::string label = index < 0 ? input_label : AllFilters()[index].label;
			cv::putText(grid, label, cv::Point(x + 4, y + caption_height - baseline - 5), font, font_scale,
			            cv::Scalar::all(40), thickness, cv::LINE_AA);

			cv::Mat enlarged;
			cv::resize(ToBgr(index < 0 ? image : results[index]), enlarged, tile, 0.0, 0.0, cv::INTER_NEAREST);
			enlarged.copyTo(grid(cv::Rect(x + (cell_width - tile.width) / 2, y + caption_height, tile.width,
			                              tile.height)));
		}
		return grid;
	}
}
