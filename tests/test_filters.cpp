// Tests for the filters in filters.h. The filters written from scratch are
// compared with OpenCV on the sample images and on random images, down to
// 1x1 pixel, and checked on synthetic images whose results are known.
//
// Usage: filters_tests [name]   runs one test, or all of them without a name.
// The exit status is 0 when every check passes. SAMPLE_DIR, set by CMake, is
// the directory that holds the sample images.

#include "filters.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef SAMPLE_DIR
#define SAMPLE_DIR "."
#endif

namespace
{
	int g_failures = 0;

	void Check(bool ok, const char* expression, const char* file, int line)
	{
		if (!ok)
		{
			std::fprintf(stderr, "%s:%d: check failed: %s\n", file, line, expression);
			++g_failures;
		}
	}

#define CHECK(expression) Check((expression), #expression, __FILE__, __LINE__)

	const int kSizes[] = {3, 5, 7};
	const char* const kSampleImages[] = {"bicycle.bmp", "dog.bmp", "edge_L-150x150.png", "south_L-150x150.png"};

	// The sample image, read as the program reads it (8-bit BGR)
	cv::Mat ReadSample(const char* name)
	{
		const std::string path = std::string(SAMPLE_DIR) + "/" + name;
		const cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
		if (image.empty())
			std::fprintf(stderr, "Could not read %s\n", path.c_str());
		return image;
	}

	// Images of many shapes, from 1x1 pixel to larger than every window, with
	// 1, 3 and 4 channels: uniform noise, and black-and-white noise for the
	// largest sums and differences.
	std::vector<cv::Mat> RandomImages()
	{
		const cv::Size shapes[] = {{1, 1}, {2, 1}, {1, 2}, {3, 2}, {2, 3}, {9, 1}, {1, 9}, {4, 7},
		                           {7, 7}, {13, 8}, {17, 31}, {64, 48}, {301, 203}};
		cv::RNG rng(2019);
		std::vector<cv::Mat> images;
		for (const cv::Size& shape : shapes)
		{
			for (int channels : {1, 3, 4})
			{
				cv::Mat noise(shape, CV_8UC(channels));
				rng.fill(noise, cv::RNG::UNIFORM, 0, 256);
				images.push_back(noise);

				cv::Mat black_and_white(shape, CV_8UC(channels));
				rng.fill(black_and_white, cv::RNG::UNIFORM, 0, 2);
				images.push_back(black_and_white * 255);
			}
		}
		return images;
	}

	double MaxDifference(const cv::Mat& a, const cv::Mat& b)
	{
		return cv::norm(a, b, cv::NORM_INF);
	}

	int CountDifferent(const cv::Mat& a, const cv::Mat& b)
	{
		cv::Mat difference;
		cv::absdiff(a, b, difference);
		return cv::countNonZero(difference.reshape(1));
	}

	cv::Mat OpenCVBox(const cv::Mat& image, int size)
	{
		cv::Mat result;
		cv::boxFilter(image, result, -1, cv::Size(size, size), cv::Point(-1, -1), true, cv::BORDER_DEFAULT);
		return result;
	}

	cv::Mat OpenCVSobel(const cv::Mat& image, int size, int dx, int dy)
	{
		cv::Mat result;
		cv::Sobel(image, result, CV_64F, dx, dy, size, 1.0, 0.0, cv::BORDER_DEFAULT);
		return result;
	}

	template <typename Function>
	bool Throws(Function function)
	{
		try
		{
			function();
			return false;
		}
		catch (const std::invalid_argument&)
		{
			return true;
		}
	}

	// The 1D kernels equal cv::getDerivKernels for every size, and the 3x3
	// Sobel kernels are the ones the 2019 code used for keys 3 and 4.
	void TestKernels()
	{
		for (int size = 3; size <= filters::kMaxSize; size += 2)
		{
			cv::Mat derivative, smoothing;
			cv::getDerivKernels(derivative, smoothing, 1, 0, size, false, CV_64F);
			const std::vector<double> own_smoothing = filters::SmoothingKernel(size);
			const std::vector<double> own_derivative = filters::DerivativeKernel(size);
			CHECK(MaxDifference(cv::Mat(own_smoothing), smoothing) == 0.0);
			CHECK(MaxDifference(cv::Mat(own_derivative), derivative) == 0.0);

			const cv::Mat sobel_x = smoothing * derivative.t();
			CHECK(MaxDifference(filters::SobelKernelX(size), sobel_x) == 0.0);
			CHECK(MaxDifference(filters::SobelKernelY(size), sobel_x.t()) == 0.0);
			CHECK(std::abs(cv::sum(filters::BoxKernel(size))[0] - 1.0) < 1e-12);
		}
		std::printf("kernels of sizes 3 to %d equal cv::getDerivKernels\n", filters::kMaxSize);

		const cv::Mat key4_2019 = (cv::Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
		const cv::Mat key3_2019 = (cv::Mat_<double>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
		CHECK(MaxDifference(filters::SobelKernelX(3), key4_2019) == 0.0);
		CHECK(MaxDifference(filters::SobelKernelY(3), key3_2019) == 0.0);

		CHECK(filters::StepResponse(3) == 4.0);
		CHECK(filters::StepResponse(5) == 48.0);
		CHECK(filters::StepResponse(7) == 640.0);
	}

	// ReflectIndex agrees with cv::borderInterpolate for BORDER_REFLECT_101,
	// also for positions several image widths outside.
	void TestReflectIndex()
	{
		int compared = 0;
		for (int length = 1; length <= 12; ++length)
		{
			for (int p = -3 * length - 5; p <= 4 * length + 5; ++p)
			{
				CHECK(filters::ReflectIndex(p, length) == cv::borderInterpolate(p, length, cv::BORDER_REFLECT_101));
				++compared;
			}
		}
		std::printf("%d positions agree with cv::borderInterpolate\n", compared);
	}

	// A single bright pixel: every window is centred on its pixel, so the box
	// filter spreads the pixel over the size x size square around it and a
	// Sobel filter returns its kernel, mirrored, around it. The 2019 code
	// shifted the 5x5 and 7x7 windows by 1 and 2 pixels.
	void TestCentredWindow()
	{
		const int side = 15, centre = 7;
		cv::Mat impulse(side, side, CV_8UC1, cv::Scalar(0));
		impulse.at<uchar>(centre, centre) = 255;

		for (int size : kSizes)
		{
			const int half = size / 2;
			const cv::Mat box = filters::BoxFilter(impulse, size);
			const cv::Mat gx = filters::SobelX(impulse, size);
			const cv::Mat gy = filters::SobelY(impulse, size);
			const cv::Mat kx = filters::SobelKernelX(size);
			const cv::Mat ky = filters::SobelKernelY(size);
			const int spread = cvRound(255.0 / (size * size));
			bool box_ok = true, sobel_ok = true;
			for (int y = 0; y < side; ++y)
			{
				for (int x = 0; x < side; ++x)
				{
					const bool inside = std::abs(y - centre) <= half && std::abs(x - centre) <= half;
					box_ok = box_ok && box.at<uchar>(y, x) == (inside ? spread : 0);
					const double expected_x = inside ? 255.0 * kx.at<double>(centre - y + half, centre - x + half) : 0.0;
					const double expected_y = inside ? 255.0 * ky.at<double>(centre - y + half, centre - x + half) : 0.0;
					sobel_ok = sobel_ok && gx.at<double>(y, x) == expected_x && gy.at<double>(y, x) == expected_y;
				}
			}
			CHECK(box_ok);
			CHECK(sobel_ok);
			std::printf("%dx%d: impulse response centred on the pixel (box value %d)\n", size, size, spread);
		}
	}

	// Step edges in all four directions. A step of h grey levels gives a
	// derivative of StepResponse(size) * h on both sides of the edge, with
	// the sign of the step, zero elsewhere and zero in the other direction.
	// The 2019 code clipped negative responses to 0, so the falling step of
	// edge_L-150x150.png (about 196 to 172) disappeared at size 3.
	void TestStepEdges()
	{
		const int side = 21, edge = 10;  // the step lies between pixels 9 and 10
		const int low = 172, high = 196;
		for (int size : kSizes)
		{
			const int half = size / 2;
			const double peak = filters::StepResponse(size) * (high - low);
			for (int direction = 0; direction < 4; ++direction)
			{
				const bool along_x = direction < 2;  // the intensity changes along x: a vertical edge
				const bool rising = direction % 2 == 0;
				cv::Mat image(side, side, CV_8UC1);
				for (int y = 0; y < side; ++y)
					for (int x = 0; x < side; ++x)
						image.at<uchar>(y, x) = static_cast<uchar>(((along_x ? x : y) < edge) == rising ? low : high);

				const cv::Mat gx = filters::SobelX(image, size);
				const cv::Mat gy = filters::SobelY(image, size);
				const cv::Mat& across = along_x ? gx : gy;
				const cv::Mat& along = along_x ? gy : gx;
				bool ok = true;
				for (int y = 0; y < side; ++y)
				{
					for (int x = 0; x < side; ++x)
					{
						const int position = along_x ? x : y;
						// Pixels within half - 1 of the edge see part of the kernel's
						// smoothing taper; the two pixels next to the edge see all of it.
						double expected = 0.0;
						if (position == edge - 1 || position == edge)
							expected = rising ? peak : -peak;
						else if (position >= edge - half && position < edge + half)
							continue;
						ok = ok && across.at<double>(y, x) == expected && along.at<double>(y, x) == 0.0;
					}
				}
				CHECK(ok);

				// The displayed magnitude at the edge is 4 times the step at every size
				const cv::Mat shown = filters::Apply(filters::Filter::OwnMagnitude, image, size);
				const int at_edge = along_x ? shown.at<uchar>(side / 2, edge) : shown.at<uchar>(edge, side / 2);
				CHECK(at_edge == 4 * (high - low));
				CHECK(MaxDifference(across, OpenCVSobel(image, size, along_x ? 1 : 0, along_x ? 0 : 1)) == 0.0);
			}
			std::printf("%dx%d: a step of %d grey levels gives a derivative of +/-%.0f, shown as %d\n", size, size,
			            high - low, peak, 4 * (high - low));
		}
	}

	// Flat images: the box filter keeps the value (a white image stays 255;
	// the 2019 5x5 box filter gave 254) and every derivative is zero.
	void TestFlatImages()
	{
		for (int value : {0, 100, 255})
		{
			for (const cv::Size& shape : {cv::Size(1, 1), cv::Size(3, 2), cv::Size(17, 9)})
			{
				for (int channels : {1, 3, 4})
				{
					const cv::Mat image(shape, CV_8UC(channels), cv::Scalar::all(value));
					const cv::Mat zero(shape, image.type(), cv::Scalar::all(0));
					for (int size : kSizes)
					{
						CHECK(MaxDifference(filters::BoxFilter(image, size), image) == 0.0);
						CHECK(cv::norm(filters::SobelX(image, size), cv::NORM_INF) == 0.0);
						CHECK(cv::norm(filters::SobelY(image, size), cv::NORM_INF) == 0.0);
						for (const filters::FilterInfo& info : filters::AllFilters())
						{
							const cv::Mat result = filters::Apply(info.filter, image, size);
							const bool smoothing = info.key == '1' || info.key == '2' || info.key == '7';
							CHECK(MaxDifference(result, smoothing ? image : zero) == 0.0);
						}
					}
				}
			}
		}
		std::printf("flat images keep their value and have zero gradient\n");
	}

	void CompareBox(const cv::Mat& image, int size, double& largest, int& different)
	{
		const cv::Mat own = filters::BoxFilter(image, size);
		const cv::Mat reference = OpenCVBox(image, size);
		CHECK(own.type() == reference.type() && own.size() == reference.size());
		largest = std::max(largest, MaxDifference(own, reference));
		different += CountDifferent(own, reference);
	}

	// Every window sum S from 0 to 255 * size^2 is rounded to the nearest
	// integer, (2S + size^2) / (2 size^2); size^2 is odd, so there are no ties.
	// Each size x size block of the test image has one sum, and the window of
	// its centre pixel is exactly that block. The 2019 code truncated, and its
	// 5x5 box filter turned white (S = 25 * 255) into 254.
	void TestBoxFilterAllSums()
	{
		for (int size : kSizes)
		{
			const int area = size * size;
			const int count = 255 * area + 1;
			const int per_row = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(count))));
			cv::Mat image(per_row * size, per_row * size, CV_8UC1, cv::Scalar(0));
			for (int sum = 0; sum < count; ++sum)
			{
				cv::Mat block = image(cv::Rect((sum % per_row) * size, (sum / per_row) * size, size, size));
				for (int i = 0; i < area; ++i)
					block.at<uchar>(i / size, i % size) = static_cast<uchar>(sum / area + (i < sum % area ? 1 : 0));
			}

			const cv::Mat own = filters::BoxFilter(image, size);
			const cv::Mat reference = OpenCVBox(image, size);
			int own_wrong = 0, opencv_wrong = 0, opencv_largest = 0;
			for (int sum = 0; sum < count; ++sum)
			{
				const int y = (sum / per_row) * size + size / 2;
				const int x = (sum % per_row) * size + size / 2;
				const int expected = (2 * sum + area) / (2 * area);
				own_wrong += own.at<uchar>(y, x) != expected ? 1 : 0;
				const int opencv_error = std::abs(reference.at<uchar>(y, x) - expected);
				opencv_wrong += opencv_error != 0 ? 1 : 0;
				opencv_largest = std::max(opencv_largest, opencv_error);
			}
			CHECK(own_wrong == 0);
			CHECK(opencv_largest <= 1);
			std::printf("%dx%d: %d window sums, rounding errors: own box filter %d, cv::boxFilter %d\n", size, size,
			            count, own_wrong, opencv_wrong);
		}
	}

	// The box filter written from scratch against cv::boxFilter. Both round to
	// the nearest integer, so they should agree exactly; one grey level is
	// allowed in case an OpenCV build rounds differently.
	void TestBoxFilterSamples()
	{
		for (const char* name : kSampleImages)
		{
			const cv::Mat image = ReadSample(name);
			CHECK(!image.empty());
			if (image.empty())
				continue;
			for (int size : kSizes)
			{
				double largest = 0.0;
				int different = 0;
				CompareBox(image, size, largest, different);
				CHECK(largest <= 1.0);
				std::printf("%-20s %dx%d: own box filter vs cv::boxFilter: largest difference %.0f, %d of %d values differ\n",
				            name, size, size, largest, different, static_cast<int>(image.total()) * image.channels());
			}
		}
	}

	void TestBoxFilterRandom()
	{
		const std::vector<cv::Mat> images = RandomImages();
		for (int size : kSizes)
		{
			double largest = 0.0;
			int different = 0;
			for (const cv::Mat& image : images)
				CompareBox(image, size, largest, different);
			CHECK(largest <= 1.0);
			std::printf("%zu random images, %dx%d: own box filter vs cv::boxFilter: largest difference %.0f, %d values differ\n",
			            images.size(), size, size, largest, different);
		}
	}

	struct SobelComparison
	{
		double derivative = 0.0;  // largest difference, own SobelX/SobelY vs cv::Sobel (CV_64F)
		double magnitude = 0.0;   // largest difference, own Magnitude vs cv::magnitude
		double shown = 0.0;       // largest difference between keys 5 and 6, in grey levels
		int shown_different = 0;  // values that differ between keys 5 and 6
		int shown_not_tie = 0;    // ... where the scaled magnitude is not half-way between integers
		int values = 0;           // values compared
	};

	void CompareSobel(const cv::Mat& image, int size, SobelComparison& result)
	{
		const cv::Mat gx = filters::SobelX(image, size);
		const cv::Mat gy = filters::SobelY(image, size);
		const cv::Mat reference_x = OpenCVSobel(image, size, 1, 0);
		const cv::Mat reference_y = OpenCVSobel(image, size, 0, 1);
		CHECK(gx.type() == reference_x.type() && gx.size() == reference_x.size());
		result.derivative = std::max({result.derivative, MaxDifference(gx, reference_x), MaxDifference(gy, reference_y)});

		const cv::Mat magnitude = filters::Magnitude(gx, gy);
		cv::Mat reference_magnitude;
		cv::magnitude(reference_x, reference_y, reference_magnitude);
		result.magnitude = std::max(result.magnitude, MaxDifference(magnitude, reference_magnitude));

		// The displayed images can differ where the scaled magnitude is exactly
		// half-way between two integers, if cv::magnitude is one unit in the last
		// place above or below the exact square root (as it is when OpenCV uses
		// Intel IPP) and so rounds the other way.
		const cv::Mat own_shown = filters::Apply(filters::Filter::OwnMagnitude, image, size);
		const cv::Mat opencv_shown = filters::Apply(filters::Filter::OpenCVMagnitude, image, size);
		result.shown = std::max(result.shown, MaxDifference(own_shown, opencv_shown));
		const double step_3x3 = filters::StepResponse(3), step = filters::StepResponse(size);
		const int count = image.cols * image.channels();
		for (int y = 0; y < image.rows; ++y)
		{
			for (int i = 0; i < count; ++i)
			{
				if (own_shown.ptr<uchar>(y)[i] == opencv_shown.ptr<uchar>(y)[i])
					continue;
				++result.shown_different;
				const double scaled = magnitude.ptr<double>(y)[i] * step_3x3 / step;
				if (std::abs(scaled - std::floor(scaled) - 0.5) > 1e-9)
					++result.shown_not_tie;
			}
		}
		result.values += count * image.rows;
	}

	void CheckSobel(const SobelComparison& result)
	{
		// The derivatives are sums of integers, and the magnitudes their square
		// roots, so double precision gives the same values up to the rounding
		// of the square root.
		CHECK(result.derivative == 0.0);
		CHECK(result.magnitude <= 1e-9);
		CHECK(result.shown <= 1.0);
		CHECK(result.shown_not_tie == 0);
	}

	void PrintSobel(const SobelComparison& result)
	{
		std::printf("largest difference: derivatives %g, magnitude %.2g, keys 5 and 6 %.0f", result.derivative,
		            result.magnitude, result.shown);
		if (result.shown_different > 0)
			std::printf(" (%d of %d values, %d not at a half)", result.shown_different, result.values,
			            result.shown_not_tie);
		std::printf("\n");
	}

	// The Sobel filters written from scratch against cv::Sobel and cv::magnitude
	void TestSobelSamples()
	{
		for (const char* name : kSampleImages)
		{
			const cv::Mat image = ReadSample(name);
			CHECK(!image.empty());
			if (image.empty())
				continue;
			for (int size : kSizes)
			{
				SobelComparison result;
				CompareSobel(image, size, result);
				CheckSobel(result);
				std::printf("%-20s %dx%d: ", name, size, size);
				PrintSobel(result);
			}
		}
	}

	void TestSobelRandom()
	{
		const std::vector<cv::Mat> images = RandomImages();
		for (int size : kSizes)
		{
			SobelComparison result;
			for (const cv::Mat& image : images)
				CompareSobel(image, size, result);
			CheckSobel(result);
			std::printf("%zu random images, %dx%d: ", images.size(), size, size);
			PrintSobel(result);
		}
	}

	// GradientToDisplay shows |value| scaled to the 3x3 kernel, rounded and
	// clipped to 255.
	void TestDisplay()
	{
		const cv::Mat values = (cv::Mat_<double>(1, 6) << 0.0, 40.0, -40.0, 101.0, -3000.0, 1e6);
		const int expected[3][6] = {
			{0, 40, 40, 101, 255, 255},   // 3x3: unchanged
			{0, 3, 3, 8, 250, 255},       // 5x5: divided by 12
			{0, 0, 0, 1, 19, 255},        // 7x7: divided by 160
		};
		for (int i = 0; i < 3; ++i)
		{
			const cv::Mat shown = filters::GradientToDisplay(values, kSizes[i]);
			CHECK(shown.type() == CV_8UC1);
			for (int j = 0; j < values.cols; ++j)
				CHECK(shown.at<uchar>(0, j) == expected[i][j]);
		}

		// Values just above and below a half are rounded in double precision
		// (two 7x7 magnitudes of dog.bmp, 116.500006 and 155.499996 after
		// scaling, which cv::Mat::convertTo rounds to 116 and 156), and exact
		// halves go to the even neighbour, as cvRound does.
		const cv::Mat near_half = (cv::Mat_<double>(1, 2) << 18640.00096566521, 24879.9993971061);
		const cv::Mat shown_7x7 = filters::GradientToDisplay(near_half, 7);
		CHECK(shown_7x7.at<uchar>(0, 0) == 117);
		CHECK(shown_7x7.at<uchar>(0, 1) == 155);
		const cv::Mat halves = (cv::Mat_<double>(1, 2) << 18.0, 30.0);  // 1.5 and 2.5 at 5x5
		const cv::Mat shown_5x5 = filters::GradientToDisplay(halves, 5);
		CHECK(shown_5x5.at<uchar>(0, 0) == 2);
		CHECK(shown_5x5.at<uchar>(0, 1) == 2);
		std::printf("display values are |g| * 4 / StepResponse(size), rounded and clipped\n");
	}

	// The seven filters by key: 8-bit results of the input's size and type,
	// and the own and OpenCV versions agree.
	void TestFiltersByKey()
	{
		const std::vector<filters::FilterInfo>& all = filters::AllFilters();
		CHECK(all.size() == 7);
		for (std::size_t i = 0; i < all.size(); ++i)
		{
			CHECK(all[i].key == static_cast<char>('1' + i));
			CHECK(static_cast<int>(all[i].filter) == static_cast<int>(i) + 1);
			for (std::size_t j = 0; j < i; ++j)
				CHECK(std::strcmp(all[i].suffix, all[j].suffix) != 0);
		}

		cv::RNG rng(4410);
		for (int channels : {1, 3, 4})
		{
			cv::Mat image(30, 40, CV_8UC(channels));
			rng.fill(image, cv::RNG::UNIFORM, 0, 256);
			cv::GaussianBlur(image, image, cv::Size(5, 5), 1.5);
			for (int size : kSizes)
			{
				std::vector<cv::Mat> results;
				for (const filters::FilterInfo& info : all)
				{
					results.push_back(filters::Apply(info.filter, image, size));
					CHECK(results.back().type() == image.type());
					CHECK(results.back().size() == image.size());
				}
				CHECK(MaxDifference(results[0], results[1]) <= 1.0);
				CHECK(MaxDifference(results[4], results[5]) <= 1.0);

				cv::Mat gaussian;
				cv::GaussianBlur(image, gaussian, cv::Size(size, size), 0.0, 0.0, cv::BORDER_DEFAULT);
				CHECK(MaxDifference(results[6], gaussian) == 0.0);

				// Keys 3 and 4 respond to horizontal and vertical edges
				CHECK(MaxDifference(results[2], filters::GradientToDisplay(OpenCVSobel(image, size, 0, 1), size)) == 0.0);
				CHECK(MaxDifference(results[3], filters::GradientToDisplay(OpenCVSobel(image, size, 1, 0), size)) == 0.0);
			}
		}
		std::printf("keys 1 to 7 give 8-bit results of the input's size and channels\n");
	}

	// The grid holds the input and the seven results, enlarged for small images
	void TestGrid()
	{
		cv::RNG rng(7);
		for (const cv::Size& shape : {cv::Size(150, 150), cv::Size(453, 300), cv::Size(1, 1)})
		{
			cv::Mat image(shape, CV_8UC3);
			rng.fill(image, cv::RNG::UNIFORM, 0, 256);
			std::vector<cv::Mat> results;
			for (const filters::FilterInfo& info : filters::AllFilters())
				results.push_back(filters::Apply(info.filter, image, 3));

			const cv::Mat grid = filters::MakeGrid(image, results, 3);
			const int scale = filters::DisplayScale(shape);
			CHECK(grid.type() == CV_8UC3);
			CHECK(grid.cols >= 4 * shape.width * scale && grid.rows >= 2 * shape.height * scale);
			std::printf("%dx%d image: %dx%d grid, enlarged %d times\n", shape.width, shape.height, grid.cols,
			            grid.rows, scale);
		}
		CHECK(filters::DisplayScale(cv::Size(150, 150)) == 2);
		CHECK(filters::DisplayScale(cv::Size(453, 300)) == 1);

		// Every image appears unchanged, enlarged twice, in its own cell: the
		// input and keys 1, 2 and 7 in the first row, keys 3 to 6 in the second
		const cv::Mat image = ReadSample("edge_L-150x150.png");
		CHECK(!image.empty());
		if (!image.empty())
		{
			std::vector<cv::Mat> results;
			for (const filters::FilterInfo& info : filters::AllFilters())
				results.push_back(filters::Apply(info.filter, image, 3));
			const cv::Mat grid = filters::MakeGrid(image, results, 3);
			const cv::Mat* panels[] = {&image,      &results[0], &results[1], &results[6],
			                           &results[2], &results[3], &results[4], &results[5]};
			for (int i = 0; i < 8; ++i)
			{
				const cv::Mat cell = grid(cv::Rect((i % 4) * grid.cols / 4, (i / 4) * grid.rows / 2, grid.cols / 4,
				                                   grid.rows / 2));
				cv::Mat enlarged, found;
				cv::resize(*panels[i], enlarged, cv::Size(), 2.0, 2.0, cv::INTER_NEAREST);
				cv::matchTemplate(cell, enlarged, found, cv::TM_SQDIFF);
				cv::Point best;
				cv::minMaxLoc(found, nullptr, nullptr, &best);
				CHECK(MaxDifference(cell(cv::Rect(best, enlarged.size())), enlarged) == 0.0);
			}
			std::printf("the input and the seven results appear unchanged in their cells\n");
		}

		CHECK(Throws([] { filters::MakeGrid(cv::Mat(4, 4, CV_8UC3), {}, 3); }));
	}

	// Window sizes, image types and kernels that the filters reject
	void TestArguments()
	{
		CHECK(filters::IsValidSize(3) && filters::IsValidSize(5) && filters::IsValidSize(7));
		CHECK(filters::IsValidSize(filters::kMaxSize));
		const cv::Mat image(8, 8, CV_8UC3, cv::Scalar::all(10));
		for (int size : {-3, 0, 1, 2, 4, 6, filters::kMaxSize + 2})
		{
			CHECK(!filters::IsValidSize(size));
			CHECK(Throws([&] { filters::BoxFilter(image, size); }));
			CHECK(Throws([&] { filters::SobelX(image, size); }));
			CHECK(Throws([&] { filters::Apply(filters::Filter::OpenCVMagnitude, image, size); }));
		}
		CHECK(Throws([] { filters::BoxFilter(cv::Mat(), 3); }));
		CHECK(Throws([] { filters::BoxFilter(cv::Mat(8, 8, CV_16UC3, cv::Scalar::all(10)), 3); }));
		CHECK(Throws([] { filters::BoxFilter(cv::Mat(8, 8, CV_8UC2, cv::Scalar::all(10)), 3); }));
		CHECK(Throws([&] { filters::ApplyKernel(image, cv::Mat(4, 4, CV_64F, cv::Scalar(0.0))); }));
		CHECK(Throws([&] { filters::ApplyKernel(image, cv::Mat(3, 3, CV_32F, cv::Scalar(0.0))); }));
		CHECK(Throws([] { filters::Magnitude(cv::Mat(4, 4, CV_64F), cv::Mat(4, 5, CV_64F)); }));
		CHECK(Throws([] { filters::GradientToDisplay(cv::Mat(4, 4, CV_8U), 3); }));
		std::printf("invalid sizes, images and kernels are rejected\n");
	}

	struct Test
	{
		const char* name;
		void (*run)();
	};

	const Test kTests[] = {
		{"kernels", TestKernels},
		{"reflect_index", TestReflectIndex},
		{"centred_window", TestCentredWindow},
		{"step_edges", TestStepEdges},
		{"flat_images", TestFlatImages},
		{"box_filter_all_sums", TestBoxFilterAllSums},
		{"box_filter_samples", TestBoxFilterSamples},
		{"box_filter_random", TestBoxFilterRandom},
		{"sobel_samples", TestSobelSamples},
		{"sobel_random", TestSobelRandom},
		{"display", TestDisplay},
		{"filters_by_key", TestFiltersByKey},
		{"grid", TestGrid},
		{"arguments", TestArguments},
	};
}

int main(int argc, char** argv)
{
	const char* only = argc > 1 ? argv[1] : nullptr;
	bool found = false;
	for (const Test& test : kTests)
	{
		if (only && std::strcmp(only, test.name) != 0)
			continue;
		found = true;
		const int before = g_failures;
		std::printf("[ RUN  ] %s\n", test.name);
		test.run();
		std::printf("[ %s ] %s\n", g_failures == before ? " OK " : "FAIL", test.name);
	}
	if (!found)
	{
		std::fprintf(stderr, "Unknown test: %s\n", only);
		return 1;
	}
	return g_failures == 0 ? 0 : 1;
}
