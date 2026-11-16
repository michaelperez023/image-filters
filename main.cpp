// main.cpp
//
// Command-line program for the filters in filters.h. It shows an image and
// applies one of seven filters when a key from 1 to 7 is pressed or, with
// --no-gui, writes all seven results and a grid of them.
//
// Michael Perez, CAP 4410 Computer Vision, Assignment 2, Florida Polytechnic
// University (fall 2019). Command-line options and the headless mode added in 2026.

#include "filters.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

namespace
{
	struct Options
	{
		std::string image = "bicycle.bmp";
		std::string output_dir;  // where to write the result images; empty = do not write
		int size = 3;            // window size: 3, 5 or 7
		bool gui = true;
	};

	void PrintUsage(std::ostream& out, const char* program)
	{
		out << "Usage: " << program << " [options] [image]\n"
		    << "\n"
		    << "Applies seven box, Sobel and Gaussian filters to an image\n"
		    << "(default: bicycle.bmp in the current directory).\n"
		    << "\n"
		    << "Options:\n"
		    << "  --size N    window size: 3, 5 or 7 (default 3)\n"
		    << "  --out DIR   write the result images to DIR\n"
		    << "  --no-gui    do not open windows: write the seven results and a grid\n"
		    << "              of them (to ./output unless --out is given) and exit\n"
		    << "  -h, --help  show this message\n"
		    << "\n"
		    << "In the window, press 1 to 7 to apply a filter, s to change the window\n"
		    << "size, and q or Esc to quit.\n";
	}

	bool ParseSize(const std::string& text, int& size)
	{
		if (text != "3" && text != "5" && text != "7")
			return false;
		size = text[0] - '0';
		return true;
	}

	enum class ParseResult { Run, Help, Error };

	ParseResult ParseArguments(int argc, char** argv, Options& options)
	{
		bool image_given = false;
		for (int i = 1; i < argc; ++i)
		{
			const std::string arg = argv[i];
			if (arg == "-h" || arg == "--help")
				return ParseResult::Help;

			if (arg == "--no-gui")
			{
				options.gui = false;
			}
			else if (arg == "--size")
			{
				if (i + 1 >= argc || !ParseSize(argv[i + 1], options.size))
				{
					std::cerr << "--size must be 3, 5 or 7\n";
					return ParseResult::Error;
				}
				++i;
			}
			else if (arg == "--out")
			{
				if (i + 1 >= argc)
				{
					std::cerr << "--out needs a directory\n";
					return ParseResult::Error;
				}
				options.output_dir = argv[++i];
			}
			else if (arg.size() > 1 && arg[0] == '-')
			{
				std::cerr << "Unknown option: " << arg << "\n";
				return ParseResult::Error;
			}
			else if (image_given)
			{
				std::cerr << "Only one image can be given\n";
				return ParseResult::Error;
			}
			else
			{
				options.image = arg;
				image_given = true;
			}
		}
		return ParseResult::Run;
	}

	bool CreateOutputDirectory(const std::string& dir)
	{
		std::error_code error;
		std::filesystem::create_directories(dir, error);
		if (error)
		{
			std::cerr << "Could not create " << dir << ": " << error.message() << "\n";
			return false;
		}
		return true;
	}

	// Output files are named after the input image and the window size, for
	// example bicycle_3x3_5_magnitude.png, so that runs with other images or
	// sizes can share a directory.
	std::string OutputPath(const Options& options, const std::string& suffix)
	{
		const std::string size = std::to_string(options.size);
		const std::string stem = std::filesystem::path(options.image).stem().string();
		const std::string name = stem + "_" + size + "x" + size + "_" + suffix + ".png";
		return (std::filesystem::path(options.output_dir) / name).string();
	}

	bool WriteImage(const cv::Mat& image, const std::string& path)
	{
		// zlib level 6 gives files about a third smaller than OpenCV's default
		// but is several times slower, so images over 4 megapixels are written
		// with the default.
		constexpr std::size_t kSmallImage = 4000000;  // pixels
		std::vector<int> png_options;
		if (image.total() <= kSmallImage)
			png_options = {cv::IMWRITE_PNG_COMPRESSION, 6};
		if (image.empty() || !cv::imwrite(path, image, png_options))
		{
			std::cerr << "Could not write " << path << "\n";
			return false;
		}
		std::cout << "Wrote " << path << "\n";
		return true;
	}

	// The largest difference, in grey levels, between two 8-bit images
	double LargestDifference(const cv::Mat& a, const cv::Mat& b)
	{
		return cv::norm(a, b, cv::NORM_INF);
	}

	int RunHeadless(const cv::Mat& image, const Options& options)
	{
		const std::vector<filters::FilterInfo>& all = filters::AllFilters();
		std::vector<cv::Mat> results;
		for (const filters::FilterInfo& info : all)
			results.push_back(filters::Apply(info.filter, image, options.size));

		std::cout << "Largest difference between the own and OpenCV filters, in grey levels:\n"
		          << "  box filter (keys 1 and 2):       " << LargestDifference(results[0], results[1]) << "\n"
		          << "  Sobel magnitude (keys 5 and 6):  " << LargestDifference(results[4], results[5]) << "\n";

		bool ok = true;
		for (std::size_t i = 0; i < all.size(); ++i)
			ok = WriteImage(results[i], OutputPath(options, all[i].suffix)) && ok;
		ok = WriteImage(filters::MakeGrid(image, results, options.size), OutputPath(options, "grid")) && ok;
		return ok ? 0 : 1;
	}

	int NextSize(int size)
	{
		return size == 3 ? 5 : size == 5 ? 7 : 3;
	}

	int RunGui(const cv::Mat& image, Options options)
	{
		const std::string window = "Image";
		cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
		cv::imshow(window, image);
		cv::waitKey(1);

		// Not every backend reports whether a window is still open, so only
		// watch for a closed window when the backend reports it as visible now.
		const bool can_detect_close = cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) >= 1;

		std::cout << "Press 1 to 7 to apply a filter, s to change the window size (now "
		          << options.size << "x" << options.size << "), q or Esc to quit." << std::endl;
		const filters::FilterInfo* shown = nullptr;  // the filter in the "Edited Image" window
		while (true)
		{
			const int key = cv::waitKey(50);
			if (key == 'q' || key == 'Q' || key == 27)
				break;
			if (can_detect_close && cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) < 1)
				break;

			const filters::FilterInfo* chosen = nullptr;
			for (const filters::FilterInfo& info : filters::AllFilters())
				if (key == info.key)
					chosen = &info;
			if (key == 's' || key == 'S')
			{
				options.size = NextSize(options.size);
				std::cout << "Window size " << options.size << "x" << options.size << std::endl;
				chosen = shown;  // show the last filter again with the new size
			}
			if (!chosen)
				continue;

			const cv::Mat result = filters::Apply(chosen->filter, image, options.size);
			std::cout << chosen->key << ": " << chosen->name << ", window " << options.size << "x"
			          << options.size << std::endl;
			cv::imshow("Edited Image", result);
			shown = chosen;
			if (!options.output_dir.empty())
				WriteImage(result, OutputPath(options, chosen->suffix));
		}
		cv::destroyAllWindows();
		return 0;
	}
}

int main(int argc, char** argv)
{
	Options options;
	switch (ParseArguments(argc, argv, options))
	{
	case ParseResult::Help:
		PrintUsage(std::cout, argv[0]);
		return 0;
	case ParseResult::Error:
		PrintUsage(std::cerr, argv[0]);
		return 2;
	case ParseResult::Run:
		break;
	}
	if (!options.gui && options.output_dir.empty())
		options.output_dir = "output";

	const cv::Mat image = cv::imread(options.image, cv::IMREAD_COLOR);
	if (image.empty())
	{
		std::cerr << "Could not read the image '" << options.image << "'.\n";
		return 1;
	}
	std::cout << "Image: " << options.image << " (" << image.cols << "x" << image.rows << "), window size "
	          << options.size << "x" << options.size << "\n";

	if (!options.output_dir.empty() && !CreateOutputDirectory(options.output_dir))
		return 1;

	try
	{
		if (!options.gui)
			return RunHeadless(image, options);
		return RunGui(image, options);
	}
	catch (const cv::Exception& e)
	{
		std::cerr << "OpenCV error: " << e.what() << "\n";
		if (options.gui)
			std::cerr << "To run without windows, use --no-gui.\n";
		return 1;
	}
	catch (const std::exception& e)
	{
		std::cerr << "Error: " << e.what() << "\n";
		return 1;
	}
}
