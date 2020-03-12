#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

void print_help()
{
	std::cerr << "Application usage:" << std::endl;
	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices, and run on the first device of the first platform" << std::endl;
	std::cerr << "  -f : specify input image file" << std::endl;
	std::cerr << "       ATTENTION: 1. \"test\" referring to \"test.ppm\" is default" << std::endl;
	std::cerr << "                  2. Only a PPM image file is accepted" << std::endl;
	std::cerr << "                  3. When using this option, please only enter the filename without the extension (i.e. test)" << std::endl;
	std::cerr << "                  4. The specified image should be put under the folder \"images\"" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
} // end function print_help

int main(int argc, char **argv)
{
	// Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test.ppm";

	for (int i = 1; i < argc; i++)
	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1)))
			platform_id = atoi(argv[++i]);
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1)))
			device_id = atoi(argv[++i]);
		else if (strcmp(argv[i], "-l") == 0)
			std::cout << ListPlatformsDevices() << std::endl;
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1)))
			image_filename = argv[++i];
		else if (strcmp(argv[i], "-h") == 0)
		{
			print_help();
			return 0;
		} // end nested if...else
	} // end for

	string image_path = "images\\" + image_filename;

	cimg::exception_mode(0);

	// detect any potential exceptions
	try
	{
		// Part 2 - image and mask info loading
		CImg<unsigned char> image_input(image_path.c_str());
		CImgDisplay disp_input(image_input, "input");

		/*
		a 3x3 convolution mask for Gaussian blur (uncomment this in Section 3.2.2);
		for more convolution masks, you can refer to: https://en.wikipedia.org/wiki/Kernel_(image_processing)
		*/
		std::vector<float> convolution_mask = { 1.f / 16, 2.f / 16, 1.f / 16,
												2.f / 16, 4.f / 16, 2.f / 16,
												1.f / 16, 2.f / 16, 1.f / 16 };

		// Part 3 - host operations
		// 3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl; // display the selected device

		cl::CommandQueue queue(context); // create a queue to which we will push commands for the device

		// 3.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		// build and debug the kernel code
		try
		{ 
			program.build();
		}
		catch (const cl::Error& err)
		{
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		} // end try...catch

		// Part 4 - device operations
		// device - buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size()); // it should be the same as input image
		cl::Buffer dev_convolution_mask(context, CL_MEM_READ_ONLY, convolution_mask.size() * sizeof(float)); // uncomment this in Section 3.2.2

		// 4.1 Copy images to device memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);
		queue.enqueueWriteBuffer(dev_convolution_mask, CL_TRUE, 0, convolution_mask.size() * sizeof(float), &convolution_mask[0]); // uncomment this in Section 3.2.2

		// 4.2 Setup and execute the kernel (i.e. device code)
		// cl::Kernel kernel = cl::Kernel(program, "identity"); // a simple 1D identity kernel that copies all pixels from A to B (original code of Section 2)
		// cl::Kernel kernel = cl::Kernel(program, "filter_r"); // perform colour channel filtering (Task4U-1 of Section 2)
		// cl::Kernel kernel = cl::Kernel(program, "invert"); // invert the intensity value of each pixel (Task4U-2 of Section 2)
		// cl::Kernel kernel = cl::Kernel(program, "rgb2grey"); // convert an input colour image into greyscale (Task4U-4 of Section 2)
		// cl::Kernel kernel = cl::Kernel(program, "identityND"); // a simple ND identity kernel (Section 3.2)
		// cl::Kernel kernel = cl::Kernel(program, "avg_filterND"); // a 2D averaging filter (Section 3.2)
		cl::Kernel kernel = cl::Kernel(program, "convolutionND"); // a 2D 3x3 convolution kernel (Section 3.2.2)
		kernel.setArg(0, dev_image_input);
		kernel.setArg(1, dev_image_output);
		kernel.setArg(2, dev_convolution_mask); // uncomment this in Section 3.2.2

		// queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange);
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image_input.width(), image_input.height(), image_input.spectrum()), cl::NullRange); // run a kernel in a 3D arrangement with image width, height, spectrum (colour channel) specifying values for 3 dimensions (Section 3.2, including Sections 3.2.1 & 3.2.2)

		vector<unsigned char> output_buffer(image_input.size());

		// 4.3 Copy the result from device to host
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image, "output");

		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC())
		{
			disp_input.wait(1);
			disp_output.wait(1);
		} // end while
	}
	catch (const cl::Error& err)
	{
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err)
	{
		std::cerr << "ERROR: " << err.what() << std::endl;
	} // end try...catch

	return 0;
} // end main