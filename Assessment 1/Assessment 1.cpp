#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

// print help info to the console when required
void PrintHelp()
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
} // end function PrintHelp

int main(int argc, char **argv)
{
	// Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test";

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
			PrintHelp();
			return 0;
		} // end nested if...else
	} // end for

	string image_path = "images\\" + image_filename + ".ppm";

	cimg::exception_mode(0);

	// detect any potential exceptions
	try
	{
		CImg<unsigned char> input_image(image_path.c_str());
		CImgDisplay input_image_display(input_image, "Input image");
		size_t input_image_size = input_image.size();

		// Part 3 - host operations
		// 3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		std::cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl; // display the selected device

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

		// Part 4 - memory allocation
		std::vector<int> H(256, 0); // vector H for a histogram
		size_t H_elements = H.size(); // number of elements
		size_t H_size = H_elements * sizeof(int); // size in bytes

		std::vector<int> CH(H_elements, 0); // vector CH for a cumulative histogram
		size_t CH_elements = CH.size(); // number of elements
		size_t CH_size = CH_elements * sizeof(int); // size in bytes

		std::vector<double> mask(CH_elements, 255.0 / (int)input_image_size); // vector mask for normalising a cumulative histogram
		size_t mask_size = mask.size() * sizeof(double); // size in bytes
		
		std::vector<int> LUT(CH_elements, 0); // vector LUT for a normalised cumulative histogram which is used as a look-up table (LUT)
		size_t LUT_size = LUT.size() * sizeof(int); // size in bytes

		// Part 5 - device operations
		// device - buffers
		cl::Buffer buffer_input_image(context, CL_MEM_READ_ONLY, input_image_size);
		cl::Buffer buffer_H(context, CL_MEM_READ_WRITE, H_size);
		cl::Buffer buffer_CH(context, CL_MEM_READ_WRITE, CH_size);
		cl::Buffer buffer_mask(context, CL_MEM_READ_ONLY, mask_size);
		cl::Buffer buffer_LUT(context, CL_MEM_READ_WRITE, LUT_size);
		cl::Buffer buffer_output_image(context, CL_MEM_READ_WRITE, input_image_size); // its size should be the same as that of the input image

		// 5.1 Copy the image and vector mask to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_input_image, CL_TRUE, 0, input_image_size, &input_image.data()[0]);
		queue.enqueueWriteBuffer(buffer_mask, CL_TRUE, 0, mask_size, &mask[0]);
		queue.enqueueFillBuffer(buffer_H, 0, 0, H_size); // zero H buffer on device memory
		queue.enqueueFillBuffer(buffer_CH, 0, 0, CH_size); // zero CH buffer on device memory
		queue.enqueueFillBuffer(buffer_LUT, 0, 0, LUT_size); // zero LUT buffer on device memory

		// 5.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel kernel_1 = cl::Kernel(program, "get_histogram");
		cl::Kernel kernel_2 = cl::Kernel(program, "get_cumulative_histogram");
		cl::Kernel kernel_3 = cl::Kernel(program, "get_lut");
		cl::Kernel kernel_4 = cl::Kernel(program, "get_processed_image");

		kernel_1.setArg(0, buffer_input_image);
		kernel_1.setArg(1, buffer_H);
		kernel_1.setArg(2, (int)H_elements);

		kernel_2.setArg(0, buffer_H);
		kernel_2.setArg(1, buffer_CH);

		kernel_3.setArg(0, buffer_CH);
		kernel_3.setArg(1, buffer_mask);
		kernel_3.setArg(2, buffer_LUT);

		kernel_4.setArg(0, buffer_input_image);
		kernel_4.setArg(1, buffer_LUT);
		kernel_4.setArg(2, buffer_output_image);

		// queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input_image.width(), input_image.height(), input_image.spectrum()), cl::NullRange);
		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_image_size), cl::NullRange);
		queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(H_elements), cl::NullRange);
		queue.enqueueNDRangeKernel(kernel_3, cl::NullRange, cl::NDRange(CH_elements), cl::NullRange);
		queue.enqueueNDRangeKernel(kernel_4, cl::NullRange, cl::NDRange(input_image_size), cl::NullRange);

		vector<unsigned char> output_buffer(input_image_size);

		// 5.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_H, CL_TRUE, 0, H_size, &H[0]);
		queue.enqueueReadBuffer(buffer_CH, CL_TRUE, 0, CH_size, &CH[0]);
		queue.enqueueReadBuffer(buffer_LUT, CL_TRUE, 0, LUT_size, &LUT[0]);

		// std::cout << "H = " << H << std::endl; // uncomment when testing
		// std::cout << "CH = " << CH << std::endl; // uncomment when testing
		// std::cout << "LUT = " << LUT << std::endl; // uncomment when testing

		queue.enqueueReadBuffer(buffer_output_image, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		CImg<unsigned char> output_image(output_buffer.data(), input_image.width(), input_image.height(), input_image.depth(), input_image.spectrum());
		CImgDisplay output_image_display(output_image, "Output image");

		while (!input_image_display.is_closed() && !output_image_display.is_closed()
			&& !input_image_display.is_keyESC() && !output_image_display.is_keyESC())
		{
			input_image_display.wait(1);
			output_image_display.wait(1);
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