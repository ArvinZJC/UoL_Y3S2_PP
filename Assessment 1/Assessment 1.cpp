/*
 * @Description: host code file of the tool applying histogram equalisation on a specified RGB image (8-bit/16-bit)
 * @Version: 1.9.2.20200322
 * @Author: Arvin Zhao
 * @Date: 2020-03-08 15:29:21
 * @Last Editors: Arvin Zhao
 * @LastEditTime: 2020-03-22 13:33:15
 */

#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

int main(int argc, char **argv)
{
	// Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	int mode_id = 0;
	string image_filename = "test.ppm";

	for (int i = 1; i < argc; i++)
	{
		// run the program according to the command line options
		if (strcmp(argv[i], "-l") == 0)
		{
			std::cout << ListPlatformsDevices();
			std::cout << "3 run modes:" << std::endl;
			std::cout << "   Mode 0, Fast Mode 1 (default)" << std::endl;
			std::cout << "      Compared to Basic Mode, program can consume less kernel execution time.\n" << std::endl;
			std::cout << "   Mode 1, Fast Mode 2" << std::endl;
			std::cout << "      Compared to Fast Mode 1, program may consume even less kernel execution time because of a different helper ";
			std::cout << "kernel. This mode only takes effect on a 16-bit iamge and is the same as Fast Mode 1 on an 8-bit image.\n" << std::endl;
			std::cout << "   Mode 2, Basic Mode" << std::endl;
			std::cout << "      This mode has brilliant compatibility but may significantly consume more kernel execution time." << std::endl;
			std::cout << "----------------------------------------------------------------" << std::endl;
		}
		else if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1)))
			platform_id = atoi(argv[++i]);
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1)))
			device_id = atoi(argv[++i]);
		else if ((strcmp(argv[i], "-m") == 0) && (i < (argc - 1)))
			mode_id = atoi(argv[++i]);
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1)))
			image_filename = argv[++i];
		else if (strcmp(argv[i], "-h") == 0)
		{
			// print help info to the console
			std::cerr << "Application usage:" << std::endl;
			std::cerr << "  __ : (no option specified) run with default input image file in default run mode on 1st device of 1st platform" << std::endl;
			std::cerr << "  -l : list all platforms, devices, and run modes, and then run as no options specified if no other options" << std::endl;
			std::cerr << "  -p : select platform" << std::endl;
			std::cerr << "  -d : select device" << std::endl;
			std::cerr << "  -m : select run mode" << std::endl;
			std::cerr << "  -f : specify input image file" << std::endl;
			std::cerr << "       ATTENTION: 1. \"test.ppm\" is default" << std::endl;
			std::cerr << "                  2. Please select a PPM image file (8-bit/16-bit RGB)" << std::endl;
			std::cerr << "                  3. The specified image should be put under the folder \"images\"" << std::endl;
			std::cerr << "  -h : print this message" << std::endl;
			return 0;
		} // end nested if...else
	} // end for

	// check if the run mode ID is valid
	if (mode_id != 0 && mode_id != 1 && mode_id != 2)
	{
		std::cout << "Program - ERROR: Inexistent run mode ID." << std::endl;
		return 0;
	} // end if

	string image_path = "images/" + image_filename;

	cimg::exception_mode(0);

	// detect any potential exceptions
	try
	{
		// Part 2 - image info loading
		CImg<unsigned short> input_image(image_path.c_str()); // read data from an RGB image file (8-bit/16-bit)
		CImg<unsigned char> input_image_8; // "unsigned char" is sufficient for data from an 8-bit image if any
		size_t input_image_elements = input_image.size(); // number of elements
		size_t input_image_size = input_image_elements * sizeof(unsigned short); // size in bytes
		int input_image_width = input_image.width(), input_image_height = input_image.height();
		int bin_count = input_image.max() <= 255 ? 256 : 65536; // bin numbers of an image (8-bit: 256, 16-bit: 65536)
		float scale = 1.0f; // the scale for displaying an image
		
		// set the scale for resizing when the image expands the standard
		if (input_image_width > 1024)
			scale = 1000.0f / input_image_width;
		else if (input_image_height > 768)
			scale = 750.0f / input_image_height;

		if (scale != 1.0f)
			std::cout << "ATTENTION: Large input and output images are resized to provide a better view. This does NOT modify the input image data for processing.\n" << std::endl;

		CImgDisplay input_image_display;

		// use proper data type to save memory transfer time
		if (bin_count == 256)
		{
			input_image_8.load(image_path.c_str());
			input_image_size = input_image_elements * sizeof(unsigned char); // it is equal to "input_image_elements" because the value of "sizeof(unsigned char)" is 1

			/*
			display the input 16-bit image;
			resize to provide a better view when necessary (this does not modify the input image data for processing)
			*/
			input_image_display.assign(CImg<unsigned char>(input_image_8).resize((int)(input_image_width * scale), (int)(input_image_height * scale)), "Input image (8-bit)");
		} // end if
		else
			/*
			display the input 16-bit image;
			resize to provide a better view when necessary (this does not modify the input image data for processing)
			*/
			input_image_display.assign(CImg<unsigned short>(input_image).resize((int)(input_image_width * scale), (int)(input_image_height * scale)), "Input image (16-bit)");
		
		// Part 3 - host operations
		// 3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		std::cout << "Running in " << (mode_id == 0 ? "Fast Mode 1" : (mode_id == 1 ? "Fast Mode 2" : "Basic Mode")) << " on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl; // display the selected device

		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE); // create a queue to which we will push commands for the device and enable profiling for the queue

		// 3.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/assessment1_kernels.cl");

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
		typedef unsigned int standard; // use "unsigned int" as the standard data type to avoid integer overflow when processing some large images

		std::vector<standard> H(bin_count, 0); // vector H for a histogram
		size_t H_elements = H.size(); // number of elements
		size_t H_size = H_elements * sizeof(standard); // size in bytes

		std::vector<standard> CH(H_elements, 0); // vector CH for a cumulative histogram
		size_t CH_elements = CH.size(); // number of elements
		size_t CH_size = CH_elements * sizeof(standard); // size in bytes

		/*
		number of local elements when processing an 8-bit image;
		this is equal to the number of bins of an 8-bit image so as to use 1 work group for a kernel since it is not a large problem;
		it is also decided based on the fact that optimised cumulative histogram kernel execution time will be longer due to helper kernels needed for multiple work groups
		*/
		size_t local_elements_8 = 256;

		size_t local_size_8 = local_elements_8 * sizeof(standard); // size in bytes

		/*
		the following part adjusts the length of global elements of the histogram kernel for an 8-bit image;
		the aim is to ensure that the global size is a multiple of the local size
		*/
		size_t kernel1_global_elements_8 = input_image_elements;
		size_t kernel1_global_elements_8_padding = kernel1_global_elements_8 % local_elements_8;

		if (kernel1_global_elements_8_padding)
			kernel1_global_elements_8 += (local_elements_8 - kernel1_global_elements_8_padding);

		size_t local_elements_16 = cl::Kernel(program, "get_CH_pro").getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(context.getInfo<CL_CONTEXT_DEVICES>()[0]); // get the max kernel workgroup size as the number of local elements when processing a 16-bit image
		size_t local_size_16 = local_elements_16 * sizeof(standard); // size in bytes
		size_t group_count = bin_count == 256 ? 1 : CH_elements / local_elements_16;

		/*
		avoid wrong results caused by an optimised cumulative histogram helper kernel due to its limitation;
		this is affected by the number of local elements when processing a 16-bit image;
		since the number is basically a multiple of 4, the limitation is seldom reached
		*/
		mode_id = (mode_id == 1 && bin_count == 65536 && group_count % 4 != 0) ? 0 : mode_id;

		/*
		the following part adjusts the length of global elements of the cumulative histogram kernel for a 16-bit image;
		the aim is to ensure that the global size is a multiple of the local size
		*/
		size_t kernel2_global_elements_16 = H_elements;
		size_t kernel2_global_elements_16_padding = kernel2_global_elements_16 % local_elements_16;

		if (kernel2_global_elements_16_padding)
			kernel2_global_elements_16 += (local_elements_16 - kernel2_global_elements_16_padding);

		std::vector<standard> BS(group_count, 0); // create a separate vector whose length is equal to the number of work groups to store the block sums
		size_t BS_size = BS.size() * sizeof(standard); // size in bytes

		std::vector<standard> BS_scanned(group_count, 0); // create a separate vector whose length is equal to the number of work groups to perform an exclusive scan on the block sums
		size_t BS_scanned_size = BS_scanned.size() * sizeof(standard); // size in bytes
		
		std::vector<standard> LUT(CH_elements, 0); // vector LUT for a normalised cumulative histogram which is used as a look-up table (LUT)
		size_t LUT_size = LUT.size() * sizeof(standard); // size in bytes

		// Part 5 - device operations
		// device - buffers
		cl::Buffer buffer_input_image(context, CL_MEM_READ_ONLY, input_image_size); // input image buffer
		cl::Buffer buffer_H(context, CL_MEM_READ_WRITE, H_size); // histogram buffer
		cl::Buffer buffer_CH(context, CL_MEM_READ_WRITE, CH_size); // cumulative histogram buffer
		cl::Buffer buffer_BS(context, CL_MEM_READ_WRITE, BS_size); // block sum buffer for cumulative histogram helper kernels
		cl::Buffer buffer_BS_scanned(context, CL_MEM_READ_WRITE, BS_scanned_size); // scanned block sum buffer for cumulative histogram helper kernels
		cl::Buffer buffer_LUT(context, CL_MEM_READ_WRITE, LUT_size); // LUT buffer
		cl::Buffer buffer_output_image(context, CL_MEM_READ_WRITE, input_image_size); // its size should be the same as that of the input image

		// 5.1 Copy the image to and initialise other arrays on device memory
		cl::Event input_image_event, H_input_event, CH_input_event, BS_input_event, BS_scanned_input_event, LUT_input_event; // add additional events to measure the upload time of each input vector

		if (bin_count == 256)
			queue.enqueueWriteBuffer(buffer_input_image, CL_TRUE, 0, input_image_size, &input_image_8.data()[0], NULL, &input_image_event);
		else
			queue.enqueueWriteBuffer(buffer_input_image, CL_TRUE, 0, input_image_size, &input_image.data()[0], NULL, &input_image_event);

		queue.enqueueFillBuffer(buffer_H, 0, 0, H_size, NULL, &H_input_event); // zero histogram buffer on device memory
		queue.enqueueFillBuffer(buffer_CH, 0, 0, CH_size, NULL, &CH_input_event); // zero cumulative histogram buffer on device memory
		queue.enqueueFillBuffer(buffer_LUT, 0, 0, LUT_size, NULL, &LUT_input_event); // zero LUT buffer on device memory

		if (bin_count == 65536)
		{
			queue.enqueueFillBuffer(buffer_BS, 0, 0, BS_size, NULL, &BS_input_event); // zero block sum buffer on device memory

			if (mode_id == 0)
				queue.enqueueFillBuffer(buffer_BS_scanned, 0, 0, BS_scanned_size, NULL, &BS_scanned_input_event); // zero scanned block sum buffer on device memory
		} // end if

		// 5.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel kernel1, kernel2, kernel2_helper1, kernel2_helper2, kernel2_helper3;

		// use optimised versions if any
		if (mode_id == 0 || mode_id == 1)
		{
			if (bin_count == 256)
			{
				std::cout << "Using optimised histogram and cumulative histogram kernels" << std::endl;

				kernel1 = cl::Kernel(program, "get_H_pro"); // Step 1: get a histogram with a specified number of bins
				kernel2 = cl::Kernel(program, "get_CH_pro"); // Step 2: get a cumulative histogram
				
				kernel1.setArg(2, cl::Local(local_size_8)); // local memory size for a local histogram
				kernel1.setArg(3, (standard)input_image_elements);

				kernel2.setArg(2, cl::Local(local_size_8)); // local memory size for a local histogram
				kernel2.setArg(3, cl::Local(local_size_8)); // local memory size for a cumulative histogram
			}
			else
			{
				std::cout << "Using optimised cumulative histogram kernel";
				/*
				Step 1: get a histogram with a specified number of bins;
				the optimised version does not support 16-bit images
				*/
				kernel1 = cl::Kernel(program, "get_H_16");

				kernel2 = cl::Kernel(program, "get_CH_pro"); // Step 2.1: get a preliminary cumulative histogram
				kernel2_helper1 = cl::Kernel(program, "get_BS"); // Step 2.2: get block sums of a preliminary cumulative histogram
				
				if (mode_id == 0 || mode_id == 2)
				{
					std::cout << std::endl;

					kernel2_helper2 = cl::Kernel(program, "get_scanned_BS_1"); // Step 2.3: get scanned block sums

					kernel2_helper2.setArg(1, buffer_BS_scanned);
				}
				else
				{
					std::cout << " including a helper kernel different from Fast Mode 1" << std::endl;

					kernel2_helper2 = cl::Kernel(program, "get_scanned_BS_2"); // Step 2.3: get scanned block sums
				} // end if...else

				kernel2_helper3 = cl::Kernel(program, "get_complete_CH"); // Step 2.4: get a complete cumulative histogram

				kernel2.setArg(2, cl::Local(local_size_16)); // local memory size for a local histogram
				kernel2.setArg(3, cl::Local(local_size_16)); // local memory size for a cumulative histogram

				kernel2_helper1.setArg(0, buffer_CH);
				kernel2_helper1.setArg(1, buffer_BS);
				kernel2_helper1.setArg(2, (int)local_elements_16);

				kernel2_helper2.setArg(0, buffer_BS);

				if (mode_id == 0 || mode_id == 2)
					kernel2_helper3.setArg(0, buffer_BS_scanned);
				else
					kernel2_helper3.setArg(0, buffer_BS);

				kernel2_helper3.setArg(1, buffer_CH);
			} // end if...else
		}
		// use basic versions
		else
		{
			std::cout << "Using basic kernels" << std::endl;

			// Step 1: get a histogram with a specified number of bins
			if (bin_count == 256)
				kernel1 = cl::Kernel(program, "get_H_8");
			else
				kernel1 = cl::Kernel(program, "get_H_16");

			kernel2 = cl::Kernel(program, "get_CH"); // Step 2: get a cumulative histogram

			kernel2.setArg(2, bin_count);
		} // end if...else

		std::cout << std::endl; // leave a blank line to provide a better console output format
		
		cl::Kernel kernel3 = cl::Kernel(program, "get_lut"); // Step 3: get a normalised cumulative histogram as an LUT
		cl::Kernel kernel4;

		// Step 4: get the output image according to the LUT
		if (bin_count == 256)
			kernel4 = cl::Kernel(program, "get_processed_image_8");
		else
			kernel4 = cl::Kernel(program, "get_processed_image_16");

		kernel1.setArg(0, buffer_input_image);
		kernel1.setArg(1, buffer_H);

		kernel2.setArg(0, buffer_H);
		kernel2.setArg(1, buffer_CH);

		kernel3.setArg(0, buffer_CH);
		kernel3.setArg(1, buffer_LUT);
		kernel3.setArg(2, bin_count);
		kernel3.setArg(3, input_image_width * input_image_height); // the total number of pixels (width * height)
		
		kernel4.setArg(0, buffer_input_image);
		kernel4.setArg(1, buffer_LUT);
		kernel4.setArg(2, buffer_output_image);

		cl::Event kernel1_event, kernel2_event, kernel2_helper1_event, kernel2_helper2_event, kernel2_helper3_event, kernel3_event, kernel4_event; // add additional events to measure the execution time of each kernel

		if ((mode_id == 0 || mode_id == 1) && bin_count == 256)
			queue.enqueueNDRangeKernel(kernel1, cl::NullRange, cl::NDRange(kernel1_global_elements_8), cl::NDRange(local_elements_8), NULL, &kernel1_event);
		else
			queue.enqueueNDRangeKernel(kernel1, cl::NullRange, cl::NDRange(input_image_elements), cl::NullRange, NULL, &kernel1_event);
		
		if ((mode_id == 0 || mode_id == 1) && bin_count == 65536)
		{
			queue.enqueueNDRangeKernel(kernel2, cl::NullRange, cl::NDRange(kernel2_global_elements_16), cl::NDRange(local_elements_16), NULL, &kernel2_event);
			queue.enqueueNDRangeKernel(kernel2_helper1, cl::NullRange, cl::NDRange(group_count), cl::NullRange, NULL, &kernel2_helper1_event);
			queue.enqueueNDRangeKernel(kernel2_helper2, cl::NullRange, cl::NDRange(group_count), cl::NullRange, NULL, &kernel2_helper2_event);
			queue.enqueueNDRangeKernel(kernel2_helper3, cl::NullRange, cl::NDRange(kernel2_global_elements_16), cl::NDRange(local_elements_16), NULL, &kernel2_helper3_event);
		}
		else if ((mode_id == 0 || mode_id == 1) && bin_count == 256)
			queue.enqueueNDRangeKernel(kernel2, cl::NullRange, cl::NDRange(H_elements), cl::NDRange(local_elements_8), NULL, &kernel2_event);
		else
			queue.enqueueNDRangeKernel(kernel2, cl::NullRange, cl::NDRange(H_elements), cl::NullRange, NULL, &kernel2_event);

		queue.enqueueNDRangeKernel(kernel3, cl::NullRange, cl::NDRange(CH_elements), cl::NullRange, NULL, &kernel3_event);
		queue.enqueueNDRangeKernel(kernel4, cl::NullRange, cl::NDRange(input_image_elements), cl::NullRange, NULL, &kernel4_event);

		// 5.3 Copy the result from device to host, print info to the console, and display the output image
		/*
		// uncomment the following section when testing
		queue.enqueueReadBuffer(buffer_H, CL_TRUE, 0, H_size, &H[0]);
		queue.enqueueReadBuffer(buffer_CH, CL_TRUE, 0, CH_size, &CH[0]);
		queue.enqueueReadBuffer(buffer_LUT, CL_TRUE, 0, LUT_size, &LUT[0]);

		std::cout << "H = " << H << std::endl;
		std::cout << "CH = " << CH << std::endl;

		if ((mode_id == 0 || mode_id == 1) && bin_count == 65536)
		{
			queue.enqueueReadBuffer(buffer_BS, CL_TRUE, 0, BS_size, &BS[0]);

			std::cout << "BS = " << BS << std::endl;

			if (mode_id == 0)
			{
				queue.enqueueReadBuffer(buffer_BS_scanned, CL_TRUE, 0, BS_scanned_size, &BS_scanned[0]);

				std::cout << "BS_scanned = " << BS_scanned << std::endl;
			} // end if
		} // end if
		
		std::cout << "LUT = " << LUT << std::endl;
		*/

		cl::Event output_image_event; // add additional events to measure the download time of each output vector
		CImgDisplay output_image_display;

		if (bin_count == 256)
		{
			vector<unsigned char> output_buffer_8(input_image_elements); // "unsigned char" is sufficient for data from the output 8-bit image buffer
			queue.enqueueReadBuffer(buffer_output_image, CL_TRUE, 0, output_buffer_8.size() * sizeof(unsigned char), &output_buffer_8.data()[0], NULL, &output_image_event);

			CImg<unsigned char> output_image_8(output_buffer_8.data(), input_image_width, input_image_height, input_image.depth(), input_image.spectrum());

			/*
			display the output 8-bit image;
			resize to provide a better view when necessary
			*/
			output_image_display.assign(output_image_8.resize((int)(input_image_width * scale), (int)(input_image_height * scale)), "Output image (8-bit)");
		}
		else
		{
			vector<unsigned short> output_buffer_16(input_image_elements); // "unsigned short" is required for data from the output 16-bit image buffer
			queue.enqueueReadBuffer(buffer_output_image, CL_TRUE, 0, output_buffer_16.size() * sizeof(unsigned short), &output_buffer_16.data()[0], NULL, &output_image_event);
			
			CImg<unsigned short> output_image_16(output_buffer_16.data(), input_image_width, input_image_height, input_image.depth(), input_image.spectrum());

			/*
			display the output 16-bit image;
			resize to provide a better view when necessary
			*/
			output_image_display.assign(output_image_16.resize((int)(input_image_width * scale), (int)(input_image_height * scale)), "Output image (16-bit)");
		} // end if...else

		cl_ulong total_upload_time = input_image_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - input_image_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()
			+ H_input_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - H_input_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()
			+ CH_input_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - CH_input_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()
			+ LUT_input_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - LUT_input_event.getProfilingInfo<CL_PROFILING_COMMAND_START>(); // total upload time of input vectors
		cl_ulong kernel1_time = kernel1_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel1_event.getProfilingInfo<CL_PROFILING_COMMAND_START>(); // histogram kernel execution time
		cl_ulong kernel2_time = kernel2_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel2_event.getProfilingInfo<CL_PROFILING_COMMAND_START>(); // cumulative histogram kernel execution time
		cl_ulong total_kernel_time = kernel1_time + kernel2_time
			+ kernel3_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel3_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()
			+ kernel4_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel4_event.getProfilingInfo<CL_PROFILING_COMMAND_START>(); // total execution time of kernels
		cl_ulong output_image_download_time = output_image_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - output_image_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

		if ((mode_id == 0 || mode_id == 1) && bin_count == 65536)
		{
			cl_ulong kernel2_helper_time = kernel2_helper1_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel2_helper1_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()
				+ kernel2_helper2_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel2_helper2_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()
				+ kernel2_helper3_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel2_helper3_event.getProfilingInfo<CL_PROFILING_COMMAND_START>(); // cumulative histogram helper kernel execution time
			total_upload_time += (BS_input_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - BS_input_event.getProfilingInfo<CL_PROFILING_COMMAND_START>());

			if (mode_id == 0)
				total_upload_time += (BS_scanned_input_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - BS_scanned_input_event.getProfilingInfo<CL_PROFILING_COMMAND_START>());

			kernel2_time += kernel2_helper_time;
			total_kernel_time += kernel2_helper_time;
		} // end if

		// display time in microseconds
		std::cout << "Memory transfer time: " << total_upload_time / 1000 << " us" << std::endl;
		std::cout << "Kernel execution time: " << total_kernel_time / 1000 << " us" << std::endl;
		std::cout << "   Histogram kernel execution time: " << kernel1_time / 1000 << " us" << std::endl;
		std::cout << "   Cumulative histogram kernel execution time: " << kernel2_time / 1000 << " us" << std::endl;
		std::cout << "Program execution time: " << (total_upload_time + total_kernel_time + output_image_download_time) / 1000 << " us" << std::endl;

		while (!input_image_display.is_closed() && !output_image_display.is_closed()
			&& !input_image_display.is_keyESC() && !output_image_display.is_keyESC())
		{
			input_image_display.wait(1);
			output_image_display.wait(1);
		} // end while
	}
	catch (const cl::Error& e)
	{
		std::cerr << "OpenCL - ERROR: " << e.what() << ", " << getErrorString(e.err()) << std::endl;
	}
	catch (CImgException& e)
	{
		std::cerr << "CImg - ERROR: " << e.what() << std::endl;
	} // end try...catch

	return 0;
} // end main