#include <iostream>
#include <vector>

#include "Utils.h"

void print_help()
{
	std::cerr << "Application usage:" << std::endl;
	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices, and run on the first device of the first platform" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
} // end function print_help

int main(int argc, char **argv)
{
	// Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)
	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1)))
			platform_id = atoi(argv[++i]);
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1)))
			device_id = atoi(argv[++i]);
		else if (strcmp(argv[i], "-l") == 0)
			std::cout << ListPlatformsDevices() << std::endl;
		else if (strcmp(argv[i], "-h") == 0)
		{
			print_help();
			return 0;
		} // end nested if...else
	} // end for

	// detect any potential exceptions
	try
	{
		// Part 2 - host operations
		// 2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl; // display the selected device

		cl::CommandQueue queue(context); // create a queue to which we will push commands for the device

		// 2.2 Load & build the device code
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

		typedef int mytype;

		// Part 3 - memory allocation
		std::vector<mytype> A(16, 1); // allocate 16 elements with an initial value 1
		size_t A_elements = A.size(); // number of elements in Vector A
		size_t A_size = A_elements * sizeof(mytype); // size in bytes

		cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, A_size); // device - buffers

		// Part 4 - device operations
		// 4.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, A_size, &A[0]);

		// 4.2 Setup and execute all kernels (i.e. device code)
		cl::Kernel kernel_1 = cl::Kernel(program, "scan_bl"); // Blelloch basic exclusive scan

		kernel_1.setArg(0, buffer_A);

		/*
		call the kernel;
		set the workgroup size to the number of elements in Vector A to ensure only one workgroup
		*/
		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(A_elements), cl::NDRange(A_elements));

		// 4.3 Copy the result from device to host
		std::cout << "A = " << A << std::endl;

		queue.enqueueReadBuffer(buffer_A, CL_TRUE, 0, A_size, &A[0]);

		std::cout << "A (final) = " << A << std::endl;
	}
	catch (cl::Error err)
	{
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	} // end try...catch

	return 0;
} // end main