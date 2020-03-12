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

		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl; // display the selected device

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
		std::vector<mytype> A(8, 1); // allocate 8 elements with an initial value 1

		/*
		the following part adjusts the length of the input vector so it can be run for a specific workgroup size;
		if the total input length is divisible by the workgroup size, this makes the code more efficient
		*/
		size_t local_size = 8;
		size_t padding_size = A.size() % local_size;

		/*
		if the input vector is not a multiple of "local_size", insert additional neutral elements (0 for addition) so that the total will not be affected
		due to the modulo operator (%), the condition has the same effect as "padding_size != 0"
		*/
		if (padding_size)
		{
			std::vector<int> A_ext(local_size - padding_size, 0); // create an extra vector with neutral values
			A.insert(A.end(), A_ext.begin(), A_ext.end()); // append that extra vector to our input
		} // end if

		size_t A_elements = A.size(); // number of elements of Vector A
		size_t A_size = A.size() * sizeof(mytype); // size in bytes
		size_t nr_groups = A_elements / local_size;

		cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, A_size); // device - buffers

		// Part 4 - device operations
		// 4.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, A_size, &A[0]);

		// 4.2 Setup and execute all kernels (i.e. device code)
		cl::Kernel kernel_1 = cl::Kernel(program, "scan_bl"); // Blelloch basic exclusive scan

		kernel_1.setArg(0, buffer_A);

		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(A_elements), cl::NDRange(local_size)); // call all kernels in a sequence

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