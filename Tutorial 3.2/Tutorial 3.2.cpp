#include <iostream>
#include <vector>

#include "Utils.h"

void print_help()
{
	std::cerr << "Application usage:" << std::endl;
	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
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
		/*
		 * host - input;
		 * allocate 10 elements with an initial value 1 - their sum is 10 so it should be easy to check the results
		 */
		std::vector<mytype> A(10, 1);
		// std::vector<mytype> A = { 6, 6, 6, 9, 9, 9, 9, 8, 5, 3}; // uncomment this in the introduction of Section 2.1
		// std::vector<mytype> A = { 6, 6, 6, 9, 9, 9, 9, 8, 5, 2333}; // uncomment this in Task4U-1 of Section 2.1

		/*
		 * the following part adjusts the length of the input vector so it can be run for a specific workgroup size;
		 * if the total input length is divisible by the workgroup size, this makes the code more efficient
		 */
		size_t local_size = 10;
		size_t padding_size = A.size() % local_size;

		/*
		 * if the input vector is not a multiple of "local_size", insert additional neutral elements (0 for addition) so that the total will not be affected
		 * due to the modulo operator (%), the condition has the same effect as "padding_size != 0"
		 */
		if (padding_size)
		{
			std::vector<int> A_ext(local_size-padding_size, 0); // create an extra vector with neutral values
			A.insert(A.end(), A_ext.begin(), A_ext.end()); // append that extra vector to our input
		} // end if

		size_t input_elements = A.size(); // number of input elements
		size_t input_size = A.size() * sizeof(mytype); // size in bytes
		size_t nr_groups = input_elements / local_size;

		// host - output
		std::vector<mytype> B(input_elements);
		/*
		 * this is suggested for Task4U-2 and Task4U-3 in Section 1.2 because we are using only a single element in the output vector;
		 * adjust the length to 1 so that there will be more memory available on a device for the input vector - it is important to perform reduce on larger datasets
		 */
		// std::vector<mytype> B(1);
		// std::vector<mytype> B(10); // uncomment this in Section 2.1

		size_t output_size = B.size() * sizeof(mytype); // size in bytes

		// device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);

		// Part 4 - device operations
		// 4.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size); // zero B buffer on device memory

		// 4.2 Setup and execute all kernels (i.e. device code)
		// cl::Kernel kernel_1 = cl::Kernel(program, "reduce_add_1"); // fixed 4-step reduce using interleaved addressing (Task4U-1 of Section 1.1)
		// cl::Kernel kernel_1 = cl::Kernel(program, "reduce_add_2"); // flexible step reduce using interleaved addressing (Task4U-2 of Section 1.1)
		// cl::Kernel kernel_1 = cl::Kernel(program, "reduce_add_3"); // reduce using local memory and interleaved addressing to be faster compared to operate directly on global memory (Task4U-1 of Section 1.2)
		// cl::Kernel kernel_1 = cl::Kernel(program, "reduce_add_4"); // reduce using local memory + accumulation of local sums into a single location and interleaved addressing (Task4U-2 of Section 1.2)
		// cl::Kernel kernel_1 = cl::Kernel(program, "reduce_add_5"); // reduce using local memory + accumulation of local sums into a single location and sequential addressing (Task4U-3 of Section 1.2)
		// cl::Kernel kernel_1 = cl::Kernel(program, "hist_1"); // a very simple histogram implementation (the introduction of Section 2.1)
		// cl::Kernel kernel_1 = cl::Kernel(program, "hist_2"); // a simple histogram implementation considering the number of bins (Task4U-1 of Section 2.1)
		// cl::Kernel kernel_1 = cl::Kernel(program, "scan_hs"); // Hillis-Steele basic inclusive scan (the introduction of Section 3)
		cl::Kernel kernel_1 = cl::Kernel(program, "scan_add"); // a double-buffered version of the Hillis-Steele inclusive scan (Task4U-1 of Section 3)

		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);
		kernel_1.setArg(2, cl::Local(local_size * sizeof(mytype))); // local memory size (uncomment this in Section 1.2 and Task 4U-1 of Section 3)
		// kernel_1.setArg(2, (int)B.size()); // the number of bins (uncomment this in Task4U-1 of Section 2.1)
		kernel_1.setArg(3, cl::Local(local_size * sizeof(mytype))); // local memory size (uncomment this in Task4U-1 of Section 3)

		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size)); // call all kernels in a sequence

		// 4.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

		std::cout << "A = " << A << std::endl;
		std::cout << "B = " << B << std::endl;
	}
	catch (cl::Error err)
	{
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	} // end try...catch

	return 0;
} // end main