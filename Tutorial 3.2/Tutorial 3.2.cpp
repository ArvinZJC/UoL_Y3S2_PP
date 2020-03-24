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
		/*
		allocate 10 elements with an initial value 1 - their sum is 10 so it should be easy to check the results;
		a large size (1000) has been tested and the program worked well (adjust output to the console accordingly to save space)
		*/
		std::vector<mytype> A(10, 1);

		/*
		the following part adjusts the length of the input vector so it can be run for a specific workgroup size;
		if the total input length is divisible by the workgroup size, this makes the code more efficient
		*/
		size_t local_size = 5;
		size_t padding_size = A.size() % local_size;

		/*
		if the input vector is not a multiple of "local_size", insert additional neutral elements (0 for addition) so that the total will not be affected
		due to the modulo operator (%), the condition has the same effect as "padding_size != 0"
		*/
		if (padding_size)
		{
			std::vector<mytype> A_ext(local_size - padding_size, 0); // create an extra vector with neutral values
			A.insert(A.end(), A_ext.begin(), A_ext.end()); // append that extra vector to our input
		} // end if

		size_t A_elements = A.size(); // number of elements in Vector A
		size_t A_size = A_elements * sizeof(mytype); // size in bytes
		size_t nr_groups = A_elements / local_size;

		std::vector<mytype> B(A_elements);

		size_t B_size = B.size() * sizeof(mytype); // size in bytes

		std::vector<mytype> C(nr_groups, 0); // create a separate vector whose length is equal to the number of work groups to store the block sums

		size_t C_size = C.size() * sizeof(mytype); // size in bytes

		std::vector<mytype> D(nr_groups, 0); // create a separate vector whose length is equal to the number of work groups to perform an exclusive scan on the block sums

		size_t D_size = D.size() * sizeof(mytype); // size in bytes

		// device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, A_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, B_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, C_size);
		cl::Buffer buffer_D(context, CL_MEM_READ_WRITE, D_size);

		// Part 4 - device operations
		// 4.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, A_size, &A[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, B_size); // zero B buffer on device memory
		queue.enqueueFillBuffer(buffer_C, 0, 0, C_size); // zero C buffer on device memory
		queue.enqueueFillBuffer(buffer_D, 0, 0, D_size); // zero D buffer on device memory

		// 4.2 Setup and execute all kernels (i.e. device code)
		// take steps to extend the basic scan to enable a full scan operation on large vectors (Task4U-2 of Section 3)
		cl::Kernel kernel_1 = cl::Kernel(program, "scan_add"); // a double-buffered version of the Hillis-Steele inclusive scan (Step 1)
		cl::Kernel kernel_2 = cl::Kernel(program, "block_sum"); // calculate the block sums (Step 2)
		cl::Kernel kernel_3 = cl::Kernel(program, "scan_add_atomic"); // simple exclusive serial scan based on atomic operations (Step 3)
		cl::Kernel kernel_4 = cl::Kernel(program, "scan_add_adjust"); // adjust the values stored in partial scans by adding block sums to corresponding blocks (Step 4)

		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);
		kernel_1.setArg(2, cl::Local(local_size * sizeof(mytype))); // local memory size
		kernel_1.setArg(3, cl::Local(local_size * sizeof(mytype)));

		kernel_2.setArg(0, buffer_B);
		kernel_2.setArg(1, buffer_C);
		kernel_2.setArg(2, (int)local_size);

		kernel_3.setArg(0, buffer_C);
		kernel_3.setArg(1, buffer_D);

		kernel_4.setArg(0, buffer_B);
		kernel_4.setArg(1, buffer_D);

		// call all kernels in a sequence
		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(A_elements), cl::NDRange(local_size));

		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, B_size, &B[0]); // record Vector B after Step 1
		std::cout << "A = " << A << std::endl;
		std::cout << "B = " << B << std::endl;

		queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(nr_groups), cl::NullRange);
		queue.enqueueNDRangeKernel(kernel_3, cl::NullRange, cl::NDRange(nr_groups), cl::NullRange);
		queue.enqueueNDRangeKernel(kernel_4, cl::NullRange, cl::NDRange(A_elements), cl::NDRange(local_size));

		// 4.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, B_size, &B[0]);
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, C_size, &C[0]);
		queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, D_size, &D[0]);

		std::cout << "C = " << C << std::endl;
		std::cout << "D = " << D << std::endl;
		std::cout << "B (final) = " << B << std::endl;
	}
	catch (cl::Error err)
	{
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	} // end try...catch

	return 0;
} // end main