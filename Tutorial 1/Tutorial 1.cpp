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

		// cl::CommandQueue queue(context); // create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE); // create a queue to which we will push commands for the device and enable profiling for the queue (Section 2.6)

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

		// Part 3 - memory allocation
		// host - input (comment the following 2 lines in Section 2.7)
		// std::vector<int> A = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }; // C++ 11 allows this type of initialisation
		// std::vector<int> B = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 };

		// accommodate larger input arrays (Section 2.7)
		std::vector<int> A(1000000);
		std::vector<int> B(1000000);
		
		size_t vector_elements = A.size(); // number of elements
		size_t vector_size = A.size() * sizeof(int); // size in bytes

		std::vector<int> C(vector_elements); // host - output

		// device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, vector_size);

		// Part 4 - device operations
		// 4.1 Copy arrays A and B to device memory (comment the following 2 lines in Section 2.7)
		// queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &A[0]);
		// queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, vector_size, &B[0]);
		// add additional events to measure the upload time for input vectors A and B
		cl::Event A_event, B_event;
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &A[0], NULL, &A_event);
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, vector_size, &B[0], NULL, &B_event);

		// 4.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel kernel_add = cl::Kernel(program, "add");
		kernel_add.setArg(0, buffer_A);
		kernel_add.setArg(1, buffer_B);
		kernel_add.setArg(2, buffer_C);

		// queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange);
		// create an event and attach it to a queue command responsible for the kernel launch (Section 2.6)
		cl::Event prof_event;
		queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange, NULL, &prof_event);
		// queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(5)); // define the work group size or local size (5) manually when calling the kernel (Section 1 in Tutorial 2)

		// cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0]; // get the device
		/*
		 * get the smallest work group size suggested (Section 1 in Tutorial 2);
		 * its multiples are also possible, up to the maximum work group size
		 */
		// cerr << kernel_add.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device) << endl;
		// cerr << kernel_add.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) << endl; // get the maximum work group size (Section 1 in Tutorial 2)

		// 4.3 Copy the result from device to host (comment the following 1 line in Section 2.7)
		// queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, vector_size, &C[0]);
		// add an additional event to measure the download time for the output vector C
		cl::Event C_event;
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, vector_size, &C[0], NULL, &C_event);

		// comment the following 3 lines in Section 2.7
		// std::cout << "A = " << A << std::endl;
		// std::cout << "B = " << B << std::endl;
		// std::cout << "C = " << C << std::endl;
		cl_ulong uploadTime_A = A_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - A_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		cl_ulong uploadTime_B = B_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - B_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		cl_ulong downloadTime_C = C_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - C_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		cl_ulong kernelExecutionTime = prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

		std::cout << "Total memory transfer time (unit: ns): " << uploadTime_A + uploadTime_B + downloadTime_C << std::endl; // display the total memory transfer time in nanoseconds (Section 2.7)
		std::cout << "Upload time for input vectors (unit: ns): A " << uploadTime_A << ", B " << uploadTime_B << std::endl; // display the upload time for input vectors A and B in nanoseconds (Section 2.7)
		std::cout << "Download time for the output vector C (unit: ns): " << downloadTime_C << std::endl << std:: endl; // display the download time for the output vector C in nanoseconds (Section 2.7)
		
		std::cout << "Kernel execution time (unit: ns): " << kernelExecutionTime << std::endl; // display the kernel execution time in nanoseconds (Section 2.6)
		std::cout << "Detailed breakdown of event (unit: us): " << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << std::endl << std::endl; // display the detailed breakdown of the event in microseconds (Section 2.6)

		std::cout << "Overall operation time (unit: ns): " << kernelExecutionTime + uploadTime_A + uploadTime_B + downloadTime_C << std::endl; // display the overall operation time in nanoseconds (Section 2.7)
	}
	catch (cl::Error err)
	{
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	} // end try...catch

	return 0;
} // end main