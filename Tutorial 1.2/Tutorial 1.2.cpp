#include <iostream>
#include <vector>

#include "Utils.h"

void print_help()
{
	std::cerr << "Application usage:" << std::endl;
	std::cerr << "  -f : select kernel function (0: add, 1: mult, 2: mult + add, 3: multadd)" << std::endl;
	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices, and run on the first device of the first platform" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
} // end function print_help

int main(int argc, char** argv)
{
	// Part 1 - handle command line options such as device selection, verbosity, etc.
	int function_id = 0;
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)
	{
		if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1)))
			function_id = atoi(argv[++i]);
		else if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1)))
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
		catch (const cl::Error & err)
		{
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		} // end try...catch

		// Part 3 - memory allocation
		// host - input
		std::vector<int> A = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }; // C++ 11 allows this type of initialisation
		std::vector<int> B = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 };

		size_t vector_elements = A.size(); // number of elements
		size_t vector_size = A.size() * sizeof(int); // size in bytes

		std::vector<int> C(vector_elements); // host - output

		// device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, vector_size);

		// Part 4 - device operations
		// 4.1 Copy arrays A and B to device memory, and add additional events to measure the upload time for input vectors A and B
		cl::Event A_event, B_event;
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &A[0], NULL, &A_event);
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, vector_size, &B[0], NULL, &B_event);

		// 4.2 Setup and execute the kernel (i.e. device code) according to the function selection
		cl::Kernel kernel_function1, kernel_function2;

		switch (function_id)
		{
		case 0:
		default:
			std::cout << "C = A + B" << std::endl;
			kernel_function1 = cl::Kernel(program, "add");
			break;

		case 1:
			std::cout << "C = A * B" << std::endl;
			kernel_function1 = cl::Kernel(program, "mult");
			break;

		case 2:
			std::cout << "C = A * B, C = C + B" << endl;
			kernel_function1 = cl::Kernel(program, "mult");
			kernel_function2 = cl::Kernel(program, "add");
			break;

		case 3:
			std::cout << "C = A * B + B" << std::endl;
			kernel_function1 = cl::Kernel(program, "multadd");
			break;
		} // end switch-case
		
		kernel_function1.setArg(0, buffer_A);
		kernel_function1.setArg(1, buffer_B);
		kernel_function1.setArg(2, buffer_C);

		// create an event and attach it to a queue command responsible for the kernel launch
		cl::Event prof_event;
		queue.enqueueNDRangeKernel(kernel_function1, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange, NULL, &prof_event);

		if (function_id == 2)
		{
			kernel_function2.setArg(0, buffer_C);
			kernel_function2.setArg(1, buffer_B);
			kernel_function2.setArg(2, buffer_C);
			queue.enqueueNDRangeKernel(kernel_function2, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange, NULL, &prof_event);
		} // end if

		// 4.3 Copy the result from device to host, and add an additional event to measure the download time for the output vector C
		cl::Event C_event;
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, vector_size, &C[0], NULL, &C_event);

		cl_ulong uploadTime_A = A_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - A_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		cl_ulong uploadTime_B = B_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - B_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		cl_ulong downloadTime_C = C_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - C_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		cl_ulong kernelExecutionTime = prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

		std::cout << "A = " << A << std::endl;
		std::cout << "B = " << B << std::endl;
		std::cout << "C = " << C << std::endl << std::endl;

		std::cout << "Total memory transfer time (unit: ns): " << uploadTime_A + uploadTime_B + downloadTime_C << std::endl; // display the total memory transfer time in nanoseconds
		std::cout << "Upload time for input vectors (unit: ns): A " << uploadTime_A << ", B " << uploadTime_B << std::endl; // display the upload time for input vectors A and B in nanoseconds
		std::cout << "Download time for the output vector C (unit: ns): " << downloadTime_C << std::endl << std::endl; // display the download time for the output vector C in nanoseconds

		std::cout << "Kernel execution time (unit: ns): " << kernelExecutionTime << std::endl; // display the kernel execution time in nanoseconds (Section 2.6)
		std::cout << "Detailed breakdown of event (unit: us): " << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << std::endl << std::endl; // display the detailed breakdown of the event in microseconds

		std::cout << "Overall operation time (unit: ns): " << kernelExecutionTime + uploadTime_A + uploadTime_B + downloadTime_C << std::endl; // display the overall operation time in nanoseconds
	}
	catch (cl::Error err)
	{
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	} // end try...catch

	return 0;
} // end main