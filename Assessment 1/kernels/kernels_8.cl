// a kernel file for processing an 8-bit image

#define BIN_COUNT 256

// get a histogram with a specified number of bins (global memory version)
kernel void get_histogram(global const uchar* image, global int* H)
{
	int id = get_global_id(0);
	
	// initialise the histogram to 0
	if (id < BIN_COUNT)
		H[id] = 0;

	/*
	compute the histogram;
	take a value from the input image as a bin index of the histogram
	*/
	atomic_inc(&H[image[id]]);
} // end function get_histogram

// get a histogram with a specified number of bins (local memory version)
kernel void get_histogram_pro(global const uchar* image, global int* H)
{
	local int H_local[BIN_COUNT];

	int local_id = get_local_id(0);
	int id = get_global_id(0);

	// initialise the local histogram to 0
	if (local_id < BIN_COUNT)
		H_local[local_id] = 0;

	barrier(CLK_LOCAL_MEM_FENCE); // wait for all local threads to finish the initialisation
	
	/*
	compute the local histogram;
	take a value from the input image as a bin index of the local histogram
	*/
	atomic_inc(&H_local[image[id]]);

	barrier(CLK_LOCAL_MEM_FENCE); // wait for all local threads to finish computing the local histogram

	// write the local histogram out to the global histogram
	if (local_id < BIN_COUNT)
		atomic_add(&H[local_id], H_local[local_id]);
} // end function get_histogram_pro

// get a cumulative histogram
kernel void get_cumulative_histogram(global int* H, global int* CH)
{
	int id = get_global_id(0);

	/*
	compute a cumulative histogram with an average histogram of the 3 colour channels' histograms;
	an average histogram is used for enabling basic histogram equalisation on both monochrome and colour images
	*/
	for (int i = id + 1; i < BIN_COUNT && id < BIN_COUNT; i++)
		atomic_add(&CH[i], H[id] / 3);
} // end function get_cumulative_histogram

// get a normalised cumulative histogram as a look-up table (LUT)
kernel void get_lut(global const int* CH, global int* LUT, const float mask)
{
	int id = get_global_id(0);

	if (id < BIN_COUNT)
		LUT[id] = CH[id] * mask;
} // end function get_lut

// get the output image according to the LUT
kernel void get_processed_image(global const uchar* input_image, global int* LUT, global uchar* output_image)
{
	int id = get_global_id(0);
	output_image[id] = LUT[input_image[id]];
} // end function get_processed_image