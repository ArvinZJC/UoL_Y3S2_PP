// get a histogram with a specified number of bins (global memory version)
kernel void get_histogram(global const ushort* image, global int* H, const int bin_count)
{
	int id = get_global_id(0);
	
	// initialise the histogram to 0
	if (id < bin_count)
		H[id] = 0;

	/*
	 * compute the histogram;
	 * take a value from the input image as a bin index of the histogram
	 */
	atomic_inc(&H[image[id]]);
} // end function get_histogram

// get a histogram with a specified number of bins (local memory version)
kernel void get_histogram_pro(global const ushort* image, global int* H, local int* H_local, const int bin_count)
{
	int lid = get_local_id(0);
	int id = get_global_id(0);

	// initialise the local histogram to 0
	if (lid < bin_count)
		H_local[lid] = 0;

	barrier(CLK_LOCAL_MEM_FENCE); // wait for all local threads to finish the initialisation
	
	/*
	 * compute the local histogram;
	 * take a value from the input image as a bin index of the local histogram
	 */
	atomic_inc(&H_local[image[id]]);

	barrier(CLK_LOCAL_MEM_FENCE); // wait for all local threads to finish computing the local histogram

	// write the local histogram out to the global histogram
	if (lid < bin_count)
		atomic_add(&H[lid], H_local[lid]);
} // end function get_histogram_pro

// get a cumulative histogram
kernel void get_cumulative_histogram(global int* H, global int* CH)
{
	int id = get_global_id(0);
	int N = get_global_size(0);

	/*
	 * compute a cumulative histogram with an average histogram of the 3 colour channels' histograms;
	 * an average histogram is used for enabling basic histogram equalisation on both monochrome and colour images
	 */
	for (int i = id + 1; i < N; i++)
		atomic_add(&CH[i], H[id] / 3);
} // end function get_cumulative_histogram

// get a normalised cumulative histogram as a look-up table (LUT)
kernel void get_lut(global const int* CH, global int* LUT, const float mask)
{
	int id = get_global_id(0);
	LUT[id] = CH[id] * mask;
} // end function get_lut

// get the output image according to the LUT
kernel void get_processed_image(global const ushort* input_image, global int* LUT, global ushort* output_image)
{
	int id = get_global_id(0);
	output_image[id] = LUT[input_image[id]];
} // end function get_processed_image