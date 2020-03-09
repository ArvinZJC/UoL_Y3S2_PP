// get a histogram array with a specified number of bins
kernel void get_histogram(global const uchar* image, global int* H, const int nr_bins)
{
	int id = get_global_id(0);
	int bin_index = image[id]; // take value as a bin index
	
	if (id < nr_bins)
		H[id] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	atomic_inc(&H[bin_index]);
} // end function get_histogram

// get a cumulative histogram array
kernel void get_cumulative_histogram(global int* H, global int* CH)
{
	int id = get_global_id(0);
	int N = get_global_size(0);

	// simple exclusive serial scan based on atomic operations - sufficient for small number of elements
	for (int i = id + 1; i < N; i++)
		atomic_add(&CH[i], H[id] / 3);
} // end function get_cumulative_histogram

// get a normalised cumulative histogram array as a look-up table (LUT)
kernel void get_lut(global const int* CH, global int* LUT, const float mask)
{
	int id = get_global_id(0);
	LUT[id] = CH[id] * mask;
} // end function get_lut

// get the output image array according to the LUT
kernel void get_processed_image(global const uchar* input_image, global int* LUT, global uchar* output_image)
{
	int id = get_global_id(0);
	output_image[id] = LUT[input_image[id]];
} // end function get_processed_image