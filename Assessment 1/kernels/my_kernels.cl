// a histogram implementation considering the number of bins
kernel void get_histogram(global const uchar* image, global int* H, int nr_bins)
{
	int id = get_global_id(0);
	int bin_index = image[id]; // take value as a bin index
	
	if (id < nr_bins)
		H[id] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	atomic_inc(&H[bin_index]);
} // end function get_histogram

// (scan_add_atomic) simple exclusive serial scan based on atomic operations - sufficient for small number of elements
kernel void get_cumulative_histogram(global int* H, global int* CH)
{
	int id = get_global_id(0);
	int N = get_global_size(0);

	for (int i = id + 1; i < N; i++)
		atomic_add(&CH[i], H[id]);
} // end function get_cumulative_histogram

// TODO
kernel void get_lut(global int* CH, global double* mask, global int* LUT)
{
	int id = get_global_id(0);
	LUT[id] = CH[id] * mask[id];
} // end function get_lut

// TODO
kernel void get_processed_image(global const uchar* input_image, global int* LUT, global uchar* output_image)
{
	int id = get_global_id(0);
	output_image[id] = LUT[input_image[id]];
} // end function get_processed_image