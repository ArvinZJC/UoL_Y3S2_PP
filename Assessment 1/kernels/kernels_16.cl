/*
 * @Description: kernel code file for applying histogram equalisation on a 16-bit RGB image
 * @Version: 1.2.0.20200312
 * @Author: Arvin Zhao
 * @Date: 2020-03-10 18:08:13
 * @Last Editors: Arvin Zhao
 * @LastEditTime: 2020-03-12 12:09:36
 */

#define BIN_COUNT 65536

// get a histogram with a specified number of bins
kernel void get_histogram(global const ushort* image, global int* H)
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
kernel void get_processed_image(global const ushort* input_image, global int* LUT, global ushort* output_image)
{
	int id = get_global_id(0);
	output_image[id] = LUT[input_image[id]];
} // end function get_processed_image