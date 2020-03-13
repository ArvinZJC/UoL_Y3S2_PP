/*
 * @Description: kernel code file for applying histogram equalisation on a 16-bit RGB image
 * @Version: 1.2.6.20200312
 * @Author: Arvin Zhao
 * @Date: 2020-03-10 18:08:13
 * @Last Editors: Arvin Zhao
 * @LastEditTime: 2020-03-12 12:09:36
 */

#define BIN_COUNT 65536

// get a histogram with a specified number of bins
kernel void get_H(global const ushort* image, global int* H)
{
	int id = get_global_id(0);
	
	// initialise the histogram to 0
	if (id < BIN_COUNT)
		H[id] = 0;

	barrier(CLK_GLOBAL_MEM_FENCE); // wait for all threads to finish the initialisation
	
	/*
	compute the histogram;
	take a value from the input image as a bin index of the histogram
	*/
	atomic_inc(&H[image[id]]);
} // end function get_H

// get a cumulative histogram (basic version)
kernel void get_CH(global const int* H, global int* CH)
{
	int id = get_global_id(0);

	/*
	compute a cumulative histogram with an average histogram of the 3 colour channels' histograms;
	an average histogram is used for enabling basic histogram equalisation on both monochrome and colour images
	*/
	for (int i = id + 1; i < BIN_COUNT && id < BIN_COUNT; i++)
		atomic_add(&CH[i], H[id] / 3);
} // end function get_CH

/*
get a cumulative histogram (optimised version);
a double-buffered version of the Hillis-Steele inclusive scan and local memory are used;
allow only for calculating partial reductions in a single work group separately;
need some helper kernels to get a complete cumulative histogram
*/
kernel void get_CH_pro(global const int* H, global int* CH, local int* H_local, local int* CH_local)
{
	int id = get_global_id(0);
	int local_id = get_local_id(0);
	local int* scratch; // used for buffer swap

	H_local[local_id] = H[id]; // cache all values of the histogram from global memory to local memory

	barrier(CLK_LOCAL_MEM_FENCE); // wait for all local threads to finish copying from global to local memory

	// "i" represents the stride
	for (int i = 1; i < get_local_size(0); i *= 2)
	{
		if (local_id >= i)
			CH_local[local_id] = H_local[local_id] + H_local[local_id - i];
		else
			CH_local[local_id] = H_local[local_id];

		barrier(CLK_LOCAL_MEM_FENCE);

		// buffer swap
		scratch = CH_local;
		CH_local = H_local;
		H_local = scratch;
	} // end for

	/*
	copy the cache to the output array to get a cumulative histogram;
	an average histogram is used for enabling basic histogram equalisation on both monochrome and colour images
	*/
	CH[id] = H_local[local_id] / 3;
} // end function get_CH_pro

/*
get block sums of a preliminary cumulative histogram;
a helper kernel of the kernel for getting a cumulative histogram
*/
kernel void get_BS(global const int* CH, global int* BS, const int local_elements)
{
	int id = get_global_id(0);
	BS[id] = CH[(id + 1) * local_elements - 1];
} // end method get_BS

/*
get scanned block sums by performing an exclusive scan;
a helper kernel of the kernel for getting a cumulative histogram
*/
kernel void get_scanned_BS(global const int* BS, global int* BS_scanned)
{
	int id = get_global_id(0);
	int size = get_global_size(0);

	for (int i = id + 1; i < size && id < size; i++)
		atomic_add(&BS_scanned[i], BS[id]); // TODO
} // end function get_scanned_BS

/*
get a complete cumulative histogram by adding block sums to corresponding blocks;
a helper kernel of the kernel for getting a cumulative histogram
*/
kernel void get_complete_CH(global const int* BS_scanned, global int* CH)
{
	CH[get_global_id(0)] += BS_scanned[get_group_id(0)];
} // end function get_complete_CH

// get a normalised cumulative histogram as a look-up table (LUT)
kernel void get_lut(global const int* CH, global int* LUT, const float mask)
{
	int id = get_global_id(0);

	if (id < BIN_COUNT)
		LUT[id] = CH[id] * mask;
} // end function get_lut

// get the output image according to the LUT
kernel void get_processed_image(global const ushort* input_image, global const int* LUT, global ushort* output_image)
{
	int id = get_global_id(0);
	output_image[id] = LUT[input_image[id]];
} // end function get_processed_image