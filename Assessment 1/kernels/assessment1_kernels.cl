/*
 * @Description: kernel code file for applying histogram equalisation on an RGB image (8-bit/16-bit)
 * @Version: 1.7.1.20200318
 * @Author: Arvin Zhao
 * @Date: 2020-03-08 15:29:21
 * @Last Editors: Arvin Zhao
 * @LastEditTime: 2020-03-18 13:07:31
 */

/*
get a histogram of an 8-bit iamge with a specified number of bins (basic version);
the sum of the elements should be equal to the total number of pixels
*/
kernel void get_H_8(global const uchar* image, global uint* H)
{
	uint id = get_global_id(0);

	/*
	compute the histogram;
	take a value from the input image as a bin index of the histogram
	*/
	atomic_inc(&H[image[id]]);
} // end function get_H_8

/*
get a histogram of a 16-bit image with a specified number of bins;
the sum of the elements should be equal to the total number of pixels
*/
kernel void get_H_16(global const ushort* image, global uint* H)
{
	uint id = get_global_id(0);

	/*
	compute the histogram;
	take a value from the input image as a bin index of the histogram
	*/
	atomic_inc(&H[image[id]]);
} // end function get_H_16

/*
get a histogram of an 8-bit image with a specified number of bins (optimised version - local memory is used);
the sum of the elements should be equal to the total number of pixels
*/
kernel void get_H_pro(global const uchar* image, global uint* H, local uint* H_local, const uint image_elements)
{
	uint id = get_global_id(0);
	int local_id = get_local_id(0);

	// initialise the local histogram to 0
	if (local_id < 256)
		H_local[local_id] = 0;

	barrier(CLK_LOCAL_MEM_FENCE); // wait for all local threads to finish the initialisation
	
	/*
	compute the local histogram;
	take a value from the input image as a bin index of the local histogram
	*/
	if (id < image_elements)
		atomic_inc(&H_local[image[id]]);

	barrier(CLK_LOCAL_MEM_FENCE); // wait for all local threads to finish computing the local histogram

	// write the local histogram out to the global histogram
	if (local_id < 256)
		atomic_add(&H[local_id], H_local[local_id]);
} // end function get_H_pro

/*
get a cumulative histogram (basic version);
the value of the last element should be equal to the total number of pixels
*/
kernel void get_CH(global const uint* H, global uint* CH, const int bin_count)
{
	int id = get_global_id(0);

	/*
	compute a cumulative histogram with an average histogram of the 3 colour channels' histograms;
	an average histogram is used for enabling basic histogram equalisation on both monochrome and colour images
	*/
	for (int i = id + 1; i < bin_count && id < bin_count; i++)
		atomic_add(&CH[i], H[id] / 3);
} // end function get_CH

/*
get a cumulative histogram (optimised version - a double-buffered version of the Hillis-Steele inclusive scan and local memory are used);
allow only for calculating partial reductions in a single work group separately;
for an 8-bit image, the number of local elements should be equal to 256;
for a 16-bit image, some helper kernels are needed to get a complete cumulative histogram;
the value of the last element in the complete cumulative histogram should be equal to the total number of pixels
*/
kernel void get_CH_pro(global const uint* H, global uint* CH, local uint* H_local, local uint* CH_local)
{
	int id = get_global_id(0);
	int local_id = get_local_id(0);
	local uint* scratch; // used for buffer swap

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
kernel void get_BS(global const uint* CH, global uint* BS, const uint local_elements)
{
	int id = get_global_id(0);
	BS[id] = CH[(id + 1) * local_elements - 1];
} // end method get_BS

/*
get scanned block sums by performing an exclusive scan (a version using exclusive serial scan based on atomic operations);
a helper kernel of the kernel for getting a cumulative histogram
*/
kernel void get_scanned_BS_1(global const uint* BS, global uint* BS_scanned)
{
	int id = get_global_id(0);
	int size = get_global_size(0);

	for (int i = id + 1; i < size && id < size; i++)
		atomic_add(&BS_scanned[i], BS[id]);
} // end function get_scanned_BS_1

/*
get scanned block sums by performing an exclusive scan (a version using Blelloch exclusive scan);
a helper kernel of the kernel for getting a cumulative histogram;
the following implementation has a limitation that the size of BS must be a power of 2;
please note that this implementation only works within a single workgroup as it uses barriers
*/
kernel void get_scanned_BS_2(global uint* BS)
{
	int id = get_global_id(0);
	int size = get_global_size(0);
	int temp;

	// up-sweep
	// "i" represents the stride
	for (int i = 1; i < size; i *= 2)
	{
		if (((id + 1) % (i * 2)) == 0)
			BS[id] += BS[id - i];

		barrier(CLK_GLOBAL_MEM_FENCE); // sync the step
	} // end for

	// down-sweep
	if (id == 0)
		BS[size - 1] = 0;

	barrier(CLK_GLOBAL_MEM_FENCE); // sync the step

	// "i" represents the stride
	for (int i = size / 2; i > 0; i /= 2)
	{
		if (((id + 1) % (i * 2)) == 0)
		{
			temp = BS[id];
			BS[id] += BS[id - i]; // reduce
			BS[id - i] = temp; // move
		} // end if

		barrier(CLK_GLOBAL_MEM_FENCE); // sync the step
	} // end for
} // end function get_scanned_BS_2

/*
get a complete cumulative histogram by adding block sums to corresponding blocks;
a helper kernel of the kernel for getting a cumulative histogram
*/
kernel void get_complete_CH(global const uint* BS_scanned, global uint* CH)
{
	CH[get_global_id(0)] += BS_scanned[get_group_id(0)];
} // end function get_complete_CH

/*
get a normalised cumulative histogram as a look-up table (LUT);
the value of the last element should be equal to "bin_count - 1"
*/
kernel void get_lut(global const uint* CH, global uint* LUT, const int bin_count, const int pixel_count)
{
	int id = get_global_id(0);

	if (id < bin_count)
		LUT[id] = ((ulong)CH[id] * (bin_count - 1)) / pixel_count; // use "ulong" to avoid integer overflow
} // end function get_lut

// get the output 8-bit image according to the LUT
kernel void get_processed_image_8(global const uchar* input_image, global const uint* LUT, global uchar* output_image)
{
	uint id = get_global_id(0);
	output_image[id] = LUT[input_image[id]];
} // end function get_processed_image_8

// get the output 16-bit image according to the LUT
kernel void get_processed_image_16(global const ushort* input_image, global const uint* LUT, global ushort* output_image)
{
	uint id = get_global_id(0);
	output_image[id] = LUT[input_image[id]];
} // end function get_processed_image_16