// a simple OpenCL kernel which copies all pixels from A to B
kernel void identity(global const uchar* A, global uchar* B)
{
	int id = get_global_id(0);
	B[id] = A[id];
} // end function identity

// perform colour channel filtering
kernel void filter_r(global const uchar* A, global uchar* B)
{
	int id = get_global_id(0);
	int image_size = get_global_size(0) / 3; // each image consists of 3 colour channels
	int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue

	switch (colour_channel)
	{
		// the red colour component (R) is kept
		case 0:
			B[id] = A[id];
			break;

		// all other components (GB) are set to 0
		case 1:
		case 2:
			B[id] = 0;
	} // end switch-case
} // end function filter_r

// invert the intensity value of each pixel
kernel void invert(global const uchar* A, global uchar* B)
{
	int id = get_global_id(0);
	B[id] = 255 - A[id];
} // end function invert

// convert an input colour image into greyscale
kernel void rgb2grey(global const uchar* A, global uchar* B)
{
	int id = get_global_id(0);
	int image_size = get_global_size(0) / 3;

	if (id <= image_size - 1) // then id = id_r
	{
		int id_g = id + image_size;
		int id_b = id + image_size * 2;
		B[id] = 0.2126 * A[id] + 0.7152 * A[id_g] + 0.0722 * A[id_b];
		B[id_g] = B[id];
		B[id_b] = B[id];
	} // end if
} // end function rgb2grey

// a simple ND identity kernel
kernel void identityND(global const uchar* A, global uchar* B)
{
	int width = get_global_size(0); // image width in pixels
	int height = get_global_size(1); // image height in pixels
	int image_size = width * height; // image size in pixels
	int channels = get_global_size(2); // number of colour channels: 3 for RGB

	int x = get_global_id(0); // current x coordinate
	int y = get_global_id(1); // current y coordinate
	int c = get_global_id(2); // current colour channel

	int id = x + y * width + c * image_size; // global id in 1D space

	B[id] = A[id];
} // end function identityND

// a 2D averaging filter
kernel void avg_filterND(global const uchar* A, global uchar* B)
{
	int width = get_global_size(0); // image width in pixels
	int height = get_global_size(1); // image height in pixels
	int image_size = width * height; // image size in pixels
	int channels = get_global_size(2); // number of colour channels: 3 for RGB

	int x = get_global_id(0); // current x coordinate
	int y = get_global_id(1); // current y coordinate
	int c = get_global_id(2); // current colour channel

	int id = x + y * width + c * image_size; // global id in 1D space
	uint result = 0;

	// one way to handle the boundary conditions (just copy the original pixel)
	if (x == 0 || x == width - 1 || y == 0 || y == height - 1)
		result = A[id];
	else
	{
		for (int i = x - 1; i <= x + 1; i++)
			for (int j = y - 1; j <= y + 1; j++) 
				result += A[i + j * width + c * image_size];

		result /= 9;
	} // end if...else

	B[id] = (uchar)result;
} // end function avg_filterND

// a 2D 3x3 convolution kernel
kernel void convolutionND(global const uchar* A, global uchar* B, constant float* mask)
{
	int width = get_global_size(0); // image width in pixels
	int height = get_global_size(1); // image height in pixels
	int image_size = width * height; // image size in pixels
	int channels = get_global_size(2); // number of colour channels: 3 for RGB

	int x = get_global_id(0); // current x coordinate
	int y = get_global_id(1); // current y coordinate
	int c = get_global_id(2); // current colour channel

	int id = x + y * width + c * image_size; // global id in 1D space
	float result = 0;

	// one way to handle the boundary conditions (just copy the original pixel)
	if (x == 0 || x == width - 1 || y == 0 || y == height - 1)
		result = A[id];
	else
		for (int i = x - 1; i <= x + 1; i++)
			for (int j = y - 1; j <= y + 1; j++) 
				result += A[i + j * width + c * image_size] * mask[i - (x - 1) + j - (y - 1)];

	B[id] = (uchar)result;
} // end function convolutionND