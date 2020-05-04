# UoL_Y3S2_PP

This repository contains code written for practising the knowledge gained from the course **Parallel Programming** during Semester 2 of Year 3 of BSc (Hons) Computer Science in UoL. Part of it is OpenCL tutorials modified from a version provided by the lecturers of the course. The original version is available in [the specified repository on GitHub](https://github.com/gcielniak/OpenCL-Tutorials).

## ATTENTION

1. By 4 May 2020, everything looks good with VS 2019 + C++ 14 + Intel SDK for OpenCL Applications 2020.0.245.

## Windows Setup

- OS & IDE suggestions: Windows 10, VS 2019 with C++ 11/14
- OpenCL SDK: the SDK enables you to develop and compile the OpenCL code. In our case, we use [Intel SDK for OpenCL Applications](https://software.intel.com/en-us/intel-opencl). You are not tied to that choice and can use SDKs by NVIDIA or AMD - just remember to make modifications in the project properties. Each SDK comes with a range of additional tools which make development of OpenCL programs easier.
- OpenCL runtime: the runtime drivers are necessary to run the OpenCL code on your hardware. Both NVIDIA and AMD GPUs have OpenCL runtime included with their card drivers. For CPUs, you will need to install a dedicated driver by [Intel](https://software.intel.com/en-us/articles/opencl-drivers) (ATTENTION: The installer of Intel CPU Runtime for OpenCL Applications is included in the installer of Intel SDK for OpenCL Applications, so you may not download the former if you already download the latter) or APP SDK for older AMD processors. It seems that AMD’s OpenCL support for newer CPU models was dropped unfortunately. You can check the existing OpenCL support on your PC using [TechPowerUp GPU-Z](https://www.techpowerup.com/gpuz/), [GPU Caps Viewer](http://www.ozone3d.net/gpu_caps_viewer/), or some other software with the similar functionality.
- Boost library (Tutorial 4 depends on it): install the recent [Boost library Windows binaries](https://sourceforge.net/projects/boost/files/boost-binaries/) (e.g. [boost_1_72_0](https://sourceforge.net/projects/boost/files/boost-binaries/1.72.0/boost_1_72_0-msvc-14.2-64.exe/download) for VS2019). Then, add two environmental variables in the command line specifying the location of the include and lib Boost directories. For example, with boost_1_72_0 the commands would look as follows: `setx BOOST_INCLUDEDIR "C:\local\boost_1_72_0"` and `setx BOOST_LIBRARYDIR "C:\local\boost_1_72_0\lib64-msvc-14.2"`.
- A useful reference if you are struggling to get going: [OpenCL on Windows](http://streamcomputing.eu/blog/2015-03-16/how-to-install-opencl-on-windows/).

## Usage

The following steps show a basic and suggested way to run an application.

1. Build/Rebuild the solution.
2. Using the command prompt, navigate to the directory containing your built project (e.g. `cd C:\SD\C++\UoL_Y3S2_PP\x64\Debug\Tutorial 1`) and run the EXE file (e.g. `"Tutorial 1.exe"` - ".exe" can be omitted).