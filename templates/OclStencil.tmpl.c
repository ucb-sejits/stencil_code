#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>

int stencil($array_decl)
{
    const unsigned int grid_size = $grid_size;  // number of elements in array
    int err;                            // error code returned from api calls

    size_t global = $grid_size;             // global domain size for our calculation
    size_t local = 32;                  // local domain size for our calculation

    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel

    cl_mem device_data;                       // device memory used for the data array

    // Connect to a compute device
    //
    int gpu = 1;
    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        return err;
    }

    // Create a compute context
    //
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return err;
    }

    // Create a command commands
    //
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return err;
    }

    // Read the kernel into a string
    //
    FILE *kernelFile = fopen($kernel_path, "rb");
    if (kernelFile == NULL) {
        printf("Error: Coudn't open kernel file.\n");
        return err;
    }

    fseek(kernelFile, 0 , SEEK_END);
    long kernelFileSize = ftell(kernelFile);
    rewind(kernelFile);

    // Allocate memory to hold kernel
    //
    char *KernelSource = malloc(kernelFileSize*sizeof(char));
    memset(KernelSource, 0, kernelFileSize);
    if (KernelSource == NULL) {
        printf("Error: failed to allocate memory to hold kernel text.\n");
        return err;
    }

    // Read the kernel into memory
    //
    int result = fread(KernelSource, sizeof(char), kernelFileSize, kernelFile);
    if (result != kernelFileSize) {
        printf("Error: read fewer bytes of kernel text than expected.\n");
        return err;
    }
    fclose(kernelFile);

    // Create the compute program from the source buffer
    //
    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return err;
    }

    // Build the program executable
    //
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return err;
    }

    // Create the compute kernel in the program we wish to run
    //
    kernel = clCreateKernel(program, $kernel_name, &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        return err;
    }

    // // Create the data array in device memory for our calculation
    // //
    // device_data = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof($array_ref[0]) * grid_size, NULL, NULL);
    // if (!device_data)
    // {
    //     printf("Error: Failed to allocate device memory!\n");
    //     return err;
    // }

    // // Write our data set into the data array in device memory
    // //
    // err = clEnqueueWriteBuffer(commands, device_data, CL_TRUE, 0, sizeof($array_ref[0]) * grid_size, $array_ref, 0, NULL, NULL);
    // if (err != CL_SUCCESS)
    // {
    //     printf("Error: Failed to write to source array!\n");
    //     return err;
    // }

    // // Set the arguments to our compute kernel
    // //
    // err = 0;
    // err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_data);
    // if (err != CL_SUCCESS)
    // {
    //     printf("Error: Failed to set kernel arguments! %d\n", err);
    //     return err;
    // }
