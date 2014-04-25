#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

int stencil($array_decl)
{
    const unsigned int grid_size = $grid_size;  // number of elements in array
    int err;                            // error code returned from api calls

    size_t global[$dim] = $global_size;             // global domain size for our calculation
    size_t local = 64;                  // local domain size for our calculation

    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    cl_event event;

    // Find number of platforms
    cl_uint num_platforms;

    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms <= 0)
    {
        printf("Error: Could not find a platform!\n");
        return err;
    }

    // Get all platforms
    cl_platform_id platforms[num_platforms];

    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to get all platforms!\n");
        return err;
    }

    // Secure a device
    for (int i = 0; i < num_platforms; i++)
    {
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);
        if (err == CL_SUCCESS)
            break;
    }
    if (device_id == NULL)
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
    commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
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
    char *KernelSource = (char *)calloc(sizeof(char), kernelFileSize);
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

    $load_params
    err = clSetKernelArg(kernel, $num_args, sizeof(float) * (local + 2) * (local + 2), NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create local memory!\n");
        return err;
    }

    // Allocate local memory
    // err = 0;
    // err  = clSetKernelArg(kernel, $local_mem_index, sizeof(char) * $local_mem_size, NULL);
    // if (err != CL_SUCCESS)
    // {
    //     printf("Error: Failed to set kernel arguments! %d\n", err);
    //     return err;
    // }

    // Get the maximum work group size for executing the kernel on the device

    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
         printf("Error: Failed to retrieve kernel work group info! %d\n", err);
         return err;
    }

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    err = clEnqueueNDRangeKernel(commands, kernel, $dim, 0, global, NULL, 0,
    NULL, &event);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        printf("Error: %d\n", err);
        return err;
    }

    // Wait for the command commands to get serviced before reading back results
    //
    clFinish(commands);

    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    *duration = (float)1.0e-9 * (time_end - time_start);

    // Read back the results from the device to verify the output
    //
    err = clEnqueueReadBuffer( commands, device_$output_ref, CL_TRUE, 0, sizeof($output_ref[0]) * $grid_size, $output_ref, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read data array! %d\n", err);
        return err;
    }

    // Shutdown and cleanup
    //
    $release_params
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return 0;
}
