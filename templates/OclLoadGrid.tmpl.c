    // Create the data array in device memory for our calculation
    //
    cl_mem device_$arg_ref = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof($arg_ref[0]) * grid_size, NULL, NULL);
    if (!device_$arg_ref)
    {
        printf("Error: Failed to allocate device memory!\n");
        return err;
    }

    // Write our data set into the data array in device memory
    //
    err = clEnqueueWriteBuffer(commands, device_$arg_ref, CL_TRUE, 0, sizeof($arg_ref[0]) * grid_size, $arg_ref, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        return err;
    }

    // Set the arguments to our compute kernel
    //
    err  = clSetKernelArg(kernel, $arg_index, sizeof(cl_mem), &device_$arg_ref);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        return err;
    }
