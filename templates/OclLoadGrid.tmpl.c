    // Create the data array in device memory for our calculation
    //
    device_data = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof($arg_ref[0]) * grid_size, NULL, NULL);
    if (!device_data)
    {
        printf("Error: Failed to allocate device memory!\n");
        return err;
    }

    // Write our data set into the data array in device memory
    //
    err = clEnqueueWriteBuffer(commands, device_data, CL_TRUE, 0, sizeof($arg_ref[0]) * grid_size, $arg_ref, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        return err;
    }

    // Set the arguments to our compute kernel
    //
    err = 0;
    err  = clSetKernelArg(kernel, $arg_index, sizeof(cl_mem), &device_data);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        return err;
    }
