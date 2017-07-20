/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sortingNetworks_common.h"



////////////////////////////////////////////////////////////////////////////////
// Validate sorted keys array (check for integrity and proper order)
////////////////////////////////////////////////////////////////////////////////
extern "C" uint validateSortedKeys(
    uint *resKey,
    uint arrayLength,
    uint dir
)
{

    if (arrayLength < 2)
    {
        printf("validateSortedKeys(): arrayLength too short, exiting...\n");
        return 1;
    }

    printf("...inspecting keys array: ");

    int flag = 1;

        if (dir)
        {
            //Ascending order
            for (uint i = 0; i < arrayLength - 1; i++)
                if (resKey[i + 1] < resKey[i])
                {
                    flag = 0;
                    printf("error index %d\n", i);
                    break;
                }
        }
        else
        {
            //Descending order
            for (uint i = 0; i < arrayLength - 1; i++)
                if (resKey[i + 1] > resKey[i])
                {
                    flag = 0;
                    printf("error index %d\n", i);
                    break;
                }
        }

        if (!flag)
        {
            printf("***Set result key array is not ordered properly***\n");
            goto brk;
        }

brk:

    if (flag) printf("OK\n");

    return flag;
}
