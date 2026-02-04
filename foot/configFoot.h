#ifndef CONFIGFOOT_H
#define CONFIGFOOT_H

#define VECTOR_DIMENSION 1000
#define NUM_LEVELS 31
#define MIN_LEVEL -1
#define MAX_LEVEL 1

#define WINDOW 100
#define N_GRAM_SIZE 3
#define DOWNSAMPLE 1
#define NUM_CLASSES 5

#define NUM_FEATURES 32
#define NORMALIZE 1

#define CUTTING_ANGLE_THRESHOLD 0.9
#define PRECOMPUTED_ITEM_MEMORY 0
#define USE_GENETIC_ITEM_MEMORY 1
#define OUTPUT_MODE OUTPUT_DETAILED

#define BIPOLAR_MODE 1


extern int output_mode;

//*************DONT CHANGE ANYTHING below this line */
#define OUTPUT_NONE 0        // No printing
#define OUTPUT_BASIC 1       // Print results
#define OUTPUT_DETAILED 2    // Print intermediate information
#define OUTPUT_DEBUG 3         // Print everything

#endif
