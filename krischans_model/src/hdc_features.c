#include "hdc_features.h"
#include <stdlib.h>
#include <string.h>

int load_csv_labels(const char* path, int** out_labels, int* out_count)
{
    FILE* f = fopen(path, "r");
    if(!f) return -1;

    char line[256];
    fgets(line, sizeof(line), f); // skip header

    int count = 0;
    while(fgets(line, sizeof(line), f))
        count++;

    rewind(f);
    fgets(line, sizeof(line), f);

    int* labels = malloc(count * sizeof(int));

    for(int i = 0; i < count; i++){
        fgets(line, sizeof(line), f);
        labels[i] = atoi(line);
    }

    fclose(f);
    *out_labels = labels;
    *out_count = count;
    return 0;
}

int load_csv_features(const char* path, float*** out_feats, int* out_count)
{
    FILE* f = fopen(path, "r");
    if(!f) return -1;

    char line[4096];

    fgets(line, sizeof(line), f); // skip header

    int count = 0;
    while(fgets(line, sizeof(line), f))
        count++;

    rewind(f);
    fgets(line, sizeof(line), f);

    float** data = malloc(count * sizeof(float*));

    for(int i = 0; i < count; i++){
        fgets(line, sizeof(line), f);
        data[i] = malloc(FEATURE_DIM * sizeof(float));

        char* token = strtok(line, ",");
        for(int j = 0; j < FEATURE_DIM; j++){
            data[i][j] = atof(token);
            token = strtok(NULL, ",");
        }
    }

    fclose(f);
    *out_feats = data;
    *out_count = count;
    return 0;
}
