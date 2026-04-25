#pragma once

#include <stdlib.h>

typedef struct {
  size_t rows;
  size_t cols;
  float *data;
} Matrix;