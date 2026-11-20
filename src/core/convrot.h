#ifndef __SD_CORE_CONVROT_H__
#define __SD_CORE_CONVROT_H__

#include <cmath>

inline void convrot_transform(float* values, int group) {
    const float scale = 1.f / std::sqrt(float(group));
    for (int i = 0; i < group; ++i)
        values[i] *= scale;
    for (int stride = 1; stride < group; stride *= 4)
        for (int base = 0; base < group; base += 4 * stride)
            for (int j = 0; j < stride; ++j) {
                const int i   = base + j;
                const float a = values[i], b = values[i + stride], c = values[i + 2 * stride], d = values[i + 3 * stride];
                values[i]              = a + b + c - d;
                values[i + stride]     = a + b - c + d;
                values[i + 2 * stride] = a - b + c + d;
                values[i + 3 * stride] = -a + b + c + d;
            }
}

#endif  // __SD_CORE_CONVROT_H__
