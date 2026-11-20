#ifndef __SD_DGL_MINIMAX_H3_REFERENCE_CACHE_H__
#define __SD_DGL_MINIMAX_H3_REFERENCE_CACHE_H__

#include <algorithm>
#include <list>
#include <string>
#include <tuple>
#include <vector>

#include "config.h"
#include "core/tensor.hpp"
#include "core/util.h"

class MiniMaxH3ReferenceCache {
    using Settings = std::tuple<bool, bool, int, int, float, float, float, std::string, bool, bool>;

    static Settings encode_settings(const sd::TilingConfig& t, bool circular_x, bool circular_y) {
        return std::make_tuple(t.enabled, t.temporal_tiling, t.tile_size_w, t.tile_size_h,
                               t.target_overlap, t.rel_size_w, t.rel_size_h,
                               std::string(t.extra_tiling_args ? t.extra_tiling_args : ""), circular_x, circular_y);
    }

    using Image = sd::Tensor<float>;
    struct Entry {
        Image pixels;
        Image latent;
        Settings settings;
    };

    std::list<Entry> entries_;
    size_t capacity_ = 0;

public:
    void set_capacity(size_t capacity) {
        if (capacity == capacity_)
            return;
        capacity_ = capacity;
        clear();
    }

    void clear() { entries_.clear(); }

    template <typename Encode>
    Image get(const Image& source, const sd::TilingConfig& tiling,
              bool circular_x, bool circular_y, Encode encode) {
        constexpr size_t max_bytes = 64 * 1024 * 1024;
        if (capacity_ == 0 || source.values().size() > max_bytes / sizeof(float))
            return encode();
        const size_t pixel_bytes = source.values().size() * sizeof(float);
        auto settings = encode_settings(tiling, circular_x, circular_y);
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->settings == settings && it->pixels.shape() == source.shape() && it->pixels.values() == source.values()) {
                entries_.splice(entries_.begin(), entries_, it);
                LOG_VERBOSE("reusing MiniMax-H3 reference image latent");
                return it->latent;
            }
        }

        auto latent = encode();
        size_t retained = pixel_bytes + latent.values().size() * sizeof(float);
        if (latent.empty() || retained > max_bytes) {
            return latent;
        }
        if (entries_.size() == capacity_) {
            entries_.pop_back();
        }
        auto it = entries_.begin();
        while (it != entries_.end()) {
            retained += it->pixels.values().size() * sizeof(float) + it->latent.values().size() * sizeof(float);
            if (retained > max_bytes) {
                break;
            }
            ++it;
        }
        entries_.erase(it, entries_.end());
        entries_.push_front({source, latent, std::move(settings)});
        return latent;
    }
};

#endif
