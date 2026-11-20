#include "api/generation_program.h"

#include <algorithm>
#include <cmath>

namespace sd::api {
    static std::string expression_list(const std::vector<std::string>& values) {
        std::string result = "[";
        for (size_t i = 0; i < values.size(); ++i)
            result += (i ? ", " : "") + values[i];
        return result + "]";
    }

    void prepare_h3_references(GenerationTemplate& request, const std::string& first, const std::string& last) {
        const auto images           = bound_tensors(request.options, "ref_images");
        const auto videos           = bound_videos(request.options, "ref_videos");
        const auto audios           = bound_audios(request.options, "ref_audios");
        const bool references       = !images.empty() || !videos.empty() || !audios.empty();
        const auto guide_videos = bound_videos(request.options, "guide_videos");
        const auto guide_audios = bound_audios(request.options, "guide_audios");
        if (!references && first.empty() && last.empty() && guide_videos.empty() && guide_audios.empty())
            return;
        auto& prepare = request.fields["references"];
        std::vector<std::string> visuals, sounds, presentation, blocks;
        const auto encode_visual = [&](const std::string& pixels, const nlohmann::json& options = nlohmann::json::object()) {
            const auto name = "h3_visual_" + std::to_string(visuals.size());
            nlohmann::json parameters{{"name", name}, {"pixels", pixels}, {"encode_audio", false}, {"resize_video", false}, {"resize_image", false}};
            parameters.update(options);
            prepare += request.render("video_reference", parameters);
            visuals.push_back(name);
        };
        const auto encode_sound = [&](const sd_audio_t& audio, int64_t limit = 0) {
            if (!request.flags.at("decode_audio").get<bool>())
                throw dgl::Error(SD_STATUS_LOAD_FAILED, "MiniMax-H3 audio conditioning requires audio VAE weights");
            const auto name = "h3_sound_" + std::to_string(sounds.size());
            bind_tensor(request.program, name + "_samples", audio.data);
            const auto media = "{samples = " + name + "_samples, sample_rate = " + std::to_string(audio.sample_rate) + "}";
            prepare += request.render("video_reference", {{"encode_audio", true}, {"name", name}, {"sample_rate", audio.sample_rate}, {"limit", limit}});
            sounds.push_back(name);
            if (!limit)
                presentation.push_back("{audio = " + media + "}");
        };
        const int frames = request.values.at("frames").get<int>();
        std::vector<std::string> key_images;
        for (const auto& [pixels, index] : std::vector<std::pair<std::string, int>>{{first, 0}, {last, frames - 1}}) {
            if (pixels.empty())
                continue;
            blocks.push_back(literal({{"kind", "video"}, {"video", visuals.size()}, {"frame_index", index}}));
            encode_visual(pixels);
            key_images.push_back(pixels + "[:, :, 0]");
        }
        if (!references && !key_images.empty()) {
            request.fields["prompt_options"] += ", media = " + expression_list(key_images);
            request.fields["negative_prompt_options"] += ", media = " + expression_list(key_images);
        }
        const auto resolve_index = [&](int index) {
            if (index < 0)
                index += frames;
            if (index < 0 || index >= frames)
                throw dgl::Error(SD_STATUS_INVALID_ARGUMENT, "MiniMax-H3 guide frame index outside target");
            return index;
        };
        const auto guide_audio = [&](const sd_audio_t& audio, int start) {
            blocks.push_back(literal({{"kind", "audio"}, {"audio", sounds.size()}, {"frame_index", start}}));
            encode_sound(audio, std::llround(frames * (40.0 / 24)) - (5LL * start + 2) / 3);
        };
        for (size_t i = 0; i < guide_videos.size(); ++i) {
            const auto& video = guide_videos[i];
            const int start = resolve_index(request.video->guide_video_indices[i]);
            const int64_t available = sd_tensor_dim(video.data, 3);
            const int64_t count = available < 5 ? 1 : available - (available - 5) % 17;
            if (available < 1 || video.fps != 24 || count > frames - start)
                throw dgl::Error(SD_STATUS_INVALID_ARGUMENT, "MiniMax-H3 visual guide requires 24 fps frames that fit at the requested index");
            const auto name = "guide_video_" + std::to_string(i);
            bind_tensor(request.program, name, video.data);
            blocks.push_back(literal({{"kind", "video"}, {"video", visuals.size()}, {"frame_index", start}}));
            const auto channels = sd_tensor_dim(video.data, 0) == 2 ? ":1" : ":";
            const auto clip = name + "[" + channels + ", :, :, :" + std::to_string(count) + "]";
            const auto width = request.values.at("width").get<int>(), height = request.values.at("height").get<int>();
            const auto w = sd_tensor_dim(video.data, 1), h = sd_tensor_dim(video.data, 2);
            const auto scale = std::max(double(width) / w, double(height) / h);
            const auto size = literal({int64_t(std::ceil(w * scale)), int64_t(std::ceil(h * scale))});
            encode_visual("h3_guide_pixels(" + clip + ", width, height, " + size + ")");
            if (video.audio.data)
                guide_audio(video.audio, start);
        }
        for (size_t i = 0; i < guide_audios.size(); ++i)
            guide_audio(guide_audios[i], resolve_index(request.video->guide_audio_indices[i]));
        for (size_t i = 0; i < images.size(); ++i) {
            const auto name = "h3_image_" + std::to_string(i);
            bind_tensor(request.program, name, images[i]);
            const int source_width = int(sd_tensor_dim(images[i], 1)), source_height = int(sd_tensor_dim(images[i], 2));
            const double scale = std::min(1.0, std::sqrt(double(request.values.at("width").get<int>()) * request.values.at("height").get<int>() / (double(source_width) * source_height)));
            const int width = std::max(32, int(std::round(source_width * scale / 32)) * 32), height = std::max(32, int(std::round(source_height * scale / 32)) * 32);
            blocks.push_back(literal({{"kind", "image"}, {"video", visuals.size()}}));
            encode_visual(name + "_pixels", {{"resize_image", true}, {"source", name}, {"width", width}, {"height", height}});
            presentation.push_back("{image = " + name + "_pixels}");
        }
        for (size_t i = 0; i < videos.size(); ++i) {
            const auto& video = videos[i];
            const auto name   = "h3_video_" + std::to_string(i);
            bind_tensor(request.program, name, video.data);
            const int count = int(sd_tensor_dim(video.data, 3));
            int frames      = std::min(request.values.at("frames").get<int>(), int(std::lround(double(count) * 24 / video.fps)));
            if (frames < 5)
                throw dgl::Error(SD_STATUS_INVALID_ARGUMENT, "MiniMax-H3 reference video needs at least 5 frames at 24 fps");
            frames = (frames - 5) / 17 * 17 + 5;
            std::vector<int> indices;
            for (int frame = 0; frame < frames; ++frame)
                indices.push_back(std::min(count - 1, int(std::floor(double(frame) * video.fps / 24))));
            const int source_width = int(sd_tensor_dim(video.data, 1)), source_height = int(sd_tensor_dim(video.data, 2));
            const double ratio = double(source_width) / source_height;
            double w = ratio >= 1 ? 768 * ratio : 768, h = ratio >= 1 ? 768 : 768 / ratio;
            if (w * h > 768.0 * 1344) {
                const auto scale = std::sqrt(768.0 * 1344 / (w * h));
                w *= scale;
                h *= scale;
            }
            int width = std::max(32, int(std::round(w / 32)) * 32), height = std::max(32, int(std::round(h / 32)) * 32);
            if (source_width * source_height < width * height) {
                width  = std::max(32, int(std::round(double(source_width) / 32)) * 32);
                height = std::max(32, int(std::round(double(source_height) / 32)) * 32);
            }
            auto block = nlohmann::json{{"kind", video.audio.data ? "video_audio" : "video"}, {"video", visuals.size()}};
            if (video.audio.data) {
                block["audio"] = sounds.size();
                encode_sound(video.audio);
            }
            blocks.push_back(literal(block));
            encode_visual(name + "_pixels", {{"resize_video", true}, {"source", name}, {"indices", literal(indices)}, {"width", width}, {"height", height}});
            presentation.push_back("{video = {frames = " + name + "_pixels, fps_num = 24, fps_den = 1}, stride = 12}");
        }
        for (const auto& audio : audios) {
            blocks.push_back(literal({{"kind", "audio"}, {"audio", sounds.size()}}));
            encode_sound(audio);
        }
        std::string options = ", ref_latents = " + expression_list(visuals);
        if (!sounds.empty())
            options += ", ref_audios = " + expression_list(sounds);
        if (!blocks.empty())
            options += ", reference_blocks = " + expression_list(blocks);
        request.fields["predictor_options"] += options;
        request.fields["negative_predictor_options"] += options;
        if (!presentation.empty()) {
            request.fields["prompt_options"] += ", media = " + expression_list(presentation);
            request.fields["negative_prompt_options"] += ", media = " + expression_list(presentation);
        }
    }
}
