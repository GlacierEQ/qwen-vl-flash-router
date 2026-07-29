/**
 * Qwen VL Vision Pipeline — CUDA Kernel for Image Patch Tokenization
 * Converts raw image patches to vision tokens via efficient GPU parallel processing.
 * Supports dynamic resolution with sliding window patch extraction.
 */

#include <cstdio>
#include <cmath>

#define PATCH_SIZE 14
#define HIDDEN_DIM 1152
#define MAX_PATCHES 4096

struct VisionPatch {
    float pixels[PATCH_SIZE * PATCH_SIZE * 3]; // RGB patch
    int patch_row;
    int patch_col;
};

struct VisionToken {
    float embedding[HIDDEN_DIM];
    float position_encoding;
    int spatial_index;
};

/**
 * CUDA kernel: Convert image patches to vision token embeddings
 * Each thread processes one patch through a simplified linear projection.
 */
__global__ void patch_to_token_kernel(
    const VisionPatch* __restrict__ patches,
    VisionToken* __restrict__ tokens,
    const float* __restrict__ projection_weights, // [PATCH_SIZE*PATCH_SIZE*3 x HIDDEN_DIM]
    int num_patches,
    int image_width,
    int image_height
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_patches) return;

    const VisionPatch& patch = patches[tid];
    VisionToken& token = tokens[tid];

    // Linear projection: patch_pixels -> embedding
    int input_dim = PATCH_SIZE * PATCH_SIZE * 3;
    for (int d = 0; d < HIDDEN_DIM; d++) {
        float sum = 0.0f;
        for (int i = 0; i < input_dim; i++) {
            sum += patch.pixels[i] * projection_weights[i * HIDDEN_DIM + d];
        }
        token.embedding[d] = sum;
    }

    // 2D rotary position encoding
    float row_pos = (float)patch.patch_row / (float)(image_height / PATCH_SIZE);
    float col_pos = (float)patch.patch_col / (float)(image_width / PATCH_SIZE);
    token.position_encoding = sinf(row_pos * 3.14159f) + cosf(col_pos * 3.14159f);
    token.spatial_index = patch.patch_row * (image_width / PATCH_SIZE) + patch.patch_col;
}

/**
 * CUDA kernel: Dynamic resolution token merging
 * Merges adjacent vision tokens when resolution exceeds budget.
 */
__global__ void token_merge_kernel(
    VisionToken* __restrict__ tokens,
    VisionToken* __restrict__ merged,
    const int* __restrict__ merge_map, // pairs of indices to merge
    int num_merges
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_merges) return;

    int idx_a = merge_map[tid * 2];
    int idx_b = merge_map[tid * 2 + 1];

    // Average the embeddings of merged tokens
    for (int d = 0; d < HIDDEN_DIM; d++) {
        merged[tid].embedding[d] = (tokens[idx_a].embedding[d] + tokens[idx_b].embedding[d]) * 0.5f;
    }
    merged[tid].position_encoding = (tokens[idx_a].position_encoding + tokens[idx_b].position_encoding) * 0.5f;
    merged[tid].spatial_index = tokens[idx_a].spatial_index;
}

/**
 * Host function: Process image through vision pipeline
 */
extern "C" int process_image_patches(
    int image_width, int image_height,
    int max_token_budget
) {
    int patches_h = image_height / PATCH_SIZE;
    int patches_w = image_width / PATCH_SIZE;
    int num_patches = patches_h * patches_w;

    if (num_patches > MAX_PATCHES) {
        // Need token merging
        int merge_ratio = (num_patches + max_token_budget - 1) / max_token_budget;
        printf("[VisionPipeline] %d patches -> merging %dx -> %d tokens\n",
               num_patches, merge_ratio, num_patches / merge_ratio);
        return num_patches / merge_ratio;
    }

    printf("[VisionPipeline] %d patches -> %d tokens (no merge needed)\n",
           num_patches, num_patches);
    return num_patches;
}
