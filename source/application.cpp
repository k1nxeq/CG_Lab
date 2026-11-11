#include "application.hpp"
#include "geometry.hpp"
#include "math.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <imgui.h>
#include <lodepng.h>

namespace application {
namespace {

const auto& context = graphics::internal::context;
using lab_geometry::MeshData;
using lab_geometry::Vertex;
using lab_math::Mat4;
using lab_math::Vec3;

constexpr uint32_t max_point_lights = 16;
constexpr uint32_t max_spot_lights = 16;
constexpr uint32_t cone_segments = 24;
constexpr float icosahedron_radius = 1.0f;

// Максимальное число моделей в одном динамическом uniform-буфере (лекция #6, слайд 6)
constexpr uint32_t max_models = 1024;
// Размер staging-буфера для копирования из RAM в VRAM (лекция #5.2, слайд 11)
constexpr VkDeviceSize staging_buffer_size = 64 * 1024 * 1024; // 64 MB

enum class ProjectionType {
    Perspective = 0,
    Orthographic = 1,
};

struct SceneUniforms {
    float viewProjection[16];
    float cameraPosition[4];
    float directionalLightDirection[4];
    float directionalLightColor[4];
    float ambientLight[4];
    uint32_t lightCounts[4];
};

struct ModelUniforms {
    float model[16];
    float albedo[4];
    float specular[4];
    float params[4]; // x: smoothNormals, y: uvScale, z: emissiveStrength, w: unused
};

// Push-константы для нетривиального сэмплирования (лекция #6, слайд 16-18)
struct PipelineConsts {
    uint32_t nontrivialSampling = 0;
    float time = 0.0f;
};

struct GpuPointLight {
    float position[3];
    float intensity;
    float color[3];
    float padding;
};

struct GpuSpotLight {
    float position[3];
    float intensity;
    float direction[3];
    float cosInner;
    float color[3];
    float cosOuter;
};

static_assert(sizeof(SceneUniforms) == 144, "SceneUniforms must match std140 layout");
static_assert(sizeof(ModelUniforms) == 112, "ModelUniforms must match std140 layout");
static_assert(sizeof(GpuPointLight) == 32, "PointLight must match std430 layout");
static_assert(sizeof(GpuSpotLight) == 48, "SpotLight must match std430 layout");

struct GpuBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    void* mapped = nullptr;
    size_t size = 0;
};

struct GpuTexture {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
};

struct TextureMaterial {
    std::string name;
    GpuTexture albedo;
    GpuTexture specular;
    GpuTexture emissive;
    bool hasOwnSpecular = false;
    bool hasOwnEmissive = false;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
};

struct Mesh {
    GpuBuffer vertexBuffer;
    GpuBuffer indexBuffer;
    uint32_t triangleIndexCount = 0;
    uint32_t edgeFirstIndex = 0;
    uint32_t edgeIndexCount = 0;
};

struct Material {
    float albedo[3] = {1.0f, 1.0f, 1.0f};
    float specular[3] = {1.0f, 1.0f, 1.0f};
    float shininess = 32.0f;
    float uvScale = 1.0f;
    float emissiveStrength = 1.0f;
    int textureIndex = 0;
};

struct SceneObject {
    const char* name = "";
    const Mesh* mesh = nullptr;
    Vec3 position{};
    Vec3 rotationDegrees{};
    Vec3 scale{1.0f, 1.0f, 1.0f};
    Material material;
    bool smoothNormals = false;
    bool nontrivialSampling = false;
    bool animatable = false;
    bool animate = false;
    float phaseOffset = 0.0f;
    float animationMultiplier = 1.0f;
    uint32_t dynamicSlotIndex = 0;
};

struct PointLightData {
    Vec3 position{0.0f, -2.5f, 4.0f};
    float color[3] = {1.0f, 1.0f, 1.0f};
    float intensity = 10.0f;
    bool orbit = false;
};

struct SpotLightData {
    Vec3 position{-4.0f, -4.0f, 1.0f};
    Vec3 direction{0.5f, 0.5f, 0.7f};
    float color[3] = {1.0f, 1.0f, 1.0f};
    float intensity = 50.0f;
    float innerAngleDegrees = 12.0f;
    float outerAngleDegrees = 22.0f;
};

constexpr uint32_t object_count = 7;
constexpr uint32_t point_marker_base_slot = object_count;
constexpr uint32_t spot_marker_base_slot = point_marker_base_slot + max_point_lights;
constexpr uint32_t spot_cone_base_slot = spot_marker_base_slot + max_spot_lights;

Mesh icosahedronMesh;
Mesh cubeMesh;
Mesh coneMesh;
SceneObject objects[object_count];

std::vector<PointLightData> pointLights;
std::vector<SpotLightData> spotLights;

Vec3 directionalLightDirection{-0.3f, 1.0f, 0.5f};
float directionalLightColor[3] = {1.0f, 1.0f, 0.95f};
float directionalLightIntensity = 0.35f;
float ambientLightColor[3] = {0.35f, 0.4f, 0.5f};
float ambientLightIntensity = 0.2f;

constexpr Vec3 default_camera_position{0.0f, -2.0f, -1.0f};
Vec3 cameraPosition = default_camera_position;
float cameraPitchDegrees = -10.0f;
float cameraYawDegrees = 0.0f;
float cameraRollDegrees = 0.0f;
float cameraMoveSpeed = 4.0f;
float cameraMouseSensitivity = 0.15f;
bool cameraLookActive = false;

ProjectionType projectionType = ProjectionType::Perspective;
float perspectiveFovDegrees = 55.0f;
float orthographicHeight = 9.0f;
float nearPlane = 0.1f;
float farPlane = 60.0f;

bool showLightMarkers = true;
bool showSpotCones = true;
bool showEdges = false;
float markerSize = 0.2f;
float coneLength = 3.0f;

bool timeInitialized = false;
double previousWallTime = 0.0;
double animationTime = 0.0;
bool animationPlaying = true;
bool resetAnimationRequested = false;
float animationSpeed = 1.0f;
float trajectoryRadius = 0.6f;
float trajectoryVerticalAmplitude = 0.5f;
float trajectoryDepthAmplitude = 0.5f;
float trajectoryVerticalFrequency = 2.0f;
float trajectoryDepthFrequency = 0.5f;
Vec3 animatedRotationSpeedDegrees{20.0f, 30.0f, 40.0f};
Vec3 orbitCenter{0.0f, -1.0f, 6.0f};
float orbitSpeedDegrees = 40.0f;

// Объекты Vulkan для Лабораторной №3
VkCommandPool uploadCommandPool = VK_NULL_HANDLE;
VkCommandBuffer oneshotCommandBuffer = VK_NULL_HANDLE;
GpuBuffer stagingBuffer;
VkDeviceSize stagingOffset = 0;

GpuBuffer modelUniformBuffer; // Динамический UBO для всех моделей
VkDeviceSize modelUniformAlignment = sizeof(ModelUniforms);

VkSampler textureSampler = VK_NULL_HANDLE;
GpuTexture blackDummyTexture; // Служебная текстура 1x1 с одним чёрным пикселем
std::vector<TextureMaterial> materials;

VkDescriptorSetLayout sceneDescriptorSetLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout modelDescriptorSetLayout = VK_NULL_HANDLE;
VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
VkDescriptorSet sceneDescriptorSet = VK_NULL_HANDLE;
GpuBuffer sceneUniformBuffer;
GpuBuffer pointLightBuffer;
GpuBuffer spotLightBuffer;
VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
VkPipeline litPipeline = VK_NULL_HANDLE;
VkPipeline markerPipeline = VK_NULL_HANDLE;
VkPipeline linePipeline = VK_NULL_HANDLE;
VkPipeline edgePipeline = VK_NULL_HANDLE;
VkShaderModule litVertexShader = VK_NULL_HANDLE;
VkShaderModule litFragmentShader = VK_NULL_HANDLE;
VkShaderModule unlitVertexShader = VK_NULL_HANDLE;
VkShaderModule unlitFragmentShader = VK_NULL_HANDLE;
VkShaderModule edgeVertexShader = VK_NULL_HANDLE;
VkShaderModule edgeFragmentShader = VK_NULL_HANDLE;

// ---------------------------------------------------------------- Шум Перлина и генерация PNG файлов
namespace procedural_tex {

float fade(float t) { return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); }
float lerp(float a, float b, float t) { return a + t * (b - a); }

float hash2D(int x, int y, int seed) {
    int n = x * 374761393 + y * 668265263 + seed * 1442695041;
    n = (n ^ (n >> 13)) * 1274126177;
    return static_cast<float>((n ^ (n >> 16)) & 0x7fffffff) / static_cast<float>(0x7fffffff);
}

float valueNoise(float x, float y, int period, int seed) {
    int xi = static_cast<int>(std::floor(x));
    int yi = static_cast<int>(std::floor(y));
    float xf = x - std::floor(x);
    float yf = y - std::floor(y);

    int x0 = ((xi % period) + period) % period;
    int y0 = ((yi % period) + period) % period;
    int x1 = (x0 + 1) % period;
    int y1 = (y0 + 1) % period;

    float u = fade(xf);
    float v = fade(yf);

    float n00 = hash2D(x0, y0, seed);
    float n10 = hash2D(x1, y0, seed);
    float n01 = hash2D(x0, y1, seed);
    float n11 = hash2D(x1, y1, seed);

    return lerp(lerp(n00, n10, u), lerp(n01, n11, u), v);
}

float perlinFbm(float u, float v, int basePeriod, int octaves, int seed) {
    float total = 0.0f;
    float amplitude = 0.5f;
    float maxVal = 0.0f;
    int period = basePeriod;
    for (int i = 0; i < octaves; ++i) {
        total += valueNoise(u * period, v * period, period, seed + i * 31) * amplitude;
        maxVal += amplitude;
        amplitude *= 0.5f;
        period *= 2;
    }
    return total / maxVal;
}

uint8_t toByte(float c) {
    return static_cast<uint8_t>(std::clamp(c * 255.0f, 0.0f, 255.0f));
}

void savePngIfNotExists(const std::filesystem::path& path, uint32_t w, uint32_t h,
                        const std::vector<uint8_t>& rgba) {
    if (std::filesystem::exists(path)) return;
    lodepng_encode32_file(path.string().c_str(), rgba.data(), w, h);
}

void ensureTextureFilesOnDisk(const std::filesystem::path& dir) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);

    constexpr uint32_t S = 256;
    std::vector<uint8_t> albedo(S * S * 4);
    std::vector<uint8_t> spec(S * S * 4);
    std::vector<uint8_t> emis(S * S * 4);

    // 1. Кирпичная кладка (Brick)
    for (uint32_t y = 0; y < S; ++y) {
        for (uint32_t x = 0; x < S; ++x) {
            float u = static_cast<float>(x) / S;
            float v = static_cast<float>(y) / S;
            float row = v * 8.0f;
            int rowIdx = static_cast<int>(std::floor(row));
            float col = u * 4.0f + (rowIdx % 2 == 1 ? 0.5f : 0.0f);
            float fx = col - std::floor(col);
            float fy = row - std::floor(row);
            bool mortar = (fx < 0.06f || fx > 0.94f || fy < 0.10f || fy > 0.90f);
            float n = perlinFbm(u, v, 16, 4, 101);
            size_t idx = (y * S + x) * 4;
            if (mortar) {
                albedo[idx + 0] = toByte(0.62f + 0.1f * n);
                albedo[idx + 1] = toByte(0.60f + 0.1f * n);
                albedo[idx + 2] = toByte(0.56f + 0.1f * n);
                spec[idx + 0] = spec[idx + 1] = spec[idx + 2] = 20;
            } else {
                float brickTint = hash2D(static_cast<int>( std::floor(col) ), rowIdx, 77) * 0.18f;
                albedo[idx + 0] = toByte(0.65f + brickTint + 0.15f * n);
                albedo[idx + 1] = toByte(0.24f + 0.5f * brickTint + 0.08f * n);
                albedo[idx + 2] = toByte(0.18f + 0.05f * n);
                uint8_t sp = toByte(0.25f + 0.35f * n);
                spec[idx + 0] = spec[idx + 1] = spec[idx + 2] = sp;
            }
            albedo[idx + 3] = spec[idx + 3] = 255;
        }
    }
    savePngIfNotExists(dir / "brick_albedo.png", S, S, albedo);
    savePngIfNotExists(dir / "brick_specular.png", S, S, spec);

    // 2. Лавовый камень с расщелинами из шума Перлина (Lava Rock: Albedo + Specular + Emissive)
    for (uint32_t y = 0; y < S; ++y) {
        for (uint32_t x = 0; x < S; ++x) {
            float u = static_cast<float>(x) / S;
            float v = static_cast<float>(y) / S;
            float n1 = perlinFbm(u, v, 6, 5, 202);
            float n2 = perlinFbm(u + 0.3f, v + 0.7f, 10, 4, 303);
            float ridge = 1.0f - std::fabs(n1 * 2.0f - 1.0f);
            float crack = std::pow(std::clamp((ridge - 0.72f) / 0.28f, 0.0f, 1.0f), 2.0f);

            size_t idx = (y * S + x) * 4;
            float rock = 0.12f + 0.22f * n2;
            albedo[idx + 0] = toByte(lerp(rock, 1.0f, crack));
            albedo[idx + 1] = toByte(lerp(rock * 0.95f, 0.45f, crack));
            albedo[idx + 2] = toByte(lerp(rock * 1.05f, 0.05f, crack));
            albedo[idx + 3] = 255;

            // Блестят только гладкие обсидиановые участки камня вне трещин
            uint8_t sp = toByte((1.0f - crack) * std::pow(n2, 1.5f) * 0.9f);
            spec[idx + 0] = spec[idx + 1] = spec[idx + 2] = sp;
            spec[idx + 3] = 255;

            // Emissive из шума Перлина: светится только лава в трещинах!
            emis[idx + 0] = toByte(crack * 1.0f);
            emis[idx + 1] = toByte(crack * crack * 0.55f);
            emis[idx + 2] = toByte(crack * crack * crack * 0.1f);
            emis[idx + 3] = 255;
        }
    }
    savePngIfNotExists(dir / "lava_albedo.png", S, S, albedo);
    savePngIfNotExists(dir / "lava_specular.png", S, S, spec);
    savePngIfNotExists(dir / "lava_emissive.png", S, S, emis);

    // 3. Кибер-панель (Sci-Fi Panel: Albedo + Specular + Emissive)
    for (uint32_t y = 0; y < S; ++y) {
        for (uint32_t x = 0; x < S; ++x) {
            float u = static_cast<float>(x) / S;
            float v = static_cast<float>(y) / S;
            float gx = std::fabs(u * 4.0f - std::floor(u * 4.0f) - 0.5f) * 2.0f;
            float gy = std::fabs(v * 4.0f - std::floor(v * 4.0f) - 0.5f) * 2.0f;
            bool border = (gx > 0.88f || gy > 0.88f);
            bool neonLine = (gx > 0.80f && gx <= 0.85f) || (gy > 0.80f && gy <= 0.85f);
            float n = perlinFbm(u, v, 8, 4, 404);

            size_t idx = (y * S + x) * 4;
            if (neonLine) {
                albedo[idx + 0] = 30;
                albedo[idx + 1] = 220;
                albedo[idx + 2] = 255;
                spec[idx + 0] = spec[idx + 1] = spec[idx + 2] = 255;
                float pulse = 0.5f + 0.5f * n;
                emis[idx + 0] = toByte(0.1f * pulse);
                emis[idx + 1] = toByte(0.85f * pulse);
                emis[idx + 2] = toByte(1.0f * pulse);
            } else if (border) {
                albedo[idx + 0] = albedo[idx + 1] = albedo[idx + 2] = 35;
                spec[idx + 0] = spec[idx + 1] = spec[idx + 2] = 60;
                emis[idx + 0] = emis[idx + 1] = emis[idx + 2] = 0;
            } else {
                albedo[idx + 0] = toByte(0.20f + 0.1f * n);
                albedo[idx + 1] = toByte(0.24f + 0.1f * n);
                albedo[idx + 2] = toByte(0.30f + 0.1f * n);
                uint8_t sp = toByte(0.65f + 0.35f * n);
                spec[idx + 0] = spec[idx + 1] = spec[idx + 2] = sp;
                emis[idx + 0] = emis[idx + 1] = emis[idx + 2] = 0;
            }
            albedo[idx + 3] = spec[idx + 3] = emis[idx + 3] = 255;
        }
    }
    savePngIfNotExists(dir / "scifi_albedo.png", S, S, albedo);
    savePngIfNotExists(dir / "scifi_specular.png", S, S, spec);
    savePngIfNotExists(dir / "scifi_emissive.png", S, S, emis);

    // 4. Мраморная плитка (Marble Tiles: Albedo + Specular, без Emissive)
    for (uint32_t y = 0; y < S; ++y) {
        for (uint32_t x = 0; x < S; ++x) {
            float u = static_cast<float>(x) / S;
            float v = static_cast<float>(y) / S;
            int tx = static_cast<int>(std::floor(u * 4.0f));
            int ty = static_cast<int>(std::floor(v * 4.0f));
            bool darkTile = ((tx + ty) % 2) == 1;
            float fx = u * 4.0f - std::floor(u * 4.0f);
            float fy = v * 4.0f - std::floor(v * 4.0f);
            bool grout = (fx < 0.04f || fx > 0.96f || fy < 0.04f || fy > 0.96f);

            float n = perlinFbm(u, v, 8, 4, 505);
            float vein = 0.5f + 0.5f * std::sin((u + v) * 18.0f + n * 6.0f);

            size_t idx = (y * S + x) * 4;
            if (grout) {
                albedo[idx + 0] = albedo[idx + 1] = albedo[idx + 2] = 50;
                spec[idx + 0] = spec[idx + 1] = spec[idx + 2] = 15;
            } else if (darkTile) {
                float c = 0.18f + 0.18f * vein;
                albedo[idx + 0] = toByte(c * 0.9f);
                albedo[idx + 1] = toByte(c * 0.95f);
                albedo[idx + 2] = toByte(c * 1.1f);
                spec[idx + 0] = spec[idx + 1] = spec[idx + 2] = toByte(0.75f + 0.2f * vein);
            } else {
                float c = 0.72f + 0.22f * vein;
                albedo[idx + 0] = toByte(c);
                albedo[idx + 1] = toByte(c * 0.98f);
                albedo[idx + 2] = toByte(c * 0.94f);
                spec[idx + 0] = spec[idx + 1] = spec[idx + 2] = toByte(0.85f + 0.15f * vein);
            }
            albedo[idx + 3] = spec[idx + 3] = 255;
        }
    }
    savePngIfNotExists(dir / "marble_albedo.png", S, S, albedo);
    savePngIfNotExists(dir / "marble_specular.png", S, S, spec);

    // 5. Матовое дерево (Wood Planks: только Albedo, без Specular и без Emissive)
    for (uint32_t y = 0; y < S; ++y) {
        for (uint32_t x = 0; x < S; ++x) {
            float u = static_cast<float>(x) / S;
            float v = static_cast<float>(y) / S;
            float n = perlinFbm(u * 0.5f, v * 2.0f, 8, 4, 606);
            float grain = 0.5f + 0.5f * std::sin(u * 40.0f + n * 10.0f);
            float plankGap = std::fabs(u * 4.0f - std::floor(u * 4.0f));
            bool gap = (plankGap < 0.03f || plankGap > 0.97f);

            size_t idx = (y * S + x) * 4;
            if (gap) {
                albedo[idx + 0] = 45;
                albedo[idx + 1] = 30;
                albedo[idx + 2] = 18;
            } else {
                albedo[idx + 0] = toByte(lerp(0.42f, 0.68f, grain));
                albedo[idx + 1] = toByte(lerp(0.24f, 0.44f, grain));
                albedo[idx + 2] = toByte(lerp(0.11f, 0.24f, grain));
            }
            albedo[idx + 3] = 255;
        }
    }
    savePngIfNotExists(dir / "wood_albedo.png", S, S, albedo);
}

} // namespace procedural_tex

// ---------------------------------------------------------------- Работа с буферами и памятью GPU

bool createHostBuffer(size_t size, VkBufferUsageFlags usage, GpuBuffer& result) {
    const size_t alignedSize = (size + 0xf) & ~static_cast<size_t>(0xf);
    const VkBufferCreateInfo bufferInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = alignedSize,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    const VmaAllocationCreateInfo allocationInfo = {
        .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                 VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };
    VmaAllocationInfo mappedInfo{};
    if (vmaCreateBuffer(context.allocator, &bufferInfo, &allocationInfo,
                        &result.buffer, &result.allocation, &mappedInfo) != VK_SUCCESS) {
        std::cerr << "Failed to create and allocate host Vulkan buffer\n";
        return false;
    }
    result.mapped = mappedInfo.pMappedData;
    result.size = alignedSize;
    return result.mapped != nullptr;
}

// Выделение буфера в VRAM (GPU) для вершин и индексов (лекция #6, слайд 13)
bool createDeviceLocalBuffer(size_t size, VkBufferUsageFlags usage, GpuBuffer& result) {
    const VkBufferCreateInfo bufferInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    const VmaAllocationCreateInfo allocationInfo = {
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
    };
    if (vmaCreateBuffer(context.allocator, &bufferInfo, &allocationInfo,
                        &result.buffer, &result.allocation, nullptr) != VK_SUCCESS) {
        std::cerr << "Failed to create and allocate VRAM Vulkan buffer\n";
        return false;
    }
    result.mapped = nullptr;
    result.size = size;
    return true;
}

void writeToBuffer(GpuBuffer& buffer, const void* data, size_t size) {
    if (size == 0) return;
    std::memcpy(buffer.mapped, data, size);
    vmaFlushAllocation(context.allocator, buffer.allocation, 0, VK_WHOLE_SIZE);
}

void destroyBuffer(GpuBuffer& buffer) {
    if (buffer.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(context.allocator, buffer.buffer, buffer.allocation);
    }
    buffer = {};
}

void destroyTexture(GpuTexture& tex) {
    if (tex.view != VK_NULL_HANDLE) {
        vkDestroyImageView(context.device, tex.view, nullptr);
    }
    if (tex.image != VK_NULL_HANDLE) {
        vmaDestroyImage(context.allocator, tex.image, tex.allocation);
    }
    tex = {};
}

VkDeviceSize allocateFromStaging(size_t bytes, size_t alignment = 16) {
    stagingOffset = (stagingOffset + alignment - 1) & ~(alignment - 1);
    if (stagingOffset + bytes > staging_buffer_size) {
        std::cerr << "Staging buffer overflow!\n";
        return 0;
    }
    const VkDeviceSize offset = stagingOffset;
    stagingOffset += bytes;
    return offset;
}

bool beginUploadCommands() {
    const VkCommandPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = context.graphics_queue_index,
    };
    if (vkCreateCommandPool(context.device, &poolInfo, nullptr, &uploadCommandPool) != VK_SUCCESS) {
        std::cerr << "Failed to create upload command pool\n";
        return false;
    }
    const VkCommandBufferAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = uploadCommandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (vkAllocateCommandBuffers(context.device, &allocInfo, &oneshotCommandBuffer) != VK_SUCCESS) {
        std::cerr << "Failed to allocate one-shot command buffer\n";
        return false;
    }
    if (!createHostBuffer(staging_buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, stagingBuffer)) {
        std::cerr << "Failed to create staging buffer\n";
        return false;
    }
    stagingOffset = 0;

    const VkCommandBufferBeginInfo beginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (vkBeginCommandBuffer(oneshotCommandBuffer, &beginInfo) != VK_SUCCESS) {
        std::cerr << "Failed to begin one-shot command buffer\n";
        return false;
    }
    return true;
}

bool endAndSubmitUploadCommands() {
    if (vkEndCommandBuffer(oneshotCommandBuffer) != VK_SUCCESS) {
        std::cerr << "Failed to end one-shot command buffer\n";
        return false;
    }
    vmaFlushAllocation(context.allocator, stagingBuffer.allocation, 0, VK_WHOLE_SIZE);

    VkFence oneshotFence = VK_NULL_HANDLE;
    const VkFenceCreateInfo fenceInfo = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    };
    if (vkCreateFence(context.device, &fenceInfo, nullptr, &oneshotFence) != VK_SUCCESS) {
        std::cerr << "Failed to create one-shot fence\n";
        return false;
    }
    const VkSubmitInfo submitInfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &oneshotCommandBuffer,
    };
    vkQueueSubmit(context.graphics_queue, 1, &submitInfo, oneshotFence);
    vkWaitForFences(context.device, 1, &oneshotFence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(context.device, oneshotFence, nullptr);
    return true;
}

// Создание VkImage, VkImageView и запись команд копирования из staging в VRAM (лекция #5.2, слайды 6-20)
bool createTextureFromPixels(const uint8_t* pixels, uint32_t width, uint32_t height, GpuTexture& outTex) {
    outTex.width = width;
    outTex.height = height;

    const VkImageCreateInfo imageInfo = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .extent = {
            .width = width,
            .height = height,
            .depth = 1,
        },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    const VmaAllocationCreateInfo allocInfo = {
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
    };

    if (vmaCreateImage(context.allocator, &imageInfo, &allocInfo,
                       &outTex.image, &outTex.allocation, nullptr) != VK_SUCCESS) {
        std::cerr << "Failed to create and allocate Vulkan image\n";
        return false;
    }

    const VkImageViewCreateInfo viewInfo = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = outTex.image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    if (vkCreateImageView(context.device, &viewInfo, nullptr, &outTex.view) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan image view\n";
        return false;
    }

    const size_t pixelBytes = static_cast<size_t>(width) * height * 4;
    const VkDeviceSize imageOffset = allocateFromStaging(pixelBytes, 16);
    std::memcpy(static_cast<uint8_t*>(stagingBuffer.mapped) + imageOffset, pixels, pixelBytes);

    // Барьер UNDEFINED -> TRANSFER_DST_OPTIMAL
    const VkImageMemoryBarrier undefToTransfer = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = outTex.image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    vkCmdPipelineBarrier(oneshotCommandBuffer,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &undefToTransfer);

    const VkBufferImageCopy copyRegion = {
        .bufferOffset = imageOffset,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .imageOffset = {0, 0, 0},
        .imageExtent = {width, height, 1},
    };

    vkCmdCopyBufferToImage(oneshotCommandBuffer,
                           stagingBuffer.buffer,
                           outTex.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1, &copyRegion);

    // Барьер TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
    const VkImageMemoryBarrier transferToSampled = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = outTex.image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    vkCmdPipelineBarrier(oneshotCommandBuffer,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &transferToSampled);

    return true;
}

// Загрузка PNG-файла из файловой системы при помощи lodepng (лекция #5.2, слайд 4)
bool loadTextureFromFile(const std::filesystem::path& filePath, GpuTexture& outTex) {
    uint32_t width = 0;
    uint32_t height = 0;
    uint8_t* pixels = nullptr;

    const unsigned error = lodepng_decode32_file(&pixels, &width, &height, filePath.string().c_str());
    if (error != 0 || pixels == nullptr) {
        std::cerr << "Failed to decode PNG image file (" << filePath.string() << "): "
                  << lodepng_error_text(error) << '\n';
        return false;
    }

    const bool ok = createTextureFromPixels(pixels, width, height, outTex);
    std::free(pixels);
    return ok;
}

bool createSampler() {
    const VkSamplerCreateInfo samplerInfo = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
    };
    if (vkCreateSampler(context.device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan image sampler\n";
        return false;
    }
    return true;
}

// Копирование вершин и индексов через staging-буфер в VRAM (лекция #6, слайды 13-15)
bool createMesh(const MeshData& data, Mesh& mesh) {
    const size_t vertexBytes = sizeof(Vertex) * data.vertices.size();
    const size_t indexBytes = sizeof(uint32_t) * data.indices.size();
    mesh.triangleIndexCount = data.triangleIndexCount;
    mesh.edgeFirstIndex = data.edgeFirstIndex;
    mesh.edgeIndexCount = data.edgeIndexCount;

    if (!createDeviceLocalBuffer(vertexBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, mesh.vertexBuffer) ||
        !createDeviceLocalBuffer(indexBytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, mesh.indexBuffer)) {
        return false;
    }

    const VkDeviceSize verticesOffset = allocateFromStaging(vertexBytes, 16);
    std::memcpy(static_cast<uint8_t*>(stagingBuffer.mapped) + verticesOffset,
                data.vertices.data(), vertexBytes);

    const VkBufferCopy vertexCopy = {
        .srcOffset = verticesOffset,
        .dstOffset = 0,
        .size = vertexBytes,
    };
    vkCmdCopyBuffer(oneshotCommandBuffer, stagingBuffer.buffer, mesh.vertexBuffer.buffer, 1, &vertexCopy);

    const VkDeviceSize indicesOffset = allocateFromStaging(indexBytes, 16);
    std::memcpy(static_cast<uint8_t*>(stagingBuffer.mapped) + indicesOffset,
                data.indices.data(), indexBytes);

    const VkBufferCopy indexCopy = {
        .srcOffset = indicesOffset,
        .dstOffset = 0,
        .size = indexBytes,
    };
    vkCmdCopyBuffer(oneshotCommandBuffer, stagingBuffer.buffer, mesh.indexBuffer.buffer, 1, &indexCopy);

    return true;
}

bool createMeshesAndTextures() {
    if (!createSampler()) return false;
    if (!beginUploadCommands()) return false;

    // 1. Записываем команды копирования вершин и индексов в VRAM
    if (!createMesh(lab_geometry::makeIcosahedron(icosahedron_radius), icosahedronMesh) ||
        !createMesh(lab_geometry::makeCube(), cubeMesh) ||
        !createMesh(lab_geometry::makeCone(cone_segments), coneMesh)) {
        return false;
    }

    // 2. Создаём служебную текстуру с одним чёрным пикселем (для отсутствующих Specular / Emissive)
    const uint8_t blackPixel[4] = {0, 0, 0, 255};
    if (!createTextureFromPixels(blackPixel, 1, 1, blackDummyTexture)) {
        return false;
    }

    // 3. Генерируем (при отсутствии) и загружаем PNG текстуры из папки textures/
    const std::filesystem::path texDir = "textures";
    procedural_tex::ensureTextureFilesOnDisk(texDir);

    struct MaterialSpec {
        const char* name;
        const char* albedoFile;
        const char* specularFile; // nullptr -> используется blackDummyTexture
        const char* emissiveFile; // nullptr -> используется blackDummyTexture
    };

    const MaterialSpec specs[] = {
        {"Lava Rock (Perlin Emissive + Specular)", "lava_albedo.png", "lava_specular.png", "lava_emissive.png"},
        {"Sci-Fi Cyber Panel (Neon Emissive + Specular)", "scifi_albedo.png", "scifi_specular.png", "scifi_emissive.png"},
        {"Brick Wall (Albedo + Specular only)", "brick_albedo.png", "brick_specular.png", nullptr},
        {"Marble Checkerboard (Albedo + Specular only)", "marble_albedo.png", "marble_specular.png", nullptr},
        {"Matte Wood Planks (Albedo only)", "wood_albedo.png", nullptr, nullptr},
    };

    materials.resize(sizeof(specs) / sizeof(specs[0]));
    for (size_t i = 0; i < materials.size(); ++i) {
        TextureMaterial& mat = materials[i];
        mat.name = specs[i].name;
        if (!loadTextureFromFile(texDir / specs[i].albedoFile, mat.albedo)) {
            return false;
        }
        if (specs[i].specularFile != nullptr) {
            if (!loadTextureFromFile(texDir / specs[i].specularFile, mat.specular)) return false;
            mat.hasOwnSpecular = true;
        }
        if (specs[i].emissiveFile != nullptr) {
            if (!loadTextureFromFile(texDir / specs[i].emissiveFile, mat.emissive)) return false;
            mat.hasOwnEmissive = true;
        }
    }

    // Отправляем буфер команд на исполнение и ждём на VkFence
    return endAndSubmitUploadCommands();
}

bool loadShaderModule(const char* name, VkShaderModule& shaderModule) {
    const std::string candidates[] = {
        std::string("shaders/") + name,
        std::string("../shaders/") + name,
        std::string("../../shaders/") + name,
    };
    std::ifstream file;
    for (const std::string& path : candidates) {
        file.open(path, std::ios::binary | std::ios::ate);
        if (file) break;
        file.clear();
    }
    if (!file) {
        std::cerr << "Failed to open shader: " << name << " (compile shaders with glslc first)\n";
        return false;
    }
    const std::streamsize fileSize = file.tellg();
    std::vector<uint32_t> code(static_cast<size_t>(fileSize) / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()), fileSize);
    file.close();
    const VkShaderModuleCreateInfo shaderInfo = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = static_cast<size_t>(fileSize),
        .pCode = code.data(),
    };
    if (vkCreateShaderModule(context.device, &shaderInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        std::cerr << "Failed to create shader module: " << name << '\n';
        return false;
    }
    return true;
}

bool createDescriptorLayouts() {
    const VkDescriptorSetLayoutBinding sceneBindings[] = {
        { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
        { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
        { .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
    };
    const VkDescriptorSetLayoutCreateInfo sceneLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = sizeof(sceneBindings) / sizeof(sceneBindings[0]),
        .pBindings = sceneBindings,
    };
    if (vkCreateDescriptorSetLayout(context.device, &sceneLayoutInfo, nullptr,
                                    &sceneDescriptorSetLayout) != VK_SUCCESS) {
        std::cerr << "Failed to create scene descriptor set layout\n";
        return false;
    }

    // Set = 1: Динамический uniform буфер (binding = 0) + 3 сэмплера текстур (bindings = 1, 2, 3)
    const VkDescriptorSetLayoutBinding modelBindings[] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
            .binding = 3,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
    };
    const VkDescriptorSetLayoutCreateInfo modelLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = sizeof(modelBindings) / sizeof(modelBindings[0]),
        .pBindings = modelBindings,
    };
    if (vkCreateDescriptorSetLayout(context.device, &modelLayoutInfo, nullptr,
                                    &modelDescriptorSetLayout) != VK_SUCCESS) {
        std::cerr << "Failed to create model descriptor set layout\n";
        return false;
    }
    return true;
}

bool allocateDescriptorSet(VkDescriptorSetLayout layout, VkDescriptorSet& descriptorSet) {
    const VkDescriptorSetAllocateInfo allocateInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &layout,
    };
    if (vkAllocateDescriptorSets(context.device, &allocateInfo, &descriptorSet) != VK_SUCCESS) {
        std::cerr << "Failed to allocate descriptor set\n";
        return false;
    }
    return true;
}

bool createDescriptorPoolAndSets() {
    // Вычисляем выравненный сдвиг динамического uniform-буфера (лекция #6, слайд 5)
    VkPhysicalDeviceProperties physicalDeviceProps{};
    vkGetPhysicalDeviceProperties(context.physical_device, &physicalDeviceProps);
    const VkDeviceSize align = physicalDeviceProps.limits.minUniformBufferOffsetAlignment;
    modelUniformAlignment = (sizeof(ModelUniforms) + align - 1) & ~(align - 1);

    if (!createHostBuffer(modelUniformAlignment * max_models,
                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, modelUniformBuffer)) {
        return false;
    }

    const uint32_t materialCount = static_cast<uint32_t>(materials.size());
    const uint32_t totalSets = 1 + materialCount;

    const VkDescriptorPoolSize poolSizes[] = {
        { .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 4 },
        { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 4 },
        { .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = materialCount + 4 },
        { .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = materialCount * 3 + 4 },
    };

    const VkDescriptorPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = totalSets + 4,
        .poolSizeCount = sizeof(poolSizes) / sizeof(poolSizes[0]),
        .pPoolSizes = poolSizes,
    };
    if (vkCreateDescriptorPool(context.device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        std::cerr << "Failed to create application descriptor pool\n";
        return false;
    }

    if (!createHostBuffer(sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                          sceneUniformBuffer) ||
        !createHostBuffer(sizeof(GpuPointLight) * max_point_lights,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, pointLightBuffer) ||
        !createHostBuffer(sizeof(GpuSpotLight) * max_spot_lights,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, spotLightBuffer)) {
        return false;
    }

    if (!allocateDescriptorSet(sceneDescriptorSetLayout, sceneDescriptorSet)) {
        return false;
    }

    const VkDescriptorBufferInfo sceneInfo = {
        .buffer = sceneUniformBuffer.buffer, .offset = 0, .range = sizeof(SceneUniforms),
    };
    const VkDescriptorBufferInfo pointLightsInfo = {
        .buffer = pointLightBuffer.buffer, .offset = 0, .range = VK_WHOLE_SIZE,
    };
    const VkDescriptorBufferInfo spotLightsInfo = {
        .buffer = spotLightBuffer.buffer, .offset = 0, .range = VK_WHOLE_SIZE,
    };
    const VkWriteDescriptorSet sceneWrites[] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = sceneDescriptorSet,
          .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          .pBufferInfo = &sceneInfo },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = sceneDescriptorSet,
          .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &pointLightsInfo },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = sceneDescriptorSet,
          .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &spotLightsInfo },
    };
    vkUpdateDescriptorSets(context.device, sizeof(sceneWrites) / sizeof(sceneWrites[0]),
                           sceneWrites, 0, nullptr);

    // Для каждого материала создаём свой набор дескрипторов с динамическим UBO и 3 текстурами
    const VkDescriptorBufferInfo dynamicUboInfo = {
        .buffer = modelUniformBuffer.buffer,
        .offset = 0,
        .range = sizeof(ModelUniforms),
    };

    for (TextureMaterial& mat : materials) {
        if (!allocateDescriptorSet(modelDescriptorSetLayout, mat.descriptorSet)) {
            return false;
        }

        const VkImageView specView = mat.hasOwnSpecular ? mat.specular.view : blackDummyTexture.view;
        const VkImageView emisView = mat.hasOwnEmissive ? mat.emissive.view : blackDummyTexture.view;

        const VkDescriptorImageInfo imageInfos[3] = {
            {
                .sampler = textureSampler,
                .imageView = mat.albedo.view,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            },
            {
                .sampler = textureSampler,
                .imageView = specView,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            },
            {
                .sampler = textureSampler,
                .imageView = emisView,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            },
        };

        const VkWriteDescriptorSet writes[4] = {
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = mat.descriptorSet,
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                .pBufferInfo = &dynamicUboInfo,
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = mat.descriptorSet,
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &imageInfos[0],
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = mat.descriptorSet,
                .dstBinding = 2,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &imageInfos[1],
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = mat.descriptorSet,
                .dstBinding = 3,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &imageInfos[2],
            },
        };

        vkUpdateDescriptorSets(context.device, 4, writes, 0, nullptr);
    }

    return true;
}

VkPipelineShaderStageCreateInfo makeShaderStage(VkShaderStageFlagBits stage,
                                                VkShaderModule module) {
    return {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = stage, .module = module, .pName = "main",
    };
}

bool createPipeline(VkShaderModule vertexShader, VkShaderModule fragmentShader,
                    VkPrimitiveTopology topology, VkCullModeFlags cullMode,
                    const char* debugName, VkPipeline& pipeline) {
    const VkVertexInputBindingDescription binding = {
        .binding = 0, .stride = sizeof(Vertex), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };
    const VkVertexInputAttributeDescription attributes[] = {
        { .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
          .offset = offsetof(Vertex, position) },
        { .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
          .offset = offsetof(Vertex, normal) },
        { .location = 2, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
          .offset = offsetof(Vertex, smoothNormal) },
        { .location = 3, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT,
          .offset = offsetof(Vertex, uv) },
    };
    const VkPipelineVertexInputStateCreateInfo vertexInput = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
        .pVertexAttributeDescriptions = attributes,
    };
    const VkPipelineInputAssemblyStateCreateInfo inputAssembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = topology,
    };
    const VkPipelineViewportStateCreateInfo viewportState = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1, .scissorCount = 1,
    };
    const VkPipelineRasterizationStateCreateInfo rasterization = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = cullMode,
        .frontFace = VK_FRONT_FACE_CLOCKWISE, .lineWidth = 1.0f,
    };
    const VkPipelineMultisampleStateCreateInfo multisample = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };
    const VkPipelineDepthStencilStateCreateInfo depthStencil = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
    };
    const VkPipelineColorBlendAttachmentState blendAttachment = {
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    const VkPipelineColorBlendStateCreateInfo blend = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1, .pAttachments = &blendAttachment,
    };
    const VkDynamicState dynamicStates[] = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
    };
    const VkPipelineDynamicStateCreateInfo dynamic = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = sizeof(dynamicStates) / sizeof(dynamicStates[0]),
        .pDynamicStates = dynamicStates,
    };
    const VkPipelineShaderStageCreateInfo stages[] = {
        makeShaderStage(VK_SHADER_STAGE_VERTEX_BIT, vertexShader),
        makeShaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, fragmentShader),
    };
    const VkGraphicsPipelineCreateInfo pipelineInfo = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2, .pStages = stages,
        .pVertexInputState = &vertexInput, .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState, .pRasterizationState = &rasterization,
        .pMultisampleState = &multisample, .pDepthStencilState = &depthStencil,
        .pColorBlendState = &blend, .pDynamicState = &dynamic,
        .layout = pipelineLayout, .renderPass = context.render_pass,
    };
    if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
                                  &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
        std::cerr << "Failed to create pipeline: " << debugName << '\n';
        return false;
    }
    return true;
}

bool createPipelines() {
    const VkDescriptorSetLayout setLayouts[2] = {
        sceneDescriptorSetLayout, modelDescriptorSetLayout,
    };
    const VkPushConstantRange pushConstRange = {
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = 0,
        .size = (sizeof(PipelineConsts) + sizeof(uint32_t) - 1) & ~(sizeof(uint32_t) - 1),
    };
    const VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 2,
        .pSetLayouts = setLayouts,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstRange,
    };
    if (vkCreatePipelineLayout(context.device, &pipelineLayoutInfo, nullptr,
                               &pipelineLayout) != VK_SUCCESS) {
        std::cerr << "Failed to create pipeline layout\n";
        return false;
    }
    if (!loadShaderModule("lit.vert.spv", litVertexShader) ||
        !loadShaderModule("lit.frag.spv", litFragmentShader) ||
        !loadShaderModule("unlit.vert.spv", unlitVertexShader) ||
        !loadShaderModule("unlit.frag.spv", unlitFragmentShader) ||
        !loadShaderModule("edge.vert.spv", edgeVertexShader) ||
        !loadShaderModule("edge.frag.spv", edgeFragmentShader)) {
        return false;
    }
    if (!createPipeline(litVertexShader, litFragmentShader,
                        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_CULL_MODE_BACK_BIT,
                        "lit", litPipeline)) {
        return false;
    }
    if (!createPipeline(unlitVertexShader, unlitFragmentShader,
                        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_CULL_MODE_BACK_BIT,
                        "marker", markerPipeline)) {
        return false;
    }
    if (!createPipeline(unlitVertexShader, unlitFragmentShader,
                        VK_PRIMITIVE_TOPOLOGY_LINE_LIST, VK_CULL_MODE_NONE,
                        "line", linePipeline)) {
        return false;
    }
    if (!createPipeline(edgeVertexShader, edgeFragmentShader,
                        VK_PRIMITIVE_TOPOLOGY_LINE_LIST, VK_CULL_MODE_NONE,
                        "edge", edgePipeline)) {
        return false;
    }
    return true;
}

Vec3 safeNormalize(const Vec3& v, const Vec3& fallback) {
    const float length = lab_math::length(v);
    return length > 1e-4f ? lab_math::scale(v, 1.0f / length) : fallback;
}

Mat4 makeModelMatrix(const SceneObject& object) {
    const float phase = static_cast<float>(
        animationTime * object.animationMultiplier + object.phaseOffset);
    Vec3 animatedOffset{};
    Vec3 animatedRotationDegrees{};
    if (object.animatable && object.animate) {
        animatedOffset = {
            trajectoryRadius * std::cos(phase),
            trajectoryVerticalAmplitude *
                std::sin(trajectoryVerticalFrequency * phase + object.phaseOffset * 0.5f),
            trajectoryDepthAmplitude *
                std::sin(trajectoryDepthFrequency * phase + object.phaseOffset),
        };
        const float t = static_cast<float>(animationTime * object.animationMultiplier);
        animatedRotationDegrees = {
            animatedRotationSpeedDegrees.x * t,
            animatedRotationSpeedDegrees.y * t,
            animatedRotationSpeedDegrees.z * t,
        };
    }
    const Vec3 finalPosition = lab_math::add(object.position, animatedOffset);
    const Vec3 finalRotationDegrees =
        lab_math::add(object.rotationDegrees, animatedRotationDegrees);
    const Mat4 S = lab_math::scaleMatrix(object.scale);
    const Mat4 Rx = lab_math::rotateX(lab_math::radians(finalRotationDegrees.x));
    const Mat4 Ry = lab_math::rotateY(lab_math::radians(finalRotationDegrees.y));
    const Mat4 Rz = lab_math::rotateZ(lab_math::radians(finalRotationDegrees.z));
    const Mat4 T = lab_math::translate(finalPosition);
    const Mat4 R = lab_math::multiply(Rz, lab_math::multiply(Ry, Rx));
    return lab_math::multiply(T, lab_math::multiply(R, S));
}

Mat4 makeViewMatrix() {
    return lab_math::viewMatrix(cameraPosition,
                                lab_math::radians(cameraPitchDegrees),
                                lab_math::radians(cameraYawDegrees),
                                lab_math::radians(cameraRollDegrees));
}

Mat4 makeProjectionMatrix() {
    const float width = static_cast<float>(context.swapchain_extent.width);
    const float height = static_cast<float>(context.swapchain_extent.height);
    const float aspect = width / std::max(height, 1.0f);
    if (projectionType == ProjectionType::Orthographic) {
        const float halfHeight = orthographicHeight * 0.5f;
        const float halfWidth = halfHeight * aspect;
        return lab_math::orthographic(-halfWidth, halfWidth,
                                      -halfHeight, halfHeight,
                                      nearPlane, farPlane);
    }
    return lab_math::perspective(lab_math::radians(perspectiveFovDegrees), aspect,
                                 nearPlane, farPlane);
}

Vec3 effectivePointLightPosition(const PointLightData& light) {
    if (!light.orbit) return light.position;
    const float angle = lab_math::radians(orbitSpeedDegrees) * static_cast<float>(animationTime);
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    const Vec3 relative = lab_math::sub(light.position, orbitCenter);
    return lab_math::add(orbitCenter, {
        relative.x * c + relative.z * s,
        relative.y,
        -relative.x * s + relative.z * c,
    });
}

void updateCameraInput(double delta) {
    const ImGuiIO& io = ImGui::GetIO();
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) && !io.WantCaptureMouse) {
        cameraLookActive = true;
    }
    if (!ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
        cameraLookActive = false;
    }
    if (cameraLookActive) {
        cameraYawDegrees += io.MouseDelta.x * cameraMouseSensitivity;
        cameraPitchDegrees -= io.MouseDelta.y * cameraMouseSensitivity;
        cameraPitchDegrees = std::clamp(cameraPitchDegrees, -89.0f, 89.0f);
    }
    if (io.WantTextInput) return;
    const Mat4 rotation = lab_math::cameraRotation(
        lab_math::radians(cameraPitchDegrees),
        lab_math::radians(cameraYawDegrees),
        lab_math::radians(cameraRollDegrees));
    const Vec3 right = {rotation.m[0][0], rotation.m[1][0], rotation.m[2][0]};
    const Vec3 forward = {rotation.m[0][2], rotation.m[1][2], rotation.m[2][2]};
    const Vec3 worldUp = {0.0f, -1.0f, 0.0f};
    Vec3 direction{};
    if (ImGui::IsKeyDown(ImGuiKey_W)) direction = lab_math::add(direction, forward);
    if (ImGui::IsKeyDown(ImGuiKey_S)) direction = lab_math::sub(direction, forward);
    if (ImGui::IsKeyDown(ImGuiKey_D)) direction = lab_math::add(direction, right);
    if (ImGui::IsKeyDown(ImGuiKey_A)) direction = lab_math::sub(direction, right);
    if (ImGui::IsKeyDown(ImGuiKey_E)) direction = lab_math::add(direction, worldUp);
    if (ImGui::IsKeyDown(ImGuiKey_Q)) direction = lab_math::sub(direction, worldUp);
    if (lab_math::length(direction) > 1e-4f) {
        const float boost = ImGui::IsKeyDown(ImGuiKey_LeftShift) ? 3.0f : 1.0f;
        const float distance = cameraMoveSpeed * boost * static_cast<float>(delta);
        cameraPosition = lab_math::add(
            cameraPosition, lab_math::scale(lab_math::normalize(direction), distance));
    }
}

void drawCameraUI() {
    if (!ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) return;
    ImGui::TextWrapped("WASD - move, Q/E - down/up, Shift - faster, hold right mouse button - look around");
    ImGui::DragFloat3("Position", &cameraPosition.x, 0.05f, -50.0f, 50.0f, "%.2f");
    ImGui::DragFloat("Pitch (+ up)", &cameraPitchDegrees, 0.5f, -89.0f, 89.0f, "%.1f deg");
    ImGui::DragFloat("Yaw (+ right)", &cameraYawDegrees, 0.5f, -360.0f, 360.0f, "%.1f deg");
    ImGui::DragFloat("Roll", &cameraRollDegrees, 0.5f, -180.0f, 180.0f, "%.1f deg");
    int projection = static_cast<int>(projectionType);
    const char* projectionItems[] = {"Perspective", "Orthographic"};
    if (ImGui::Combo("Projection", &projection, projectionItems, 2)) {
        projectionType = static_cast<ProjectionType>(projection);
    }
    if (projectionType == ProjectionType::Perspective) {
        ImGui::SliderFloat("FOV Y", &perspectiveFovDegrees, 20.0f, 120.0f, "%.1f deg");
    } else {
        ImGui::SliderFloat("Ortho area height", &orthographicHeight, 2.0f, 30.0f, "%.2f");
    }
    if (ImGui::SliderFloat("Near", &nearPlane, 0.01f, 10.0f, "%.2f")) {
        nearPlane = std::min(nearPlane, farPlane - 0.01f);
    }
    if (ImGui::SliderFloat("Far", &farPlane, 1.0f, 200.0f, "%.1f")) {
        farPlane = std::max(farPlane, nearPlane + 0.01f);
    }
    ImGui::SliderFloat("Move speed", &cameraMoveSpeed, 0.5f, 20.0f, "%.1f");
    ImGui::SliderFloat("Mouse sensitivity", &cameraMouseSensitivity, 0.02f, 0.5f, "%.2f");
    if (ImGui::Button("Reset camera")) {
        cameraPosition = default_camera_position;
        cameraPitchDegrees = -10.0f;
        cameraYawDegrees = 0.0f;
        cameraRollDegrees = 0.0f;
    }
}

void drawGlobalLightUI() {
    if (!ImGui::CollapsingHeader("Ambient & directional light", ImGuiTreeNodeFlags_DefaultOpen)) return;
    ImGui::ColorEdit3("Ambient color", ambientLightColor);
    ImGui::SliderFloat("Ambient intensity", &ambientLightIntensity, 0.0f, 1.0f, "%.2f");
    ImGui::Separator();
    ImGui::DragFloat3("Direction (light travels)", &directionalLightDirection.x,
                      0.01f, -1.0f, 1.0f, "%.2f");
    ImGui::ColorEdit3("Directional color", directionalLightColor);
    ImGui::SliderFloat("Directional intensity", &directionalLightIntensity, 0.0f, 3.0f, "%.2f");
}

void drawPointLightsUI() {
    if (!ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) return;
    ImGui::Text("%zu / %u", pointLights.size(), max_point_lights);
    ImGui::SameLine();
    ImGui::BeginDisabled(pointLights.size() >= max_point_lights);
    if (ImGui::Button("Add point light")) {
        pointLights.push_back(PointLightData{});
    }
    ImGui::EndDisabled();
    int removeIndex = -1;
    for (size_t i = 0; i < pointLights.size(); ++i) {
        PointLightData& light = pointLights[i];
        ImGui::PushID(static_cast<int>(i));
        const std::string label = "Point light " + std::to_string(i);
        if (ImGui::TreeNode(label.c_str())) {
            ImGui::DragFloat3("Position", &light.position.x, 0.05f, -30.0f, 30.0f, "%.2f");
            ImGui::ColorEdit3("Color", light.color);
            ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 100.0f, "%.2f",
                               ImGuiSliderFlags_Logarithmic);
            ImGui::Checkbox("Orbit", &light.orbit);
            if (ImGui::Button("Remove")) removeIndex = static_cast<int>(i);
            ImGui::TreePop();
        }
        ImGui::PopID();
    }
    if (removeIndex >= 0) {
        pointLights.erase(pointLights.begin() + removeIndex);
    }
}

void drawSpotLightsUI() {
    if (!ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) return;
    ImGui::Text("%zu / %u", spotLights.size(), max_spot_lights);
    ImGui::SameLine();
    ImGui::BeginDisabled(spotLights.size() >= max_spot_lights);
    if (ImGui::Button("Add spot light")) {
        spotLights.push_back(SpotLightData{});
    }
    ImGui::EndDisabled();
    int removeIndex = -1;
    for (size_t i = 0; i < spotLights.size(); ++i) {
        SpotLightData& light = spotLights[i];
        ImGui::PushID(static_cast<int>(i));
        const std::string label = "Spot light " + std::to_string(i);
        if (ImGui::TreeNode(label.c_str())) {
            ImGui::DragFloat3("Position", &light.position.x, 0.05f, -30.0f, 30.0f, "%.2f");
            ImGui::DragFloat3("Direction", &light.direction.x, 0.01f, -1.0f, 1.0f, "%.2f");
            ImGui::ColorEdit3("Color", light.color);
            ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 500.0f, "%.2f",
                               ImGuiSliderFlags_Logarithmic);
            if (ImGui::SliderFloat("Inner angle", &light.innerAngleDegrees, 0.0f, 85.0f, "%.1f deg")) {
                light.outerAngleDegrees = std::max(light.outerAngleDegrees, light.innerAngleDegrees);
            }
            if (ImGui::SliderFloat("Outer angle", &light.outerAngleDegrees, 1.0f, 85.0f, "%.1f deg")) {
                light.innerAngleDegrees = std::min(light.innerAngleDegrees, light.outerAngleDegrees);
            }
            if (ImGui::Button("Remove")) removeIndex = static_cast<int>(i);
            ImGui::TreePop();
        }
        ImGui::PopID();
    }
    if (removeIndex >= 0) {
        spotLights.erase(spotLights.begin() + removeIndex);
    }
}

void drawVisualizationUI() {
    if (!ImGui::CollapsingHeader("Light visualization")) return;
    ImGui::Checkbox("Show light markers", &showLightMarkers);
    ImGui::Checkbox("Show spot light cones", &showSpotCones);
    ImGui::SliderFloat("Marker size", &markerSize, 0.05f, 0.6f, "%.2f");
    ImGui::SliderFloat("Cone length", &coneLength, 0.5f, 10.0f, "%.2f");
}

void drawAnimationUI() {
    if (!ImGui::CollapsingHeader("Animation")) return;
    if (ImGui::Button(animationPlaying ? "Pause" : "Play")) {
        animationPlaying = !animationPlaying;
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset time")) {
        resetAnimationRequested = true;
    }
    ImGui::SliderFloat("Speed", &animationSpeed, 0.0f, 3.0f, "%.2fx");
    ImGui::Separator();
    ImGui::Text("Object trajectory");
    ImGui::SliderFloat("Trajectory radius", &trajectoryRadius, 0.0f, 2.5f, "%.2f");
    ImGui::SliderFloat("Y amplitude", &trajectoryVerticalAmplitude, 0.0f, 2.5f, "%.2f");
    ImGui::SliderFloat("Z amplitude", &trajectoryDepthAmplitude, 0.0f, 2.0f, "%.2f");
    ImGui::SliderFloat("Y frequency", &trajectoryVerticalFrequency, 0.1f, 5.0f, "%.2f");
    ImGui::SliderFloat("Z frequency", &trajectoryDepthFrequency, 0.1f, 3.0f, "%.2f");
    ImGui::DragFloat3("Rotation speed", &animatedRotationSpeedDegrees.x,
                      1.0f, -180.0f, 180.0f, "%.1f deg/s");
    ImGui::Separator();
    ImGui::Text("Point light orbit");
    ImGui::DragFloat3("Orbit center", &orbitCenter.x, 0.05f, -30.0f, 30.0f, "%.2f");
    ImGui::SliderFloat("Orbit speed", &orbitSpeedDegrees, -180.0f, 180.0f, "%.1f deg/s");
}

void drawObjectUI(SceneObject& object) {
    if (!ImGui::TreeNode(object.name)) return;

    // Выбор текстуры (материала) из выпадающего списка
    std::vector<const char*> matNames;
    matNames.reserve(materials.size());
    for (const TextureMaterial& m : materials) {
        matNames.push_back(m.name.c_str());
    }
    ImGui::Combo("Texture Material", &object.material.textureIndex,
                 matNames.data(), static_cast<int>(matNames.size()));

    ImGui::Checkbox("Non-trivial sampling (Push Const)", &object.nontrivialSampling);
    ImGui::SliderFloat("UV Tiling Scale", &object.material.uvScale, 0.25f, 10.0f, "%.2f");
    ImGui::SliderFloat("Emissive Strength", &object.material.emissiveStrength, 0.0f, 4.0f, "%.2f");

    ImGui::Separator();
    if (object.animatable) {
        ImGui::Checkbox("Animate", &object.animate);
        ImGui::DragFloat("Speed multiplier", &object.animationMultiplier,
                         0.01f, 0.0f, 3.0f, "%.2f");
        ImGui::DragFloat("Phase offset", &object.phaseOffset,
                         0.01f, -lab_math::PI, lab_math::PI, "%.2f rad");
        ImGui::Separator();
    }
    ImGui::DragFloat3("Position", &object.position.x, 0.01f, -20.0f, 20.0f, "%.2f");
    ImGui::DragFloat3("Rotation", &object.rotationDegrees.x, 1.0f, -360.0f, 360.0f, "%.1f deg");
    ImGui::DragFloat3("Scale", &object.scale.x, 0.01f, 0.1f, 20.0f, "%.2f");
    ImGui::Separator();
    ImGui::Text("Material multipliers (Blinn-Phong)");
    ImGui::ColorEdit3("Albedo tint", object.material.albedo);
    ImGui::ColorEdit3("Specular color", object.material.specular);
    ImGui::SliderFloat("Shininess", &object.material.shininess, 1.0f, 512.0f, "%.1f",
                       ImGuiSliderFlags_Logarithmic);
    ImGui::Checkbox("Smooth normals", &object.smoothNormals);
    ImGui::TreePop();
}

void drawObjectsUI() {
    if (!ImGui::CollapsingHeader("Objects & Textures", ImGuiTreeNodeFlags_DefaultOpen)) return;
    for (SceneObject& object : objects) {
        drawObjectUI(object);
    }
}

void drawUI() {
    ImGui::Begin("Vulkan Lab 3 - Texture Mapping");
    drawObjectsUI();
    drawCameraUI();
    drawGlobalLightUI();
    drawPointLightsUI();
    drawSpotLightsUI();
    drawVisualizationUI();
    drawAnimationUI();
    if (ImGui::CollapsingHeader("Rendering")) {
        ImGui::Checkbox("Show icosahedron edges", &showEdges);
        ImGui::Text("Dynamic UBO alignment: %llu bytes",
                    static_cast<unsigned long long>(modelUniformAlignment));
    }
    ImGui::End();
}

// Запись в динамический uniform-буфер по индексу слота (лекция #6, слайд 10)
void writeDynamicModelUniforms(uint32_t slotIndex, const Mat4& model,
                               const float albedo[3], const float specular[3],
                               float shininess, bool smoothNormals,
                               float uvScale, float emissiveStrength) {
    if (slotIndex >= max_models) return;
    uint8_t* basePtr = static_cast<uint8_t*>(modelUniformBuffer.mapped);
    ModelUniforms* uniforms = reinterpret_cast<ModelUniforms*>(
        basePtr + slotIndex * modelUniformAlignment);

    lab_math::toColumnMajor(model, uniforms->model);
    std::memcpy(uniforms->albedo, albedo, 3 * sizeof(float));
    uniforms->albedo[3] = 1.0f;
    std::memcpy(uniforms->specular, specular, 3 * sizeof(float));
    uniforms->specular[3] = shininess;
    uniforms->params[0] = smoothNormals ? 1.0f : 0.0f;
    uniforms->params[1] = uvScale;
    uniforms->params[2] = emissiveStrength;
    uniforms->params[3] = 0.0f;
}

void updateSceneUniform() {
    SceneUniforms scene{};
    const Mat4 viewProjection = lab_math::multiply(makeProjectionMatrix(), makeViewMatrix());
    lab_math::toColumnMajor(viewProjection, scene.viewProjection);
    scene.cameraPosition[0] = cameraPosition.x;
    scene.cameraPosition[1] = cameraPosition.y;
    scene.cameraPosition[2] = cameraPosition.z;
    const Vec3 direction = safeNormalize(directionalLightDirection, {0.0f, 1.0f, 0.0f});
    scene.directionalLightDirection[0] = direction.x;
    scene.directionalLightDirection[1] = direction.y;
    scene.directionalLightDirection[2] = direction.z;
    for (int i = 0; i < 3; ++i) {
        scene.directionalLightColor[i] = directionalLightColor[i] * directionalLightIntensity;
        scene.ambientLight[i] = ambientLightColor[i] * ambientLightIntensity;
    }
    scene.lightCounts[0] = static_cast<uint32_t>(pointLights.size());
    scene.lightCounts[1] = static_cast<uint32_t>(spotLights.size());
    writeToBuffer(sceneUniformBuffer, &scene, sizeof(scene));
}

void updateLightBuffers() {
    std::vector<GpuPointLight> gpuPointLights;
    gpuPointLights.reserve(pointLights.size());
    for (const PointLightData& light : pointLights) {
        const Vec3 position = effectivePointLightPosition(light);
        GpuPointLight gpu{};
        gpu.position[0] = position.x;
        gpu.position[1] = position.y;
        gpu.position[2] = position.z;
        gpu.intensity = light.intensity;
        std::memcpy(gpu.color, light.color, sizeof(gpu.color));
        gpuPointLights.push_back(gpu);
    }
    std::vector<GpuSpotLight> gpuSpotLights;
    gpuSpotLights.reserve(spotLights.size());
    for (const SpotLightData& light : spotLights) {
        const Vec3 direction = safeNormalize(light.direction, {0.0f, 1.0f, 0.0f});
        GpuSpotLight gpu{};
        gpu.position[0] = light.position.x;
        gpu.position[1] = light.position.y;
        gpu.position[2] = light.position.z;
        gpu.intensity = light.intensity;
        gpu.direction[0] = direction.x;
        gpu.direction[1] = direction.y;
        gpu.direction[2] = direction.z;
        gpu.cosInner = std::cos(lab_math::radians(light.innerAngleDegrees));
        gpu.cosOuter = std::cos(lab_math::radians(light.outerAngleDegrees));
        std::memcpy(gpu.color, light.color, sizeof(gpu.color));
        gpuSpotLights.push_back(gpu);
    }
    writeToBuffer(pointLightBuffer, gpuPointLights.data(),
                  sizeof(GpuPointLight) * gpuPointLights.size());
    writeToBuffer(spotLightBuffer, gpuSpotLights.data(),
                  sizeof(GpuSpotLight) * gpuSpotLights.size());
}

void updateAllModelUniforms() {
    for (SceneObject& object : objects) {
        writeDynamicModelUniforms(object.dynamicSlotIndex, makeModelMatrix(object),
                                  object.material.albedo, object.material.specular,
                                  object.material.shininess, object.smoothNormals,
                                  object.material.uvScale, object.material.emissiveStrength);
    }

    static const float noSpecular[3] = {0.0f, 0.0f, 0.0f};
    static const float white[3] = {1.0f, 1.0f, 1.0f};
    for (size_t i = 0; i < pointLights.size(); ++i) {
        const PointLightData& light = pointLights[i];
        const Mat4 model = lab_math::multiply(
            lab_math::translate(effectivePointLightPosition(light)),
            lab_math::scaleMatrix({markerSize, markerSize, markerSize}));
        writeDynamicModelUniforms(point_marker_base_slot + static_cast<uint32_t>(i),
                                  model, light.color, noSpecular, 1.0f, false, 1.0f, 0.0f);
    }
    for (size_t i = 0; i < spotLights.size(); ++i) {
        const SpotLightData& light = spotLights[i];
        const Mat4 markerModel = lab_math::multiply(
            lab_math::translate(light.position),
            lab_math::scaleMatrix({markerSize, markerSize, markerSize}));
        writeDynamicModelUniforms(spot_marker_base_slot + static_cast<uint32_t>(i),
                                  markerModel, light.color, noSpecular, 1.0f, false, 1.0f, 0.0f);

        const float radius = coneLength * std::tan(lab_math::radians(light.outerAngleDegrees));
        const Mat4 coneModel = lab_math::multiply(
            lab_math::translate(light.position),
            lab_math::multiply(
                lab_math::directionBasis(safeNormalize(light.direction, {0.0f, 1.0f, 0.0f})),
                lab_math::scaleMatrix({radius, radius, coneLength})));
        writeDynamicModelUniforms(spot_cone_base_slot + static_cast<uint32_t>(i),
                                  coneModel, white, noSpecular, 1.0f, false, 1.0f, 0.0f);
    }

    vmaFlushAllocation(context.allocator, modelUniformBuffer.allocation, 0, VK_WHOLE_SIZE);
}

// Отрисовка модели с передачей динамического сдвига в vkCmdBindDescriptorSets и push-констант
void drawModelDynamic(const graphics::internal::FrameData& fd, const Mesh& mesh,
                      VkDescriptorSet materialDescriptorSet, uint32_t slotIndex,
                      bool nontrivialSampling) {
    const VkDeviceSize vertexBufferOffset = 0;
    vkCmdBindVertexBuffers(fd.command_buffer, 0, 1,
                           &mesh.vertexBuffer.buffer, &vertexBufferOffset);
    vkCmdBindIndexBuffer(fd.command_buffer, mesh.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

    const uint32_t dynamicOffset = static_cast<uint32_t>(slotIndex * modelUniformAlignment);
    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout, 1, 1, &materialDescriptorSet,
                            1, &dynamicOffset);

    const PipelineConsts consts = {
        .nontrivialSampling = nontrivialSampling ? 1u : 0u,
        .time = static_cast<float>(animationTime),
    };
    vkCmdPushConstants(fd.command_buffer, pipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(PipelineConsts), &consts);

    vkCmdDrawIndexed(fd.command_buffer, mesh.triangleIndexCount, 1, 0, 0, 0);
}

void setupScene() {
    const auto setup = [](uint32_t idx, const char* name, const Mesh* mesh,
                          Vec3 position, Vec3 scale, int texIdx, float uvScale,
                          float shininess, bool nontrivial) {
        SceneObject& object = objects[idx];
        object.name = name;
        object.mesh = mesh;
        object.position = position;
        object.scale = scale;
        object.material.albedo[0] = 1.0f;
        object.material.albedo[1] = 1.0f;
        object.material.albedo[2] = 1.0f;
        object.material.specular[0] = 1.0f;
        object.material.specular[1] = 1.0f;
        object.material.specular[2] = 1.0f;
        object.material.shininess = shininess;
        object.material.textureIndex = texIdx;
        object.material.uvScale = uvScale;
        object.material.emissiveStrength = 1.3f;
        object.nontrivialSampling = nontrivial;
        object.dynamicSlotIndex = idx;
    };

    // Икосаэдры с разными текстурами (Lava Rock, Sci-Fi с нетривиальным сэмплированием, Marble)
    setup(0, "Icosahedron A (Lava Perlin Emissive)", &icosahedronMesh,
          {-2.6f, 0.0f, 6.0f}, {1.0f, 1.0f, 1.0f}, 0, 1.0f, 64.0f, false);
    setup(1, "Icosahedron B (Non-trivial Sampling)", &icosahedronMesh,
          {0.0f, 0.0f, 6.5f}, {1.2f, 1.2f, 1.2f}, 1, 1.0f, 32.0f, true);
    setup(2, "Icosahedron C (Marble Specular)", &icosahedronMesh,
          {2.6f, 0.0f, 7.0f}, {1.0f, 1.0f, 1.0f}, 3, 1.0f, 128.0f, false);

    for (uint32_t i = 0; i < 3; ++i) {
        objects[i].animatable = true;
        objects[i].animate = true;
        objects[i].phaseOffset = 2.0f * lab_math::PI * static_cast<float>(i) / 3.0f;
    }

    // Окружение: пол и стены с тайлингом текстур
    setup(3, "Floor", &cubeMesh,
          {0.0f, 3.1f, 4.0f}, {16.0f, 0.2f, 14.0f}, 3, 4.0f, 32.0f, false);
    setup(4, "Back wall", &cubeMesh,
          {0.0f, -1.0f, 10.1f}, {16.0f, 8.2f, 0.2f}, 2, 3.0f, 16.0f, false);
    setup(5, "Left wall", &cubeMesh,
          {-8.1f, -1.0f, 4.0f}, {0.2f, 8.2f, 14.0f}, 4, 3.0f, 8.0f, false);
    setup(6, "Right wall", &cubeMesh,
          {8.1f, -1.0f, 4.0f}, {0.2f, 8.2f, 14.0f}, 1, 3.0f, 32.0f, false);

    PointLightData warm;
    warm.position = {0.0f, -3.0f, 4.0f};
    warm.color[0] = 1.0f; warm.color[1] = 0.75f; warm.color[2] = 0.5f;
    warm.intensity = 12.0f; warm.orbit = true;
    pointLights.push_back(warm);

    PointLightData cool;
    cool.position = {4.0f, -2.0f, 2.0f};
    cool.color[0] = 0.4f; cool.color[1] = 0.6f; cool.color[2] = 1.0f;
    cool.intensity = 10.0f;
    pointLights.push_back(cool);

    SpotLightData spot;
    spot.position = {-5.0f, -4.0f, 0.0f};
    spot.direction = {5.0f, 4.0f, 6.0f};
    spotLights.push_back(spot);
}

} // namespace

bool initialize() {
    setupScene();
    if (!createMeshesAndTextures()) {
        std::cerr << "Failed to create meshes and textures\n";
        return false;
    }
    if (!createDescriptorLayouts()) return false;
    if (!createDescriptorPoolAndSets()) return false;
    if (!createPipelines()) return false;
    timeInitialized = false;
    animationTime = 0.0;
    animationPlaying = true;
    return true;
}

void shutdown() {
    vkQueueWaitIdle(context.graphics_queue);

    for (TextureMaterial& mat : materials) {
        destroyTexture(mat.albedo);
        if (mat.hasOwnSpecular) destroyTexture(mat.specular);
        if (mat.hasOwnEmissive) destroyTexture(mat.emissive);
    }
    materials.clear();
    destroyTexture(blackDummyTexture);

    if (textureSampler != VK_NULL_HANDLE) {
        vkDestroySampler(context.device, textureSampler, nullptr);
    }

    destroyBuffer(stagingBuffer);
    if (uploadCommandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(context.device, uploadCommandPool, nullptr);
    }

    destroyBuffer(modelUniformBuffer);
    destroyBuffer(sceneUniformBuffer);
    destroyBuffer(pointLightBuffer);
    destroyBuffer(spotLightBuffer);

    for (Mesh* m : {&icosahedronMesh, &cubeMesh, &coneMesh}) {
        destroyBuffer(m->vertexBuffer);
        destroyBuffer(m->indexBuffer);
    }

    vkDestroyPipeline(context.device, litPipeline, nullptr);
    vkDestroyPipeline(context.device, markerPipeline, nullptr);
    vkDestroyPipeline(context.device, linePipeline, nullptr);
    vkDestroyPipeline(context.device, edgePipeline, nullptr);
    vkDestroyShaderModule(context.device, litVertexShader, nullptr);
    vkDestroyShaderModule(context.device, litFragmentShader, nullptr);
    vkDestroyShaderModule(context.device, unlitVertexShader, nullptr);
    vkDestroyShaderModule(context.device, unlitFragmentShader, nullptr);
    vkDestroyShaderModule(context.device, edgeVertexShader, nullptr);
    vkDestroyShaderModule(context.device, edgeFragmentShader, nullptr);
    vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
    vkDestroyDescriptorPool(context.device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(context.device, sceneDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(context.device, modelDescriptorSetLayout, nullptr);
}

void update(double time) {
    if (!timeInitialized) {
        previousWallTime = time;
        timeInitialized = true;
    }
    const double delta = std::min(time - previousWallTime, 0.1);
    previousWallTime = time;
    updateCameraInput(delta);
    drawUI();
    if (resetAnimationRequested) {
        animationTime = 0.0;
        resetAnimationRequested = false;
    } else if (animationPlaying) {
        animationTime += delta * static_cast<double>(animationSpeed);
    }
}

void render(const graphics::internal::FrameData& fd) {
    updateSceneUniform();
    updateLightBuffers();
    updateAllModelUniforms();

    vkResetCommandBuffer(fd.command_buffer, 0);
    const VkCommandBufferBeginInfo commandBufferBegin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (vkBeginCommandBuffer(fd.command_buffer, &commandBufferBegin) != VK_SUCCESS) {
        std::cerr << "Failed to begin application command buffer\n";
        return;
    }
    const VkClearValue clearValues[] = {
        { .color = { .float32 = {0.02f, 0.02f, 0.03f, 1.0f} } },
        { .depthStencil = {1.0f, 0} },
    };
    const VkRenderPassBeginInfo renderPassBegin = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = context.render_pass,
        .framebuffer = fd.framebuffer,
        .renderArea = { .extent = context.swapchain_extent },
        .clearValueCount = sizeof(clearValues) / sizeof(clearValues[0]),
        .pClearValues = clearValues,
    };
    vkCmdBeginRenderPass(fd.command_buffer, &renderPassBegin, VK_SUBPASS_CONTENTS_INLINE);
    const VkViewport viewport = {
        .x = 0.0f, .y = 0.0f,
        .width = static_cast<float>(context.swapchain_extent.width),
        .height = static_cast<float>(context.swapchain_extent.height),
        .minDepth = 0.0f, .maxDepth = 1.0f,
    };
    const VkRect2D scissor = { .extent = context.swapchain_extent };
    vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
    vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);
    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout, 0, 1, &sceneDescriptorSet, 0, nullptr);

    // Освещаемые текстурированные модели
    vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, litPipeline);
    for (const SceneObject& object : objects) {
        const int texIdx = std::clamp(object.material.textureIndex, 0,
                                      static_cast<int>(materials.size()) - 1);
        drawModelDynamic(fd, *object.mesh, materials[texIdx].descriptorSet,
                         object.dynamicSlotIndex, object.nontrivialSampling);
    }

    // Рёбра икосаэдров
    if (showEdges) {
        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, edgePipeline);
        for (uint32_t i = 0; i < 3; ++i) {
            const SceneObject& object = objects[i];
            if (object.mesh == &icosahedronMesh && icosahedronMesh.edgeIndexCount > 0) {
                const VkDeviceSize vertexBufferOffset = 0;
                vkCmdBindVertexBuffers(fd.command_buffer, 0, 1,
                                       &icosahedronMesh.vertexBuffer.buffer, &vertexBufferOffset);
                vkCmdBindIndexBuffer(fd.command_buffer, icosahedronMesh.indexBuffer.buffer,
                                     0, VK_INDEX_TYPE_UINT32);
                const uint32_t dynamicOffset =
                    static_cast<uint32_t>(object.dynamicSlotIndex * modelUniformAlignment);
                vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        pipelineLayout, 1, 1, &materials[0].descriptorSet,
                                        1, &dynamicOffset);
                vkCmdDrawIndexed(fd.command_buffer, icosahedronMesh.edgeIndexCount, 1,
                                 icosahedronMesh.edgeFirstIndex, 0, 0);
            }
        }
    }

    // Маркеры источников света (используют тот же динамический UBO с разными сдвигами)
    if (showLightMarkers) {
        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, markerPipeline);
        for (size_t i = 0; i < pointLights.size(); ++i) {
            drawModelDynamic(fd, cubeMesh, materials[0].descriptorSet,
                             point_marker_base_slot + static_cast<uint32_t>(i), false);
        }
        for (size_t i = 0; i < spotLights.size(); ++i) {
            drawModelDynamic(fd, cubeMesh, materials[0].descriptorSet,
                             spot_marker_base_slot + static_cast<uint32_t>(i), false);
        }
    }

    // Конусы прожекторов
    if (showSpotCones) {
        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, linePipeline);
        for (size_t i = 0; i < spotLights.size(); ++i) {
            drawModelDynamic(fd, coneMesh, materials[0].descriptorSet,
                             spot_cone_base_slot + static_cast<uint32_t>(i), false);
        }
    }

    vkCmdEndRenderPass(fd.command_buffer);
    if (vkEndCommandBuffer(fd.command_buffer) != VK_SUCCESS) {
        std::cerr << "Failed to end application command buffer\n";
    }
}

} // namespace application