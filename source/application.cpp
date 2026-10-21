#include "application.hpp"
#include "geometry.hpp"
#include "math.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <imgui.h>

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
    float params[4];
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

struct Mesh {
    GpuBuffer vertexBuffer;
    GpuBuffer indexBuffer;
    uint32_t triangleIndexCount = 0;
    uint32_t edgeFirstIndex = 0;
    uint32_t edgeIndexCount = 0;
};

struct ModelSlot {
    GpuBuffer uniformBuffer;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
};

struct Material {
    float albedo[3] = {0.8f, 0.8f, 0.8f};
    float specular[3] = {1.0f, 1.0f, 1.0f};
    float shininess = 32.0f;
};

struct SceneObject {
    const char* name = "";
    const Mesh* mesh = nullptr;
    Vec3 position{};
    Vec3 rotationDegrees{};
    Vec3 scale{1.0f, 1.0f, 1.0f};
    Material material;
    bool smoothNormals = false;
    bool animatable = false;
    bool animate = false;
    float phaseOffset = 0.0f;
    float animationMultiplier = 1.0f;
    ModelSlot slot;
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
constexpr uint32_t model_slot_count = object_count + max_point_lights + 2 * max_spot_lights;

Mesh icosahedronMesh;
Mesh cubeMesh;
Mesh coneMesh;
SceneObject objects[object_count];

ModelSlot pointMarkerSlots[max_point_lights];
ModelSlot spotMarkerSlots[max_spot_lights];
ModelSlot spotConeSlots[max_spot_lights];

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
bool showEdges = true;  // <-- ДОБАВЛЕНО
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
VkPipeline edgePipeline = VK_NULL_HANDLE;  // <-- ДОБАВЛЕНО
VkShaderModule litVertexShader = VK_NULL_HANDLE;
VkShaderModule litFragmentShader = VK_NULL_HANDLE;
VkShaderModule unlitVertexShader = VK_NULL_HANDLE;
VkShaderModule unlitFragmentShader = VK_NULL_HANDLE;
VkShaderModule edgeVertexShader = VK_NULL_HANDLE;   // <-- ДОБАВЛЕНО
VkShaderModule edgeFragmentShader = VK_NULL_HANDLE; // <-- ДОБАВЛЕНО

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
        std::cerr << "Failed to create and allocate Vulkan buffer\n";
        return false;
    }
    result.mapped = mappedInfo.pMappedData;
    result.size = alignedSize;
    if (result.mapped == nullptr) {
        std::cerr << "Vulkan buffer memory is not host-mapped\n";
        return false;
    }
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

bool createMesh(const MeshData& data, Mesh& mesh) {
    const size_t vertexBytes = sizeof(Vertex) * data.vertices.size();
    const size_t indexBytes = sizeof(uint32_t) * data.indices.size();
    mesh.triangleIndexCount = data.triangleIndexCount;
    mesh.edgeFirstIndex = data.edgeFirstIndex;
    mesh.edgeIndexCount = data.edgeIndexCount;
    if (!createHostBuffer(vertexBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, mesh.vertexBuffer)) {
        return false;
    }
    writeToBuffer(mesh.vertexBuffer, data.vertices.data(), vertexBytes);
    if (!createHostBuffer(indexBytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, mesh.indexBuffer)) {
        return false;
    }
    writeToBuffer(mesh.indexBuffer, data.indices.data(), indexBytes);
    return true;
}

bool createMeshes() {
    return createMesh(lab_geometry::makeIcosahedron(icosahedron_radius), icosahedronMesh) &&
           createMesh(lab_geometry::makeCube(), cubeMesh) &&
           createMesh(lab_geometry::makeCone(cone_segments), coneMesh);
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
    const VkDescriptorSetLayoutBinding modelBinding = {
        .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
    };
    const VkDescriptorSetLayoutCreateInfo modelLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1, .pBindings = &modelBinding,
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

bool createModelSlot(ModelSlot& slot) {
    if (!createHostBuffer(sizeof(ModelUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                          slot.uniformBuffer)) {
        return false;
    }
    if (!allocateDescriptorSet(modelDescriptorSetLayout, slot.descriptorSet)) {
        return false;
    }
    const VkDescriptorBufferInfo bufferInfo = {
        .buffer = slot.uniformBuffer.buffer, .offset = 0, .range = sizeof(ModelUniforms),
    };
    const VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = slot.descriptorSet, .dstBinding = 0, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &bufferInfo,
    };
    vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
    return true;
}

bool createDescriptorPoolAndSets() {
    constexpr uint32_t setCount = 1 + model_slot_count;
    const VkDescriptorPoolSize poolSizes[] = {
        { .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1 + model_slot_count },
        { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 },
    };
    const VkDescriptorPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = setCount,
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
    for (SceneObject& object : objects) {
        if (!createModelSlot(object.slot)) return false;
    }
    for (ModelSlot& slot : pointMarkerSlots) {
        if (!createModelSlot(slot)) return false;
    }
    for (ModelSlot& slot : spotMarkerSlots) {
        if (!createModelSlot(slot)) return false;
    }
    for (ModelSlot& slot : spotConeSlots) {
        if (!createModelSlot(slot)) return false;
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
    const VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 2, .pSetLayouts = setLayouts,
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
    // <-- ДОБАВЛЕНО: конвейер для рёбер икосаэдров
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
    if (!ImGui::CollapsingHeader("Light visualization", ImGuiTreeNodeFlags_DefaultOpen)) return;
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
    ImGui::Text("Material (Blinn-Phong)");
    ImGui::ColorEdit3("Albedo", object.material.albedo);
    ImGui::ColorEdit3("Specular color", object.material.specular);
    ImGui::SliderFloat("Shininess", &object.material.shininess, 1.0f, 512.0f, "%.1f",
                       ImGuiSliderFlags_Logarithmic);
    ImGui::Checkbox("Smooth normals", &object.smoothNormals);
    ImGui::TreePop();
}

void drawObjectsUI() {
    if (!ImGui::CollapsingHeader("Objects")) return;
    for (SceneObject& object : objects) {
        drawObjectUI(object);
    }
}

void drawUI() {
    ImGui::Begin("Vulkan Lab 2 - Camera & Lighting");
    drawCameraUI();
    drawGlobalLightUI();
    drawPointLightsUI();
    drawSpotLightsUI();
    drawVisualizationUI();
    drawAnimationUI();
    drawObjectsUI();
    // <-- ДОБАВЛЕНО: чекбокс Show edges
    if (ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Show edges", &showEdges);
        ImGui::Text("Icosahedron: vertices: 12, edges: %u, faces: %u",
                    icosahedronMesh.edgeIndexCount / 2,
                    icosahedronMesh.triangleIndexCount / 3);
    }
    ImGui::End();
}

void writeModelUniforms(ModelSlot& slot, const Mat4& model,
                        const float albedo[3], const float specular[3],
                        float shininess, bool smoothNormals) {
    ModelUniforms uniforms{};
    lab_math::toColumnMajor(model, uniforms.model);
    std::memcpy(uniforms.albedo, albedo, 3 * sizeof(float));
    std::memcpy(uniforms.specular, specular, 3 * sizeof(float));
    uniforms.specular[3] = shininess;
    uniforms.params[0] = smoothNormals ? 1.0f : 0.0f;
    writeToBuffer(slot.uniformBuffer, &uniforms, sizeof(uniforms));
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

void updateObjectUniforms() {
    for (SceneObject& object : objects) {
        writeModelUniforms(object.slot, makeModelMatrix(object),
                           object.material.albedo, object.material.specular,
                           object.material.shininess, object.smoothNormals);
    }
}

void updateMarkerUniforms() {
    static const float noSpecular[3] = {0.0f, 0.0f, 0.0f};
    static const float white[3] = {1.0f, 1.0f, 1.0f};
    for (size_t i = 0; i < pointLights.size(); ++i) {
        const PointLightData& light = pointLights[i];
        const Mat4 model = lab_math::multiply(
            lab_math::translate(effectivePointLightPosition(light)),
            lab_math::scaleMatrix({markerSize, markerSize, markerSize}));
        writeModelUniforms(pointMarkerSlots[i], model, light.color, noSpecular, 1.0f, false);
    }
    for (size_t i = 0; i < spotLights.size(); ++i) {
        const SpotLightData& light = spotLights[i];
        const Mat4 markerModel = lab_math::multiply(
            lab_math::translate(light.position),
            lab_math::scaleMatrix({markerSize, markerSize, markerSize}));
        writeModelUniforms(spotMarkerSlots[i], markerModel, light.color, noSpecular, 1.0f, false);
        const float radius = coneLength * std::tan(lab_math::radians(light.outerAngleDegrees));
        const Mat4 coneModel = lab_math::multiply(
            lab_math::translate(light.position),
            lab_math::multiply(
                lab_math::directionBasis(safeNormalize(light.direction, {0.0f, 1.0f, 0.0f})),
                lab_math::scaleMatrix({radius, radius, coneLength})));
        writeModelUniforms(spotConeSlots[i], coneModel, white, noSpecular, 1.0f, false);
    }
}

void drawModel(const graphics::internal::FrameData& fd, const Mesh& mesh,
               const ModelSlot& slot) {
    const VkDeviceSize vertexBufferOffset = 0;
    vkCmdBindVertexBuffers(fd.command_buffer, 0, 1,
                           &mesh.vertexBuffer.buffer, &vertexBufferOffset);
    vkCmdBindIndexBuffer(fd.command_buffer, mesh.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout, 1, 1, &slot.descriptorSet, 0, nullptr);
    vkCmdDrawIndexed(fd.command_buffer, mesh.triangleIndexCount, 1, 0, 0, 0);
}

void setupScene() {
    const auto setup = [](SceneObject& object, const char* name, const Mesh* mesh,
                          Vec3 position, Vec3 scale, Material material) {
        object.name = name;
        object.mesh = mesh;
        object.position = position;
        object.scale = scale;
        object.material = material;
    };
    setup(objects[0], "Icosahedron A", &icosahedronMesh,
          {-2.6f, 0.0f, 6.0f}, {1.0f, 1.0f, 1.0f},
          {{0.9f, 0.25f, 0.25f}, {1.0f, 1.0f, 1.0f}, 64.0f});
    setup(objects[1], "Icosahedron B", &icosahedronMesh,
          {0.0f, 0.0f, 6.5f}, {1.2f, 1.2f, 1.2f},
          {{0.25f, 0.8f, 0.35f}, {1.0f, 1.0f, 1.0f}, 16.0f});
    // <-- ИСПРАВЛЕНО: был {0.8f, 1.3f, 0.8f}, теперь {1.0f, 1.0f, 1.0f}
    setup(objects[2], "Icosahedron C", &icosahedronMesh,
          {2.6f, 0.0f, 7.0f}, {1.0f, 1.0f, 1.0f},
          {{0.3f, 0.4f, 0.95f}, {1.0f, 1.0f, 1.0f}, 128.0f});
    for (uint32_t i = 0; i < 3; ++i) {
        objects[i].animatable = true;
        objects[i].animate = true;
        objects[i].phaseOffset = 2.0f * lab_math::PI * static_cast<float>(i) / 3.0f;
    }
    setup(objects[3], "Floor", &cubeMesh,
          {0.0f, 3.1f, 4.0f}, {16.0f, 0.2f, 14.0f},
          {{0.6f, 0.6f, 0.6f}, {0.3f, 0.3f, 0.3f}, 8.0f});
    setup(objects[4], "Back wall", &cubeMesh,
          {0.0f, -1.0f, 10.1f}, {16.0f, 8.2f, 0.2f},
          {{0.75f, 0.7f, 0.6f}, {0.2f, 0.2f, 0.2f}, 8.0f});
    setup(objects[5], "Left wall", &cubeMesh,
          {-8.1f, -1.0f, 4.0f}, {0.2f, 8.2f, 14.0f},
          {{0.6f, 0.7f, 0.8f}, {0.2f, 0.2f, 0.2f}, 8.0f});
    setup(objects[6], "Right wall", &cubeMesh,
          {8.1f, -1.0f, 4.0f}, {0.2f, 8.2f, 14.0f},
          {{0.8f, 0.65f, 0.65f}, {0.2f, 0.2f, 0.2f}, 8.0f});
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
    if (!createMeshes()) {
        std::cerr << "Failed to create meshes\n";
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
    for (SceneObject& object : objects) destroyBuffer(object.slot.uniformBuffer);
    for (ModelSlot& slot : pointMarkerSlots) destroyBuffer(slot.uniformBuffer);
    for (ModelSlot& slot : spotMarkerSlots) destroyBuffer(slot.uniformBuffer);
    for (ModelSlot& slot : spotConeSlots) destroyBuffer(slot.uniformBuffer);
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
    vkDestroyPipeline(context.device, edgePipeline, nullptr);  // <-- ДОБАВЛЕНО
    vkDestroyShaderModule(context.device, litVertexShader, nullptr);
    vkDestroyShaderModule(context.device, litFragmentShader, nullptr);
    vkDestroyShaderModule(context.device, unlitVertexShader, nullptr);
    vkDestroyShaderModule(context.device, unlitFragmentShader, nullptr);
    vkDestroyShaderModule(context.device, edgeVertexShader, nullptr);   // <-- ДОБАВЛЕНО
    vkDestroyShaderModule(context.device, edgeFragmentShader, nullptr); // <-- ДОБАВЛЕНО
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
    updateObjectUniforms();
    updateMarkerUniforms();
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

    // Освещаемые модели
    vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, litPipeline);
    for (const SceneObject& object : objects) {
        drawModel(fd, *object.mesh, object.slot);
    }


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
                vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        pipelineLayout, 1, 1, &object.slot.descriptorSet, 0, nullptr);
                vkCmdDrawIndexed(fd.command_buffer, icosahedronMesh.edgeIndexCount, 1,
                                 icosahedronMesh.edgeFirstIndex, 0, 0);
            }
        }
    }

    // Маркеры источников света
    if (showLightMarkers) {
        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, markerPipeline);
        for (size_t i = 0; i < pointLights.size(); ++i) {
            drawModel(fd, cubeMesh, pointMarkerSlots[i]);
        }
        for (size_t i = 0; i < spotLights.size(); ++i) {
            drawModel(fd, cubeMesh, spotMarkerSlots[i]);
        }
    }
    // Конусы прожекторов
    if (showSpotCones) {
        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, linePipeline);
        for (size_t i = 0; i < spotLights.size(); ++i) {
            drawModel(fd, coneMesh, spotConeSlots[i]);
        }
    }
    vkCmdEndRenderPass(fd.command_buffer);
    if (vkEndCommandBuffer(fd.command_buffer) != VK_SUCCESS) {
        std::cerr << "Failed to end application command buffer\n";
    }
}

} // namespace application