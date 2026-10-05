#include "application.hpp"
#include "icosahedron.hpp"
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


constexpr uint32_t object_count = 3;

constexpr float icosahedron_radius = 1.0f;

enum class ProjectionType {
    Perspective = 0,
    Orthographic = 1,
};


struct GlobalUniforms {
    float projection[16];
};


struct ObjectUniforms {
    float model[16];
    float baseColor[4];
};

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

struct Icosahedron {
    const char* name = "";

    Vec3 position{};
    Vec3 rotationDegrees{};
    Vec3 scale{1.0f, 1.0f, 1.0f};
    float color[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    float phaseOffset = 0.0f;
    float animationMultiplier = 1.0f;
    bool animate = true;

    Mat4 modelMatrix{};

    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    GpuBuffer uniformBuffer;
};



Mesh mesh;
Icosahedron objects[object_count];

VkDescriptorSetLayout globalDescriptorSetLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout objectDescriptorSetLayout = VK_NULL_HANDLE;
VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
VkDescriptorSet globalDescriptorSet = VK_NULL_HANDLE;
GpuBuffer globalUniformBuffer;

VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
VkPipeline fillPipeline = VK_NULL_HANDLE;
VkPipeline edgePipeline = VK_NULL_HANDLE;

VkShaderModule fillVertexShader = VK_NULL_HANDLE;
VkShaderModule fillFragmentShader = VK_NULL_HANDLE;
VkShaderModule edgeVertexShader = VK_NULL_HANDLE;
VkShaderModule edgeFragmentShader = VK_NULL_HANDLE;

ProjectionType projectionType = ProjectionType::Perspective;

float orthographicHeight = 6.0f;
float perspectiveFovDegrees = 55.0f;
float nearPlane = 0.1f;
float farPlane = 50.0f;

bool showEdges = true;

bool timeInitialized = false;
double previousWallTime = 0.0;
double animationTime = 0.0;

bool animationPlaying = true;
bool resetAnimationRequested = false;
float animationSpeed = 1.0f;


float trajectoryRadius = 1.0f;
float trajectoryVerticalAmplitude = 0.8f;
float trajectoryDepthAmplitude = 0.7f;
float trajectoryVerticalFrequency = 2.0f;
float trajectoryDepthFrequency = 0.5f;
Vec3 animatedRotationSpeedDegrees{20.0f, 30.0f, 40.0f};



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
                        &result.buffer, &result.allocation,
                        &mappedInfo) != VK_SUCCESS) {
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
        if (file) {
            break;
        }
        file.clear();
    }

    if (!file) {
        std::cerr << "Failed to open shader: " << name
                  << " (compile shaders with glslc first)\n";
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


bool createMesh() {
    const MeshData data = lab_geometry::makeIcosahedron(icosahedron_radius);

    mesh.triangleIndexCount = data.triangleIndexCount;
    mesh.edgeFirstIndex = data.edgeFirstIndex;
    mesh.edgeIndexCount = data.edgeIndexCount;

    const size_t vertexBytes = sizeof(Vertex) * data.vertices.size();
    const size_t indexBytes = sizeof(uint32_t) * data.indices.size();

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



bool createDescriptorLayouts() {
    // set 0: GlobalUniforms, нужен только вершинному шейдеру
    const VkDescriptorSetLayoutBinding globalBinding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
    };

    const VkDescriptorSetLayoutCreateInfo globalLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &globalBinding,
    };

    if (vkCreateDescriptorSetLayout(context.device, &globalLayoutInfo, nullptr,
                                    &globalDescriptorSetLayout) != VK_SUCCESS) {
        std::cerr << "Failed to create global descriptor set layout\n";
        return false;
    }

    
    const VkDescriptorSetLayoutBinding objectBinding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
    };

    const VkDescriptorSetLayoutCreateInfo objectLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &objectBinding,
    };

    if (vkCreateDescriptorSetLayout(context.device, &objectLayoutInfo, nullptr,
                                    &objectDescriptorSetLayout) != VK_SUCCESS) {
        std::cerr << "Failed to create object descriptor set layout\n";
        return false;
    }

    return true;
}

bool allocateAndWriteDescriptorSet(VkDescriptorSetLayout layout,
                                   const GpuBuffer& uniformBuffer,
                                   VkDeviceSize range,
                                   VkDescriptorSet& descriptorSet) {
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

    const VkDescriptorBufferInfo bufferInfo = {
        .buffer = uniformBuffer.buffer,
        .offset = 0,
        .range = range,
    };

    const VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = descriptorSet,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &bufferInfo,
    };

    vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);

    return true;
}

bool createDescriptorPoolAndSets() {
    
    constexpr uint32_t setCount = object_count + 1;

    const VkDescriptorPoolSize poolSize = {
        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = setCount,
    };

    const VkDescriptorPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = setCount,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize,
    };

    if (vkCreateDescriptorPool(context.device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        std::cerr << "Failed to create application descriptor pool\n";
        return false;
    }

    if (!createHostBuffer(sizeof(GlobalUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                          globalUniformBuffer)) {
        return false;
    }

    if (!allocateAndWriteDescriptorSet(globalDescriptorSetLayout, globalUniformBuffer,
                                       sizeof(GlobalUniforms), globalDescriptorSet)) {
        return false;
    }

    for (Icosahedron& object : objects) {
        if (!createHostBuffer(sizeof(ObjectUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                              object.uniformBuffer)) {
            return false;
        }

        if (!allocateAndWriteDescriptorSet(objectDescriptorSetLayout, object.uniformBuffer,
                                           sizeof(ObjectUniforms), object.descriptorSet)) {
            return false;
        }
    }

    return true;
}



VkPipelineShaderStageCreateInfo makeShaderStage(VkShaderStageFlagBits stage,
                                                VkShaderModule module) {
    return {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = stage,
        .module = module,
        .pName = "main",
    };
}

bool createPipelines() {
    const VkDescriptorSetLayout setLayouts[2] = {
        globalDescriptorSetLayout, // set 0
        objectDescriptorSetLayout, // set 1
    };

    const VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 2,
        .pSetLayouts = setLayouts,
    };

    if (vkCreatePipelineLayout(context.device, &pipelineLayoutInfo, nullptr,
                               &pipelineLayout) != VK_SUCCESS) {
        std::cerr << "Failed to create pipeline layout\n";
        return false;
    }

    if (!loadShaderModule("icosahedron.vert.spv", fillVertexShader) ||
        !loadShaderModule("icosahedron.frag.spv", fillFragmentShader) ||
        !loadShaderModule("edge.vert.spv", edgeVertexShader) ||
        !loadShaderModule("edge.frag.spv", edgeFragmentShader)) {
        return false;
    }

   
    const VkVertexInputBindingDescription binding = {
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };

    const VkVertexInputAttributeDescription attributes[] = {
        {
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, position),
        },
        {
            .location = 1,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, color),
        },
    };

    const VkPipelineVertexInputStateCreateInfo vertexInput = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
        .pVertexAttributeDescriptions = attributes,
    };

    const VkPipelineInputAssemblyStateCreateInfo fillAssembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    const VkPipelineInputAssemblyStateCreateInfo edgeAssembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
    };

    const VkPipelineViewportStateCreateInfo viewportState = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };

    
    const VkPipelineRasterizationStateCreateInfo fillRaster = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_CLOCKWISE,
        .depthBiasEnable = VK_TRUE,
        .depthBiasConstantFactor = 1.0f,
        .depthBiasSlopeFactor = 1.0f,
        .lineWidth = 1.0f,
    };

    const VkPipelineRasterizationStateCreateInfo edgeRaster = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_CLOCKWISE,
        .lineWidth = 1.0f,
    };

    const VkPipelineMultisampleStateCreateInfo multisampleState = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };

    const VkPipelineDepthStencilStateCreateInfo fillDepth = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
    };

    
    const VkPipelineDepthStencilStateCreateInfo edgeDepth = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_FALSE,
        .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
    };

    const VkPipelineColorBlendAttachmentState blendAttachment = {
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                          VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT |
                          VK_COLOR_COMPONENT_A_BIT,
    };

    const VkPipelineColorBlendStateCreateInfo blendState = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &blendAttachment,
    };

    const VkDynamicState dynamicStates[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };

    const VkPipelineDynamicStateCreateInfo dynamicState = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = sizeof(dynamicStates) / sizeof(dynamicStates[0]),
        .pDynamicStates = dynamicStates,
    };

    const VkPipelineShaderStageCreateInfo fillStages[] = {
        makeShaderStage(VK_SHADER_STAGE_VERTEX_BIT, fillVertexShader),
        makeShaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, fillFragmentShader),
    };

    const VkPipelineShaderStageCreateInfo edgeStages[] = {
        makeShaderStage(VK_SHADER_STAGE_VERTEX_BIT, edgeVertexShader),
        makeShaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, edgeFragmentShader),
    };

    const VkGraphicsPipelineCreateInfo fillPipelineInfo = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = fillStages,
        .pVertexInputState = &vertexInput,
        .pInputAssemblyState = &fillAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &fillRaster,
        .pMultisampleState = &multisampleState,
        .pDepthStencilState = &fillDepth,
        .pColorBlendState = &blendState,
        .pDynamicState = &dynamicState,
        .layout = pipelineLayout,
        .renderPass = context.render_pass,
    };

    if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
                                  &fillPipelineInfo, nullptr,
                                  &fillPipeline) != VK_SUCCESS) {
        std::cerr << "Failed to create fill pipeline\n";
        return false;
    }

    const VkGraphicsPipelineCreateInfo edgePipelineInfo = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = edgeStages,
        .pVertexInputState = &vertexInput,
        .pInputAssemblyState = &edgeAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &edgeRaster,
        .pMultisampleState = &multisampleState,
        .pDepthStencilState = &edgeDepth,
        .pColorBlendState = &blendState,
        .pDynamicState = &dynamicState,
        .layout = pipelineLayout,
        .renderPass = context.render_pass,
    };

    if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
                                  &edgePipelineInfo, nullptr,
                                  &edgePipeline) != VK_SUCCESS) {
        std::cerr << "Failed to create edge pipeline\n";
        return false;
    }

    return true;
}


Mat4 makeModelMatrix(const Icosahedron& object) {
    const float phase = static_cast<float>(
        animationTime * object.animationMultiplier + object.phaseOffset);

    Vec3 animatedOffset{};
    Vec3 animatedRotationDegrees{};

    if (object.animate) {
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



void drawProjectionUI() {
    if (!ImGui::CollapsingHeader("Projection", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    int projection = static_cast<int>(projectionType);

    const char* projectionItems[] = {
        "Perspective",
        "Orthographic",
    };

    if (ImGui::Combo("Projection type", &projection, projectionItems, 2)) {
        projectionType = static_cast<ProjectionType>(projection);
    }

    if (projectionType == ProjectionType::Perspective) {
        ImGui::SliderFloat("FOV Y", &perspectiveFovDegrees, 25.0f, 120.0f, "%.1f deg");
    } else {
        ImGui::SliderFloat("Ortho area height", &orthographicHeight, 2.0f, 20.0f, "%.2f");
    }

    if (ImGui::SliderFloat("Near", &nearPlane, 0.01f, 10.0f, "%.2f")) {
        nearPlane = std::min(nearPlane, farPlane - 0.01f);
    }

    if (ImGui::SliderFloat("Far", &farPlane, 1.0f, 100.0f, "%.1f")) {
        farPlane = std::max(farPlane, nearPlane + 0.01f);
    }
}

void drawAnimationUI() {
    if (!ImGui::CollapsingHeader("Animation & Trajectory", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    if (ImGui::Button(animationPlaying ? "Pause" : "Play")) {
        animationPlaying = !animationPlaying;
    }

    ImGui::SameLine();
    if (ImGui::Button("Reset time")) {
        resetAnimationRequested = true;
    }

    ImGui::SliderFloat("Speed", &animationSpeed, 0.0f, 3.0f, "%.2fx");
    ImGui::SliderFloat("Trajectory radius", &trajectoryRadius, 0.0f, 2.5f, "%.2f");
    ImGui::SliderFloat("Y amplitude", &trajectoryVerticalAmplitude, 0.0f, 2.5f, "%.2f");
    ImGui::SliderFloat("Z amplitude", &trajectoryDepthAmplitude, 0.0f, 2.0f, "%.2f");
    ImGui::SliderFloat("Y frequency", &trajectoryVerticalFrequency, 0.1f, 5.0f, "%.2f");
    ImGui::SliderFloat("Z frequency", &trajectoryDepthFrequency, 0.1f, 3.0f, "%.2f");
    ImGui::DragFloat3("Rotation speed", &animatedRotationSpeedDegrees.x,
                      1.0f, -180.0f, 180.0f, "%.1f deg/s");
}

void drawObjectUI(Icosahedron& object) {
    if (!ImGui::TreeNode(object.name)) {
        return;
    }

    ImGui::Checkbox("Animate", &object.animate);
    ImGui::DragFloat("Speed multiplier", &object.animationMultiplier,
                     0.01f, 0.0f, 3.0f, "%.2f");
    ImGui::DragFloat("Phase offset", &object.phaseOffset,
                     0.01f, -lab_math::PI, lab_math::PI, "%.2f rad");

    ImGui::Separator();

    ImGui::DragFloat3("Position", &object.position.x, 0.01f, -10.0f, 10.0f, "%.2f");
    ImGui::DragFloat3("Rotation", &object.rotationDegrees.x,
                      1.0f, -360.0f, 360.0f, "%.1f deg");
    ImGui::DragFloat3("Scale", &object.scale.x, 0.01f, 0.1f, 4.0f, "%.2f");

    
    ImGui::ColorEdit3("Color", object.color);

    ImGui::TreePop();
}

void drawUI() {
    ImGui::Begin("Vulkan Lab 1 - Icosahedron");

    drawProjectionUI();
    drawAnimationUI();

    if (ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Show edges", &showEdges);
        ImGui::Text("Vertices: 12, edges: %u, faces: %u",
                    mesh.edgeIndexCount / 2, mesh.triangleIndexCount / 3);
    }

    if (ImGui::CollapsingHeader("Objects", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (Icosahedron& object : objects) {
            drawObjectUI(object);
        }
    }

    ImGui::End();
}



void updateGlobalUniform() {
    GlobalUniforms global{};

    lab_math::toColumnMajor(makeProjectionMatrix(), global.projection);

    writeToBuffer(globalUniformBuffer, &global, sizeof(global));
}

void updateObjectUniform(Icosahedron& object) {
    ObjectUniforms uniforms{};

    object.modelMatrix = makeModelMatrix(object);
    lab_math::toColumnMajor(object.modelMatrix, uniforms.model);

    std::memcpy(uniforms.baseColor, object.color, sizeof(uniforms.baseColor));

    writeToBuffer(object.uniformBuffer, &uniforms, sizeof(uniforms));
}

} 

bool initialize() {
    objects[0].name = "Icosahedron A";
    objects[0].position = {-2.6f, 0.0f, 6.0f};
    objects[0].color[0] = 1.0f;
    objects[0].color[1] = 1.0f;
    objects[0].color[2] = 1.0f;
    objects[0].phaseOffset = 0.0f;

    objects[1].name = "Icosahedron B";
    objects[1].position = {0.0f, 0.0f, 6.5f};
    objects[1].scale = {1.2f, 1.2f, 1.2f};
    objects[1].color[0] = 1.0f;
    objects[1].color[1] = 0.8f;
    objects[1].color[2] = 0.7f;
    objects[1].phaseOffset = 2.0f * lab_math::PI / 3.0f;

    objects[2].name = "Icosahedron C";
    objects[2].position = {2.6f, 0.0f, 7.0f};
    objects[2].scale = {1.0f, 1.0f, 1.0f};
    objects[2].color[0] = 0.7f;
    objects[2].color[1] = 0.85f;
    objects[2].color[2] = 1.0f;
    objects[2].phaseOffset = 4.0f * lab_math::PI / 3.0f;

    if (!createMesh()) {
        std::cerr << "Failed to create icosahedron mesh\n";
        return false;
    }

    if (!createDescriptorLayouts()) {
        return false;
    }

    if (!createDescriptorPoolAndSets()) {
        return false;
    }

    if (!createPipelines()) {
        return false;
    }

    timeInitialized = false;
    animationTime = 0.0;
    animationPlaying = true;

    return true;
}

void shutdown() {
    vkQueueWaitIdle(context.graphics_queue);

    for (Icosahedron& object : objects) {
        destroyBuffer(object.uniformBuffer);
    }

    destroyBuffer(globalUniformBuffer);
    destroyBuffer(mesh.vertexBuffer);
    destroyBuffer(mesh.indexBuffer);

    vkDestroyPipeline(context.device, fillPipeline, nullptr);
    vkDestroyPipeline(context.device, edgePipeline, nullptr);

    vkDestroyShaderModule(context.device, fillVertexShader, nullptr);
    vkDestroyShaderModule(context.device, fillFragmentShader, nullptr);
    vkDestroyShaderModule(context.device, edgeVertexShader, nullptr);
    vkDestroyShaderModule(context.device, edgeFragmentShader, nullptr);

    vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);

   
    vkDestroyDescriptorPool(context.device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(context.device, globalDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(context.device, objectDescriptorSetLayout, nullptr);
}

void update(double time) {
    if (!timeInitialized) {
        previousWallTime = time;
        timeInitialized = true;
    }

   
    const double delta = std::min(time - previousWallTime, 0.1);
    previousWallTime = time;

    drawUI();

    if (resetAnimationRequested) {
        animationTime = 0.0;
        resetAnimationRequested = false;
    } else if (animationPlaying) {
        animationTime += delta * static_cast<double>(animationSpeed);
    }
}

void render(const graphics::internal::FrameData& fd) {
   
    updateGlobalUniform();
    for (Icosahedron& object : objects) {
        updateObjectUniform(object);
    }

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
        {
            .color = {
                .float32 = {0.035f, 0.04f, 0.055f, 1.0f},
            },
        },
        {
            .depthStencil = {1.0f, 0},
        },
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
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(context.swapchain_extent.width),
        .height = static_cast<float>(context.swapchain_extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    const VkRect2D scissor = { .extent = context.swapchain_extent };

    vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
    vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

    
    const VkDeviceSize vertexBufferOffset = 0;

    vkCmdBindVertexBuffers(fd.command_buffer, 0, 1,
                           &mesh.vertexBuffer.buffer, &vertexBufferOffset);
    vkCmdBindIndexBuffer(fd.command_buffer, mesh.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

   
    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout, 0, 1, &globalDescriptorSet, 0, nullptr);

    
    vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, fillPipeline);

    for (const Icosahedron& object : objects) {
        
        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayout, 1, 1, &object.descriptorSet, 0, nullptr);

        vkCmdDrawIndexed(fd.command_buffer, mesh.triangleIndexCount, 1, 0, 0, 0);
    }

    if (showEdges) {
        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, edgePipeline);

        for (const Icosahedron& object : objects) {
            vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipelineLayout, 1, 1, &object.descriptorSet, 0, nullptr);

            vkCmdDrawIndexed(fd.command_buffer, mesh.edgeIndexCount, 1,
                             mesh.edgeFirstIndex, 0, 0);
        }
    }

    vkCmdEndRenderPass(fd.command_buffer);

    if (vkEndCommandBuffer(fd.command_buffer) != VK_SUCCESS) {
        std::cerr << "Failed to end application command buffer\n";
    }
}

} 
