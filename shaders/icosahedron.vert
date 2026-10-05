#version 450 core

// set 0: данные кадра (общие для всех объектов)
layout(std140, set = 0, binding = 0) uniform GlobalUniforms {
    mat4 projection;
} globalUniforms;

// set 1: данные конкретного объекта (у каждого объекта свой набор дескрипторов)
layout(std140, set = 1, binding = 0) uniform ObjectUniforms {
    mat4 model;
    vec4 baseColor;
} objectUniforms;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 outColor;

void main() {
    gl_Position = globalUniforms.projection *
                  objectUniforms.model *
                  vec4(inPosition, 1.0);

    // Цвет вершины считается на CPU по локальной позиции вершины,
    // между вершинами его интерполирует GPU.
    outColor = inColor;
}
