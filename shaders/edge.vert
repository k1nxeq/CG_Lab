#version 450 core

layout(std140, set = 0, binding = 0) uniform GlobalUniforms {
    mat4 projection;
} globalUniforms;

layout(std140, set = 1, binding = 0) uniform ObjectUniforms {
    mat4 model;
    vec4 baseColor;
} objectUniforms;

layout(location = 0) in vec3 inPosition;

void main() {
    gl_Position = globalUniforms.projection *
                  objectUniforms.model *
                  vec4(inPosition, 1.0);
}
