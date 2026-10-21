#version 450 core

layout(std140, set = 1, binding = 0) uniform ModelUniforms {
    mat4 model;
    vec4 albedo;
    vec4 specular;
    vec4 params;
} model_uniforms;

layout(location = 0) out vec4 out_color;

void main() {
    // Постоянный цвет: маркеры источников света не освещаются.
    out_color = vec4(model_uniforms.albedo.rgb, 1.0);
}
