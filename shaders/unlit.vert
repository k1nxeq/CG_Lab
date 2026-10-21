#version 450 core

layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec4 camera_position;
    vec4 directional_light_direction;
    vec4 directional_light_color;
    vec4 ambient_light;
    uvec4 light_counts;
} scene;

layout(std140, set = 1, binding = 0) uniform ModelUniforms {
    mat4 model;
    vec4 albedo;
    vec4 specular;
    vec4 params;
} model_uniforms;

layout(location = 0) in vec3 in_position;

void main() {
    gl_Position = scene.view_projection * model_uniforms.model * vec4(in_position, 1.0);
}
