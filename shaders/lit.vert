#version 450 core

// set = 0: данные сцены (общие для всех объектов)
layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec4 camera_position;               // xyz
    vec4 directional_light_direction;   // xyz: куда светит свет (от источника в сцену)
    vec4 directional_light_color;       // rgb * интенсивность
    vec4 ambient_light;                 // rgb * интенсивность
    uvec4 light_counts;                 // x: точечные, y: прожекторные
} scene;

// set = 1: данные модели
layout(std140, set = 1, binding = 0) uniform ModelUniforms {
    mat4 model;
    vec4 albedo;     // rgb
    vec4 specular;   // rgb: цвет блика, a: коэффициент блика (shininess)
    vec4 params;     // x: 1 = сглаженные нормали
} model_uniforms;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_smooth_normal;

layout(location = 0) out vec3 out_world_position;
layout(location = 1) out vec3 out_world_normal;

void main() {
    vec4 world_position = model_uniforms.model * vec4(in_position, 1.0);

    // Нормаль нельзя умножать на матрицу модели напрямую: при неравномерном
    // масштабе она перестанет быть перпендикулярной поверхности. Для неё
    // нужна обратная транспонированная матрица.
    mat3 normal_matrix = transpose(inverse(mat3(model_uniforms.model)));
    vec3 local_normal = mix(in_normal, in_smooth_normal, model_uniforms.params.x);

    out_world_position = world_position.xyz;
    out_world_normal = normalize(normal_matrix * local_normal);

    gl_Position = scene.view_projection * world_position;
}
