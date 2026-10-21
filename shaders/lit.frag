#version 450 core

struct PointLight {
    vec3 position;
    float intensity;
    vec3 color;
    float padding;
};

struct SpotLight {
    vec3 position;
    float intensity;
    vec3 direction;     // нормированное направление оси
    float cos_inner;    // cos внутреннего угла
    vec3 color;
    float cos_outer;    // cos внешнего угла
};

layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec4 camera_position;
    vec4 directional_light_direction;
    vec4 directional_light_color;
    vec4 ambient_light;
    uvec4 light_counts;
} scene;

// Массивы источников света - storage buffer'ы с массивами динамической длины:
// максимальное число элементов в шейдере не задаётся.
layout(std430, set = 0, binding = 1) readonly buffer PointLights {
    PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight spot_lights[];
};

layout(std140, set = 1, binding = 0) uniform ModelUniforms {
    mat4 model;
    vec4 albedo;
    vec4 specular;
    vec4 params;
} model_uniforms;

layout(location = 0) in vec3 in_world_position;
layout(location = 1) in vec3 in_world_normal;

layout(location = 0) out vec4 out_color;

// Модель Блинна-Фонга для одного источника (слайд 23):
//   C = I * (cos(theta) * Kd + Ks * cos(phi)^alpha)
//   cos(theta) = n . l,  cos(phi) = n . h,  h = normalize(l + v)
// n - нормаль, v - направление на камеру, l - направление на источник,
// radiance - цвет * интенсивность света, дошедшего до точки.
vec3 blinn_phong(vec3 n, vec3 v, vec3 l, vec3 radiance) {
    float n_dot_l = max(dot(n, l), 0.0);
    vec3 h = normalize(l + v);
    float n_dot_h = max(dot(n, h), 0.0);

    vec3 diffuse = model_uniforms.albedo.rgb * n_dot_l;
    vec3 specular = model_uniforms.specular.rgb * pow(n_dot_h, model_uniforms.specular.a);

    // Блик не появляется на стороне, которую свет не освещает.
    if (n_dot_l <= 0.0) {
        specular = vec3(0.0);
    }

    return radiance * (diffuse + specular);
}

void main() {
    vec3 n = normalize(in_world_normal);
    vec3 v = normalize(scene.camera_position.xyz - in_world_position);

    // Рассеянный свет (ambient): I_a * K_a, в качестве K_a берём альбедо.
    vec3 color = scene.ambient_light.rgb * model_uniforms.albedo.rgb;

    // Направленный источник: все лучи параллельны, затухания нет.
    vec3 directional_l = normalize(-scene.directional_light_direction.xyz);
    color += blinn_phong(n, v, directional_l, scene.directional_light_color.rgb);

    // Точечные источники: затухание по закону обратных квадратов 1 / D^2.
    for (uint i = 0u; i < scene.light_counts.x; ++i) {
        PointLight light = point_lights[i];

        vec3 to_light = light.position - in_world_position;
        float distance_squared = max(dot(to_light, to_light), 0.0001);
        vec3 l = to_light * inversesqrt(distance_squared);

        vec3 radiance = light.color * light.intensity / distance_squared;
        color += blinn_phong(n, v, l, radiance);
    }

    // Прожекторные источники: то же затухание плюс ограничение конусом
    // с мягкими краями между внутренним и внешним углами.
    for (uint i = 0u; i < scene.light_counts.y; ++i) {
        SpotLight light = spot_lights[i];

        vec3 to_light = light.position - in_world_position;
        float distance_squared = max(dot(to_light, to_light), 0.0001);
        vec3 l = to_light * inversesqrt(distance_squared);

        // Косинус угла между осью прожектора и лучом от источника к точке.
        float cos_theta = dot(-l, light.direction);

        float spot_factor;
        if (light.cos_inner - light.cos_outer > 0.00001) {
            spot_factor = smoothstep(light.cos_outer, light.cos_inner, cos_theta);
        } else {
            // Внутренний угол равен внешнему: резкая граница.
            spot_factor = step(light.cos_outer, cos_theta);
        }

        vec3 radiance = light.color * light.intensity * spot_factor / distance_squared;
        color += blinn_phong(n, v, l, radiance);
    }

    out_color = vec4(color, 1.0);
}
