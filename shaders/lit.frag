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

// Текстурные сэмплеры материала (set = 1, bindings = 1, 2, 3)
layout(set = 1, binding = 1) uniform sampler2D albedo_texture;
layout(set = 1, binding = 2) uniform sampler2D specular_texture;
layout(set = 1, binding = 3) uniform sampler2D emissive_texture;

// Push-константы для управления нетривиальным сэмплированием (слайд 18 лекции #6)
layout(std140, push_constant) uniform PipelineConsts {
    uint nontrivial_sampling;
    float time;
} push_consts;

layout(location = 0) in vec3 in_world_position;
layout(location = 1) in vec3 in_world_normal;
layout(location = 2) in vec2 in_uv;

layout(location = 0) out vec4 out_color;

vec3 blinn_phong(vec3 n, vec3 v, vec3 l, vec3 radiance, vec3 kd, vec3 ks) {
    float n_dot_l = max(dot(n, l), 0.0);
    vec3 h = normalize(l + v);
    float n_dot_h = max(dot(n, h), 0.0);

    vec3 diffuse = kd * n_dot_l;
    vec3 specular = ks * pow(n_dot_h, model_uniforms.specular.a);

    if (n_dot_l <= 0.0) {
        specular = vec3(0.0);
    }

    return radiance * (diffuse + specular);
}

void main() {
    vec3 n = normalize(in_world_normal);
    vec3 v = normalize(scene.camera_position.xyz - in_world_position);

    vec3 sampled_albedo;
    vec3 sampled_specular;
    vec3 sampled_emissive;

    // Нетривиальное сэмплирование: множественная выборка разными UV от времени + модуляция цвета
    if (push_consts.nontrivial_sampling != 0u) {
        float t = push_consts.time;

        // Первая пара текстурных координат: волновая деформация
        vec2 uv1 = in_uv + 0.08 * vec2(
            sin(in_uv.y * 10.0 + t * 2.5),
            cos(in_uv.x * 10.0 - t * 2.0)
        );

        // Вторая пара текстурных координат: вращение и пульсация масштаба вокруг центра
        float angle = t * 0.8;
        mat2 rot = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
        float scale = 1.15 + 0.25 * sin(t * 1.7);
        vec2 uv2 = rot * ((in_uv - 0.5) * scale) + 0.5;

        // Множественное сэмплирование одного и того же изображения разными UV
        vec3 tex1 = texture(albedo_texture, uv1).rgb;
        vec3 tex2 = texture(albedo_texture, uv2).rgb;

        // Модуляция цвета сэмплированного изображения от времени
        vec3 color_modulation = 0.75 + 0.35 * cos(t * 2.2 + vec3(0.0, 2.1, 4.2) + in_uv.xyx * 3.14159);
        sampled_albedo = mix(tex1, tex2, 0.45) * color_modulation;

        vec3 spec1 = texture(specular_texture, uv1).rgb;
        vec3 spec2 = texture(specular_texture, uv2).rgb;
        sampled_specular = mix(spec1, spec2, 0.45);

        vec3 emis1 = texture(emissive_texture, uv1).rgb;
        vec3 emis2 = texture(emissive_texture, uv2).rgb;
        sampled_emissive = mix(emis1, emis2, 0.5) * (0.8 + 0.4 * sin(t * 3.5));
    } else {
        sampled_albedo = texture(albedo_texture, in_uv).rgb;
        sampled_specular = texture(specular_texture, in_uv).rgb;
        sampled_emissive = texture(emissive_texture, in_uv).rgb;
    }

    // Альбедо из текстуры (модулируется цветом модели, по умолчанию белым {1,1,1})
    vec3 kd = sampled_albedo * model_uniforms.albedo.rgb;
    // Specular текстура диктует, какая часть геометрии должна блестеть
    vec3 ks = sampled_specular * model_uniforms.specular.rgb;

    // Рассеянный свет (ambient)
    vec3 color = scene.ambient_light.rgb * kd;

    // Направленный источник
    vec3 directional_l = normalize(-scene.directional_light_direction.xyz);
    color += blinn_phong(n, v, directional_l, scene.directional_light_color.rgb, kd, ks);

    // Точечные источники
    for (uint i = 0u; i < scene.light_counts.x; ++i) {
        PointLight light = point_lights[i];

        vec3 to_light = light.position - in_world_position;
        float distance_squared = max(dot(to_light, to_light), 0.0001);
        vec3 l = to_light * inversesqrt(distance_squared);

        vec3 radiance = light.color * light.intensity / distance_squared;
        color += blinn_phong(n, v, l, radiance, kd, ks);
    }

    // Прожекторные источники
    for (uint i = 0u; i < scene.light_counts.y; ++i) {
        SpotLight light = spot_lights[i];

        vec3 to_light = light.position - in_world_position;
        float distance_squared = max(dot(to_light, to_light), 0.0001);
        vec3 l = to_light * inversesqrt(distance_squared);

        float cos_theta = dot(-l, light.direction);

        float spot_factor;
        if (light.cos_inner - light.cos_outer > 0.00001) {
            spot_factor = smoothstep(light.cos_outer, light.cos_inner, cos_theta);
        } else {
            spot_factor = step(light.cos_outer, cos_theta);
        }

        vec3 radiance = light.color * light.intensity * spot_factor / distance_squared;
        color += blinn_phong(n, v, l, radiance, kd, ks);
    }

    // Emissive текстура суммируется с посчитанным цветом после Блинн-Фонга (светится в темноте)
    color += sampled_emissive * model_uniforms.params.z;

    out_color = vec4(color, 1.0);
}