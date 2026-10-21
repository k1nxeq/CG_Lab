#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>
#include "math.hpp"

namespace lab_geometry {

struct Vertex {
    float position[3];
    float normal[3];
    float smoothNormal[3];
};

struct MeshData {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    uint32_t triangleIndexCount = 0;
    uint32_t edgeFirstIndex = 0;
    uint32_t edgeIndexCount = 0;
};

namespace detail {

inline Vertex makeVertex(const lab_math::Vec3& position,
                         const lab_math::Vec3& normal,
                         const lab_math::Vec3& smoothNormal) {
    return {
        {position.x, position.y, position.z},
        {normal.x, normal.y, normal.z},
        {smoothNormal.x, smoothNormal.y, smoothNormal.z},
    };
}

inline lab_math::Vec3 positionOf(const MeshData& mesh, uint32_t index) {
    const Vertex& v = mesh.vertices[index];
    return {v.position[0], v.position[1], v.position[2]};
}

inline void pushTriangle(MeshData& mesh, uint32_t a, uint32_t b, uint32_t c,
                         const lab_math::Vec3& outward) {
    const lab_math::Vec3 normal = lab_math::cross(
        lab_math::sub(positionOf(mesh, b), positionOf(mesh, a)),
        lab_math::sub(positionOf(mesh, c), positionOf(mesh, a)));
    if (lab_math::dot(normal, outward) > 0.0f) {
        std::swap(b, c);
    }
    mesh.indices.push_back(a);
    mesh.indices.push_back(b);
    mesh.indices.push_back(c);
}

} // namespace detail

inline MeshData makeIcosahedron(float radius) {
    const float phi = (1.0f + std::sqrt(5.0f)) * 0.5f;
    const lab_math::Vec3 base[12] = {
        {-1.0f,  phi, 0.0f}, { 1.0f,  phi, 0.0f},
        {-1.0f, -phi, 0.0f}, { 1.0f, -phi, 0.0f},
        {0.0f, -1.0f,  phi}, {0.0f,  1.0f,  phi},
        {0.0f, -1.0f, -phi}, {0.0f,  1.0f, -phi},
        { phi, 0.0f, -1.0f}, { phi, 0.0f,  1.0f},
        {-phi, 0.0f, -1.0f}, {-phi, 0.0f,  1.0f},
    };
    const uint32_t faces[20][3] = {
        {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
        {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
        {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
        {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1},
    };
    lab_math::Vec3 positions[12];
    for (int i = 0; i < 12; ++i) {
        positions[i] = lab_math::scale(lab_math::normalize(base[i]), radius);
    }
    MeshData mesh;
    mesh.vertices.reserve(60);
    mesh.indices.reserve(120);

    for (const auto& face : faces) {
        const lab_math::Vec3& a = positions[face[0]];
        const lab_math::Vec3& b = positions[face[1]];
        const lab_math::Vec3& c = positions[face[2]];
        lab_math::Vec3 faceNormal = lab_math::normalize(
            lab_math::cross(lab_math::sub(b, a), lab_math::sub(c, a)));
        const lab_math::Vec3 centroid = lab_math::scale(
            lab_math::add(lab_math::add(a, b), c), 1.0f / 3.0f);
        if (lab_math::dot(faceNormal, centroid) < 0.0f) {
            faceNormal = lab_math::scale(faceNormal, -1.0f);
        }
        const uint32_t first = static_cast<uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back(detail::makeVertex(a, faceNormal, lab_math::normalize(a)));
        mesh.vertices.push_back(detail::makeVertex(b, faceNormal, lab_math::normalize(b)));
        mesh.vertices.push_back(detail::makeVertex(c, faceNormal, lab_math::normalize(c)));
        detail::pushTriangle(mesh, first, first + 1, first + 2, faceNormal);
    }

    // Сохраняем количество индексов треугольников
    mesh.triangleIndexCount = static_cast<uint32_t>(mesh.indices.size());

    // Генерируем рёбра (30 рёбер × 2 вершины = 60 индексов)
    mesh.edgeFirstIndex = static_cast<uint32_t>(mesh.indices.size());
    for (uint32_t i = 0; i < 20; ++i) {
        uint32_t first = i * 3;
        mesh.indices.push_back(first);
        mesh.indices.push_back(first + 1);
        mesh.indices.push_back(first + 1);
        mesh.indices.push_back(first + 2);
        mesh.indices.push_back(first + 2);
        mesh.indices.push_back(first);
    }
    mesh.edgeIndexCount = static_cast<uint32_t>(mesh.indices.size()) - mesh.edgeFirstIndex;

    return mesh;
}

inline MeshData makeCube() {
    MeshData mesh;
    mesh.vertices.reserve(24);
    mesh.indices.reserve(36);
    const float corners[4][2] = {
        {-0.5f, -0.5f}, {0.5f, -0.5f}, {0.5f, 0.5f}, {-0.5f, 0.5f},
    };
    for (int axis = 0; axis < 3; ++axis) {
        for (int sign = -1; sign <= 1; sign += 2) {
            const int u = (axis + 1) % 3;
            const int v = (axis + 2) % 3;
            float normalArray[3] = {0.0f, 0.0f, 0.0f};
            normalArray[axis] = static_cast<float>(sign);
            const lab_math::Vec3 normal = {normalArray[0], normalArray[1], normalArray[2]};
            const uint32_t first = static_cast<uint32_t>(mesh.vertices.size());
            for (const auto& corner : corners) {
                float p[3];
                p[axis] = 0.5f * static_cast<float>(sign);
                p[u] = corner[0];
                p[v] = corner[1];
                mesh.vertices.push_back(
                    detail::makeVertex({p[0], p[1], p[2]}, normal, normal));
            }
            detail::pushTriangle(mesh, first, first + 1, first + 2, normal);
            detail::pushTriangle(mesh, first, first + 2, first + 3, normal);
        }
    }
    mesh.triangleIndexCount = static_cast<uint32_t>(mesh.indices.size());
    return mesh;
}

inline MeshData makeCone(uint32_t segments) {
    MeshData mesh;
    const lab_math::Vec3 zero = {0.0f, 0.0f, 0.0f};
    mesh.vertices.push_back(detail::makeVertex({0.0f, 0.0f, 0.0f}, zero, zero));
    mesh.vertices.push_back(detail::makeVertex({0.0f, 0.0f, 1.0f}, zero, zero));
    for (uint32_t i = 0; i < segments; ++i) {
        const float angle = 2.0f * lab_math::PI * static_cast<float>(i) /
                            static_cast<float>(segments);
        mesh.vertices.push_back(detail::makeVertex(
            {std::cos(angle), std::sin(angle), 1.0f}, zero, zero));
    }
    for (uint32_t i = 0; i < segments; ++i) {
        mesh.indices.push_back(2 + i);
        mesh.indices.push_back(2 + (i + 1) % segments);
    }
    const uint32_t step = std::max<uint32_t>(segments / 4, 1);
    for (uint32_t i = 0; i < segments; i += step) {
        mesh.indices.push_back(0);
        mesh.indices.push_back(2 + i);
    }
    mesh.indices.push_back(0);
    mesh.indices.push_back(1);
    mesh.triangleIndexCount = static_cast<uint32_t>(mesh.indices.size());
    return mesh;
}

} // namespace lab_geometry