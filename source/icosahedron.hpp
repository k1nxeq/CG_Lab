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
    float color[3];
};

struct MeshData {
    std::vector<Vertex> vertices;

    
    std::vector<uint32_t> indices;

    uint32_t triangleIndexCount = 0;
    uint32_t edgeFirstIndex = 0;
    uint32_t edgeIndexCount = 0;
};


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

    MeshData mesh;
    mesh.vertices.reserve(12);

    lab_math::Vec3 positions[12];

    for (int i = 0; i < 12; ++i) {
        positions[i] = lab_math::scale(lab_math::normalize(base[i]), radius);

        const lab_math::Vec3& p = positions[i];
        mesh.vertices.push_back({
            {p.x, p.y, p.z},
            {
                0.5f + 0.5f * p.x / radius,
                0.5f + 0.5f * p.y / radius,
                0.5f + 0.5f * p.z / radius,
            },
        });
    }

    mesh.indices.reserve(60 + 60);

    for (const auto& face : faces) {
        uint32_t a = face[0];
        uint32_t b = face[1];
        uint32_t c = face[2];

        const lab_math::Vec3 normal = lab_math::cross(
            lab_math::sub(positions[b], positions[a]),
            lab_math::sub(positions[c], positions[a]));

        const lab_math::Vec3 centroid = lab_math::scale(
            lab_math::add(lab_math::add(positions[a], positions[b]), positions[c]),
            1.0f / 3.0f);

       
        if (lab_math::dot(normal, centroid) > 0.0f) {
            std::swap(b, c);
        }

        mesh.indices.push_back(a);
        mesh.indices.push_back(b);
        mesh.indices.push_back(c);
    }

    mesh.triangleIndexCount = static_cast<uint32_t>(mesh.indices.size());

    
    std::vector<std::pair<uint32_t, uint32_t>> edges;
    edges.reserve(60);

    for (size_t i = 0; i < mesh.triangleIndexCount; i += 3) {
        for (size_t k = 0; k < 3; ++k) {
            const uint32_t v0 = mesh.indices[i + k];
            const uint32_t v1 = mesh.indices[i + (k + 1) % 3];
            edges.emplace_back(std::min(v0, v1), std::max(v0, v1));
        }
    }

    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    mesh.edgeFirstIndex = static_cast<uint32_t>(mesh.indices.size());

    for (const auto& edge : edges) {
        mesh.indices.push_back(edge.first);
        mesh.indices.push_back(edge.second);
    }

    mesh.edgeIndexCount =
        static_cast<uint32_t>(mesh.indices.size()) - mesh.edgeFirstIndex;

    return mesh;
}

}
