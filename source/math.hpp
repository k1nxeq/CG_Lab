#pragma once

#include <cmath>

namespace lab_math {

const float PI = 3.14159265358979323846f;

inline float radians(float degrees) {
    return degrees * PI / 180.0f;
}


struct Vec3 {
    float x;
    float y;
    float z;
};

inline Vec3 add(const Vec3& a, const Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3 sub(const Vec3& a, const Vec3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3 scale(const Vec3& v, float s) {
    return {v.x * s, v.y * s, v.z * s};
}

inline float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

inline float length(const Vec3& v) {
    return std::sqrt(dot(v, v));
}

inline Vec3 normalize(const Vec3& v) {
    return scale(v, 1.0f / length(v));
}


struct Mat4 {
    float m[4][4]{};
};

inline Mat4 identity() {
    Mat4 result{};
    result.m[0][0] = 1.0f;
    result.m[1][1] = 1.0f;
    result.m[2][2] = 1.0f;
    result.m[3][3] = 1.0f;
    return result;
}

inline Mat4 multiply(const Mat4& a, const Mat4& b) {
    Mat4 result{};

    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            for (int k = 0; k < 4; ++k) {
                result.m[row][column] += a.m[row][k] * b.m[k][column];
            }
        }
    }

    return result;
}


inline Mat4 translate(const Vec3& t) {
    Mat4 result = identity();

    result.m[0][3] = t.x;
    result.m[1][3] = t.y;
    result.m[2][3] = t.z;

    return result;
}


inline Mat4 scaleMatrix(const Vec3& s) {
    Mat4 result = identity();

    result.m[0][0] = s.x;
    result.m[1][1] = s.y;
    result.m[2][2] = s.z;

    return result;
}

inline Mat4 rotateX(float angle) {
    Mat4 result = identity();
    const float c = std::cos(angle);
    const float s = std::sin(angle);

    result.m[1][1] = c;
    result.m[1][2] = -s;
    result.m[2][1] = s;
    result.m[2][2] = c;

    return result;
}

inline Mat4 rotateY(float angle) {
    Mat4 result = identity();
    const float c = std::cos(angle);
    const float s = std::sin(angle);

    result.m[0][0] = c;
    result.m[0][2] = s;
    result.m[2][0] = -s;
    result.m[2][2] = c;

    return result;
}

inline Mat4 rotateZ(float angle) {
    Mat4 result = identity();
    const float c = std::cos(angle);
    const float s = std::sin(angle);

    result.m[0][0] = c;
    result.m[0][1] = -s;
    result.m[1][0] = s;
    result.m[1][1] = c;

    return result;
}


inline Mat4 orthographic(float left, float right,
                         float top, float bottom,
                         float nearPlane, float farPlane) {
    Mat4 result{};

    result.m[0][0] = 2.0f / (right - left);
    result.m[1][1] = 2.0f / (bottom - top);
    result.m[2][2] = 1.0f / (farPlane - nearPlane);

    result.m[0][3] = -(right + left) / (right - left);
    result.m[1][3] = -(bottom + top) / (bottom - top);
    result.m[2][3] = -nearPlane / (farPlane - nearPlane);

    result.m[3][3] = 1.0f;

    return result;
}


inline Mat4 perspective(float fovYRadians, float aspect,
                        float nearPlane, float farPlane) {
    Mat4 result{};

    const float tanHalfFov = std::tan(fovYRadians * 0.5f);

    result.m[0][0] = 1.0f / (aspect * tanHalfFov);
    result.m[1][1] = 1.0f / tanHalfFov;
    result.m[2][2] = farPlane / (farPlane - nearPlane);
    result.m[2][3] = -farPlane * nearPlane / (farPlane - nearPlane);
    result.m[3][2] = 1.0f;

    return result;
}

inline void toColumnMajor(const Mat4& matrix, float out[16]) {
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            out[column * 4 + row] = matrix.m[row][column];
        }
    }
}

}