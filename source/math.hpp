#pragma once

#include <cmath>

namespace lab_math {

const float PI = 3.14159265358979323846f;

inline float radians(float degrees) {
    return degrees * PI / 180.0f;
}

// ---------------------------------------------------------------- Vec3

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

// ---------------------------------------------------------------- Mat4
//
// Матрицы хранятся по строкам (m[row][column]) и применяются к вектору-столбцу:
//     p' = M * p
// В GLSL матрица ожидается по столбцам, поэтому перед записью в uniform-буфер
// вызывается toColumnMajor().

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

// Перемещение (аффинное преобразование, однородные координаты).
inline Mat4 translate(const Vec3& t) {
    Mat4 result = identity();

    result.m[0][3] = t.x;
    result.m[1][3] = t.y;
    result.m[2][3] = t.z;

    return result;
}

// Растяжение.
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

// Ортографическая проекция (лекция 1): композиция сдвига и растяжения
// видового объёма в канонический объём Vulkan.
//   x -> [-1, 1], y -> [-1, 1], z -> [0, 1].
// В Vulkan ось Y экрана направлена вниз, поэтому "top" - это значение y,
// которое попадёт в NDC y = -1 (верх экрана), а "bottom" - в NDC y = +1.
// Параметры near/far названы nearPlane/farPlane: на Windows <windows.h>
// определяет макросы near и far.
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

// Перспективная проекция (лекция 1, слайд 28), z -> [0, 1], камера смотрит
// вдоль +Z, w' = z.
//   [ 1/(aspect*tan(fov/2))       0           0            0      ]
//   [          0             1/tan(fov/2)     0            0      ]
//   [          0                  0        f/(f-n)    -fn/(f-n)   ]
//   [          0                  0           1            0      ]
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

// ---------------------------------------------------------------- Камера (лекция 4)
//
// Камера - такой же объект в мире, как и модели: у неё есть матрица
//     C = T(position) * R,
// где R = Ry(yaw) * Rx(pitch) * Rz(roll). Чтобы объекты двигались
// относительно камеры, нужна обратная матрица:
//     V = C^-1 = R^-1 * T^-1 = R^T * T(-position)
// (обратное вращение - транспонирование, обратный сдвиг - смена знака,
// порядок множителей меняется на противоположный).
//
// Камера смотрит вдоль +Z, +X - вправо, +Y - вниз (как в Vulkan).
// Положительный pitch - взгляд вверх, положительный yaw - поворот вправо.

inline Mat4 transposeRotation(const Mat4& rotation) {
    Mat4 result = identity();

    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            result.m[row][column] = rotation.m[column][row];
        }
    }

    return result;
}

inline Mat4 cameraRotation(float pitchRadians, float yawRadians, float rollRadians) {
    return multiply(rotateY(yawRadians),
                    multiply(rotateX(pitchRadians), rotateZ(rollRadians)));
}

inline Mat4 viewMatrix(const Vec3& position,
                       float pitchRadians, float yawRadians, float rollRadians) {
    const Mat4 inverseRotation =
        transposeRotation(cameraRotation(pitchRadians, yawRadians, rollRadians));

    return multiply(inverseRotation, translate(scale(position, -1.0f)));
}

// Матрица поворота, переводящая локальную ось +Z в заданное направление.
// Нужна, чтобы повернуть конус прожектора вдоль его направления.
inline Mat4 directionBasis(const Vec3& direction) {
    const Vec3 forward = normalize(direction);
    const Vec3 reference = std::fabs(forward.y) < 0.99f ? Vec3{0.0f, 1.0f, 0.0f}
                                                        : Vec3{1.0f, 0.0f, 0.0f};
    const Vec3 right = normalize(cross(reference, forward));
    const Vec3 up = cross(forward, right);

    Mat4 result = identity();

    result.m[0][0] = right.x;   result.m[1][0] = right.y;   result.m[2][0] = right.z;
    result.m[0][1] = up.x;      result.m[1][1] = up.y;      result.m[2][1] = up.z;
    result.m[0][2] = forward.x; result.m[1][2] = forward.y; result.m[2][2] = forward.z;

    return result;
}

inline void toColumnMajor(const Mat4& matrix, float out[16]) {
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            out[column * 4 + row] = matrix.m[row][column];
        }
    }
}

} // namespace lab_math
