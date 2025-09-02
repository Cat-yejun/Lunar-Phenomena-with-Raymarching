from PIL import Image
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import glm
import time

AU = 1.495978707e11
rs = AU*1e-4

Sun_Radius = 695700e3
Mercury_Radius = 2440e3
Venus_Radius = 6051.84e3
Earth_Radius = 6371.01e3
Moon_Radius = 1737.53e3
Mars_Radius = 3389.92e3
Jupiter_Radius = 69911e3
Saturn_Radius = 58232e3
Uranus_Radius = 25362e3
Neptune_Radius = 24624e3

Radi = [Sun_Radius/rs, Mercury_Radius/rs, Venus_Radius/rs, Earth_Radius/rs, Moon_Radius/rs, Mars_Radius/rs, Jupiter_Radius/rs, Saturn_Radius/rs, Uranus_Radius/rs, Neptune_Radius/rs]
print(Radi)

vertex_shader = """
#version 330
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 aTexCoords;

out vec2 TexCoords;

void main() {
    gl_Position = vec4(position, 1.0);
    TexCoords = aTexCoords;
}
"""

fragment_shader = """
#version 330 core

uniform vec3 cameraPos;
uniform vec3 cameraTarget;
uniform vec3 cameraUp;
uniform vec3 lightPos;
uniform vec4 spheres[11]; // 중심과 반지름
uniform int spheresCount;
uniform sampler2D earth_texture;
uniform sampler2D moon_texture;

out vec4 FragColor;
in vec2 TexCoords;

const float MAX_DIST = 100.0;
const float MIN_DIST = 0.001;
const int MAX_STEPS = 100;
const float SHADOW_ACUITY = 16.0; // 그림자의 선명도
const float LIGHT_INTENSITY = 100000000.0;
const float PI = 3.14159265359;


vec3 calculateCameraDirection(vec3 cameraPos, vec3 cameraTarget) {
    return normalize(cameraTarget - cameraPos);
}

// 구면 좌표계에서 UV 좌표를 계산하는 함수
vec2 sphericalMapping(vec3 point, vec3 camDir, vec3 up, vec3 sphereCenter) {
    vec3 vp = normalize(point - sphereCenter); // 구 중심에서 점까지의 벡터
    vec3 northPole = normalize(-up);
    vec3 equator = normalize(cross(northPole, camDir));

    float u = 0.74 + atan(dot(cross(northPole, vp), equator), dot(vp, equator)) / (2.0 * PI);
    float v = 0.56 - asin(dot(vp, northPole)) / PI;
    vec2 uv = vec2(u, v);
    return uv;
}

vec2 getSphereUV(vec3 p, vec3 sphereCenter) {
    vec3 p_normalized = p - sphereCenter;
    float u = 0.5 + atan(p_normalized.z, p_normalized.x) / (2.0 * PI);
    float v = 0.5 - asin(p_normalized.y / length(p_normalized)) / PI;
    vec2 uv = vec2(u, v);
    return uv;
}


// Scene's Signed Distance Function
float sceneSDF(vec3 point) {
    float minDist = MAX_DIST;
    for (int i = 1; i < spheresCount; i++) {
        float dist = length(point - spheres[i].xyz) - spheres[i].w;
        minDist = min(dist, minDist);
    }
    return minDist;
}

// Ray Marching for scene geometry
float rayMarch(vec3 ro, vec3 rd, float start, float end) {
    float depth = start;
    for (int i = 0; i < MAX_STEPS; i++) {
        float dist = sceneSDF(ro + rd * depth);
        if (dist < MIN_DIST) break;
        depth += dist;
        if (depth >= end) return end;
    }
    return depth;
}

// Shadow ray marching
float softShadow(vec3 ro, vec3 rd, float start, float end) {
    float shadow = 1.0;
    float depth = start;
    for (int i = 0; i < MAX_STEPS; i++) {
        float dist = sceneSDF(ro + rd * depth);
        if (dist < MIN_DIST) return 0.0;
        // Use Penumbra factor to soften shadows at edges
        shadow = min(shadow, SHADOW_ACUITY * dist / depth);
        depth += dist;
        if (depth >= end) break;
    }
    return clamp(shadow, 0.0, 1.0);
}

vec3 getNormal(vec3 p) {
    float d = sceneSDF(p);
    vec2 e = vec2(0.01, 0);

    vec3 n = d - vec3(
        sceneSDF(p - e.xyy),
        sceneSDF(p - e.yxy),
        sceneSDF(p - e.yyx)
    );

    return normalize(n);
}

vec3 rayMarching(vec3 ro, vec3 rd, float start, float end) {
    float t = rayMarch(ro, rd, start, end);
    if (t < MAX_DIST) {
        vec3 p = ro + t * rd;
        vec3 normal = getNormal(p);
        vec3 lightDir = normalize(lightPos - p);
        float d = length(lightPos - p); // 광원과의 거리
        float attenuation = 1.0 / (1.0 + 0.09 * d + 0.032 * d * d); // 감쇠 계수

        float diff = 2.0*max(dot(normal, lightDir), 0.0) * attenuation;
        
        vec3 reflectDir = reflect(-lightDir, normal);
        float spec = pow(max(dot(reflectDir, -rd), 0.0), 16.0) * attenuation;
        
        float shadow = softShadow(p, normalize(lightPos-p), 0.01, length(lightPos-p));
        
        vec3 ambient = vec3(0.18);
        vec3 diffuse = vec3(0.15, 0.15, 0.15) * diff;
        vec3 specular = vec3(0.01) * spec;
        
        vec3 color = ambient + (diffuse + specular) * shadow * LIGHT_INTENSITY;
        
        
        
        if (length(p - spheres[3].xyz) - spheres[3].w < MIN_DIST) {
            vec3 camDir = calculateCameraDirection(cameraPos, cameraTarget);
            vec3 up = normalize(cameraUp);
            vec2 uv = sphericalMapping(p, camDir, up, spheres[3].xyz);
            color *= texture(earth_texture, uv).rgb; // 텍스처 색상으로 색상 조정
        }
        
        if (length(p - spheres[4].xyz) - spheres[4].w < MIN_DIST) {
            vec3 camDir = calculateCameraDirection(cameraPos, cameraTarget);
            vec3 up = normalize(cameraUp);
            vec2 uv = sphericalMapping(p, camDir, up, spheres[4].xyz);
            color *= texture(moon_texture, uv).rgb; // 텍스처 색상으로 색상 조정
        }

        return color;
    }
    return vec3(0.0, 0.0, 0.0); // Background color
}

void main() {
    // Calculate ray direction
    vec3 fwd = normalize(cameraTarget - cameraPos);
    vec3 right = normalize(cross(fwd, cameraUp));
    vec3 up = cross(right, fwd);
    vec2 uv = (gl_FragCoord.xy - 0.5 * vec2(800, 800)) / 800.0;
    vec3 rd = normalize(uv.x * right + uv.y * up + fwd);

    // Perform ray marching
    vec3 color = rayMarching(cameraPos, rd, 0.0, MAX_DIST);

    FragColor = vec4(color, 1.0);
}

"""
# np.set_printoptions(threshold=np.inf)

def load_texture(image_path):
    image = Image.open(image_path)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    img_data = np.array(list(image.getdata()), np.uint8)

    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    glGenerateMipmap(GL_TEXTURE_2D)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glBindTexture(GL_TEXTURE_2D, 0)

    return texture


def load_planet_positions():
    realX = np.load('data/high_dt/realX.npy')
    realY = np.load('data/high_dt/realY.npy')
    realZ = np.load('data/high_dt/realZ.npy')
    
    Numbers = len(realX[0])
    
    X = [[0 for j in range(Numbers)] for i in range(len(realX))]
    Y = [[0 for j in range(Numbers)] for i in range(len(realX))]
    Z = [[0 for j in range(Numbers)] for i in range(len(realX))]

    Q = [[] for i in range(len(realX))]
 
    frameRate = 1

    for i in range(len(realX)):
        X[i] = realX[i][18400::frameRate]
        Y[i] = realY[i][18400::frameRate]
        Z[i] = realZ[i][18400::frameRate]

        Q[i] = np.dstack((X[i], Y[i], Z[i]))[0]
        
    position = [[] for _ in range(len(Q))]
    
    for i in range(len(Q)):
        for j in range(len(Q[i])):
            position[i].append(glm.vec4(Q[i][j][0], Q[i][j][1], Q[i][j][2], Radi[i]))
    
    return position

current_index = 0  # 현재 행성 위치 인덱스

def update_sphere_positions(shader_program, positions):
    global current_index  # 전역 변수 사용

    spheresCount = len(positions)
    glUniform1i(glGetUniformLocation(shader_program, "spheresCount"), spheresCount)
    
    spheres = []
    
    for i in range(spheresCount):
        # positions[i]는 i번째 행성의 위치 정보 리스트임
        # current_frame % len(positions[i])는 위치 정보 리스트의 길이를 넘지 않도록 함
        current_position = positions[i][current_index % len(positions[i])]
        spheres.append(glm.vec4(current_position.x, current_position.y, current_position.z, current_position.w))
        
    for i, sphere in enumerate(spheres):
        glUniform4fv(glGetUniformLocation(shader_program, f"spheres[{i}]"), 1, glm.value_ptr(sphere))
        
    current_index += 1  # 다음 프레임을 위해 인덱스 증가
    
def update_light_position(shader_program, positions):
    global current_index
    
    light_position = glm.vec3(positions[0][current_index % len(positions[0])].x,
                              positions[0][current_index % len(positions[0])].y,
                              positions[0][current_index % len(positions[0])].z)
    
    glUniform3fv(glGetUniformLocation(shader_program, "lightPos"), 1, glm.value_ptr(light_position))

def update_camera_position(shader_program, positions):
    global current_index
    
    
    
    earth_position = glm.vec3(positions[3][current_index % len(positions[3])].x,
                              positions[3][current_index % len(positions[3])].y,
                              positions[3][current_index % len(positions[3])].z)

    moon_position = glm.vec3(positions[4][current_index % len(positions[4])].x,
                             positions[4][current_index % len(positions[4])].y,
                             positions[4][current_index % len(positions[4])].z)
    
    sun_position = glm.vec3(positions[0][current_index % len(positions[0])].x,
                            positions[0][current_index % len(positions[0])].y,
                            positions[0][current_index % len(positions[0])].z)
    
    distance = 50
   
    
    # observer_position = glm.vec3(moon_position.x - earth_position.x, moon_position.y - earth_position.y, moon_position.z - earth_position.z)
    # absolute = np.sqrt(observer_position.x**2+observer_position.y**2+observer_position.z**2)
    # observer_position1 = glm.vec3(observer_position.x/absolute, observer_position.y/absolute, observer_position.z/absolute)
    # observer_position2 = glm.vec3(observer_position1.x*Radi[3]*distance, observer_position1.y*Radi[3]*distance, observer_position1.z*Radi[3]*distance)
    # observer_position3 = glm.vec3(observer_position2.x + earth_position.x, observer_position2.y + earth_position.y, observer_position2.z + earth_position.z)
    
    # observer_position = moon_position - earth_position
    # observer_position = observer_position / np.sqrt(observer_position.x**2 + observer_position.y**2 + observer_position.z**2)
    # observer_position = observer_position*Radi[3]*distance
    # observer_position = observer_position + earth_position
    
    sun_view = earth_position - sun_position
    
    # camera_pos = sun_position + sun_view * 1.002
    camera_pos = earth_position + (moon_position - earth_position) * 0.95 # view from sun
    # camera_pos = observer_position # view from earth
    # camera_pos = glm.vec3(earth_position.x + 11, earth_position.y, earth_position.z)
    camera_target = moon_position
    
    
    glUniform3fv(glGetUniformLocation(shader_program, "cameraPos"), 1, glm.value_ptr(camera_pos))
    glUniform3fv(glGetUniformLocation(shader_program, "cameraTarget"), 1, glm.value_ptr(camera_target))


def timer(fps):
    glutPostRedisplay()  # 화면을 다시 그리도록 요청
    glutTimerFunc(int(1000 / fps), timer, fps)  # 다음 타이머 이벤트를 fps에 맞추어 등록


def main():
    global shader_program, VAO, VBO
    
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(800, 800)
    glutCreateWindow(b"Ray Marching Sphere")

    # Compile shaders and link them into a program
    shader_program = compileProgram(
        compileShader(vertex_shader, GL_VERTEX_SHADER),
        compileShader(fragment_shader, GL_FRAGMENT_SHADER),
    )

    glUseProgram(shader_program)


    # Set up vertex data (and buffer(s)) and configure vertex attributes
    vertices = np.array([
        -1.0, -1.0, 0.0,
         1.0, -1.0, 0.0,
        -1.0,  1.0, 0.0,
        -1.0,  1.0, 0.0,
         1.0, -1.0, 0.0,
         1.0,  1.0, 0.0,
    ], dtype=np.float32)

    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)

    glBindVertexArray(VAO)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * vertices.itemsize, None)
    glEnableVertexAttribArray(0)

    glBindBuffer(GL_ARRAY_BUFFER, 0)

    glBindVertexArray(0)

    # 셰이더 프로그램 설정 부분에 추가
    # 카메라 및 조명 위치 설정
    camera_pos = glm.vec3(0, 4, 4)
    camera_target = glm.vec3(0, 0, 0)
    camera_up = glm.normalize(glm.vec3(0, 0, 1))
    
    light_pos = glm.vec3(0, 0, 0)

    glUniform3fv(glGetUniformLocation(shader_program, "cameraPos"), 1, glm.value_ptr(camera_pos))
    glUniform3fv(glGetUniformLocation(shader_program, "cameraTarget"), 1, glm.value_ptr(camera_target))
    glUniform3fv(glGetUniformLocation(shader_program, "cameraUp"), 1, glm.value_ptr(camera_up))
    glUniform3fv(glGetUniformLocation(shader_program, "lightPos"), 1, glm.value_ptr(light_pos))

    # 구체 정보 설정
    # spheres = [glm.vec4(0, 3, 0, 1), glm.vec4(0.1, 1, 0, 0.1)]  # 예시: 두 구체
    # glUniform1i(glGetUniformLocation(shader_program, "spheresCount"), len(spheres))
    # for i, sphere in enumerate(spheres):
    #     glUniform4fv(glGetUniformLocation(shader_program, f"spheres[{i}]"), 1, glm.value_ptr(sphere))
        
    spherePositions = load_planet_positions()
    
    earth_texture = load_texture("textures/earthmap.bmp")
    moon_texture = load_texture("textures/moon.bmp")
    
    def render_scene():
        global current_index
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # 텍스처 활성화 및 바인딩
        glActiveTexture(GL_TEXTURE0)  # GL_TEXTURE0은 첫 번째 텍스처 유닛을 의미함
        glBindTexture(GL_TEXTURE_2D, earth_texture)
        
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, moon_texture)
        
        # 쉐이더 프로그램에 텍스처 유니폼 설정
        glUniform1i(glGetUniformLocation(shader_program, "earth_texture"), 0)  # 여기서 0은 GL_TEXTURE0 유닛에 해당함
        glUniform1i(glGetUniformLocation(shader_program, "moon_texture"), 1)
                
        # 매 프레임마다 행성의 위치를 업데이트
        update_camera_position(shader_program, spherePositions)
        update_light_position(shader_program, spherePositions)
        update_sphere_positions(shader_program, spherePositions)
        # print(spherePositions[1][current_index % len(spherePositions[1])])

        glBindVertexArray(VAO)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        glutSwapBuffers()

    fps = 30  # 초당 프레임 수
    glutDisplayFunc(render_scene)
    glutTimerFunc(0, timer, fps)  # fps에 맞추어 타이머 이벤트 등록
    glutMainLoop()

if __name__ == "__main__":
    main()
