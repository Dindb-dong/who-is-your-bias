// generated from ../bundle/gaussian-head-renderer.js (splat shaders) + real NDC depth
export const VS = `#version 300 es
precision highp float; precision highp int; precision highp usampler2D;
uniform sampler2D uLocal;   // N x 4: local xyz, opacity(sigmoid)
uniform sampler2D uRotS;    // N x 2 texels: (qw qx qy qz), (sx sy sz, -)
uniform usampler2D uBind;   // N x 1: face index
uniform sampler2D uFace;    // F x 2 texels: (cx cy cz s), (qw qx qy qz)
uniform sampler2D uColor;   // N x 1: rgb
uniform mat4 uView; uniform vec2 uFocal; uniform vec2 uViewport; uniform float uNear; uniform float uFar;
in vec2 aCorner; in uint aIndex;
out vec4 vColor; out vec2 vPos;
ivec2 at(uint i, int w) { return ivec2(int(i) % w, int(i) / w); }
vec3 qrot(vec4 q, vec3 v) { vec3 u = q.yzw; return v + 2.0 * cross(u, cross(u, v) + q.x * v); }
vec4 qmul(vec4 a, vec4 b) { return vec4(a.x*b.x - dot(a.yzw,b.yzw), a.x*b.yzw + b.x*a.yzw + cross(a.yzw,b.yzw)); }
mat3 qmat(vec4 q) {
  float w=q.x,x=q.y,y=q.z,z=q.w;
  return mat3(1.-2.*(y*y+z*z), 2.*(x*y+w*z), 2.*(x*z-w*y),
              2.*(x*y-w*z), 1.-2.*(x*x+z*z), 2.*(y*z+w*x),
              2.*(x*z+w*y), 2.*(y*z-w*x), 1.-2.*(x*x+y*y));
}
void main() {
  int W = textureSize(uLocal, 0).x; int WF = textureSize(uFace, 0).x / 2 * 2;
  vec4 L = texelFetch(uLocal, at(aIndex, W), 0);
  uint f = texelFetch(uBind, at(aIndex, W), 0).r;
  int fw = textureSize(uFace, 0).x / 2;
  ivec2 fp = ivec2(int(f) % fw * 2, int(f) / fw);
  vec4 F0 = texelFetch(uFace, fp, 0), F1 = texelFetch(uFace, fp + ivec2(1, 0), 0);
  int WR = textureSize(uRotS, 0).x / 2;
  ivec2 rp = ivec2(int(aIndex) % WR * 2, int(aIndex) / WR);
  vec4 q = normalize(texelFetch(uRotS, rp, 0)); vec3 sc = texelFetch(uRotS, rp + ivec2(1, 0), 0).xyz;
  vec4 qf = normalize(F1);
  vec3 world = qrot(qf, L.xyz) * F0.w + F0.xyz;
  mat3 R = qmat(qmul(qf, q)); vec3 S = sc * F0.w;
  mat3 M = R * mat3(S.x,0.,0., 0.,S.y,0., 0.,0.,S.z);
  mat3 Sigma = M * transpose(M);
  vec4 cam = uView * vec4(world, 1.0);
  float z = -cam.z;
  if (z < 0.01) { gl_Position = vec4(0., 0., 2., 1.); return; }
  mat3 Wv = mat3(uView);
  mat3 J = mat3(uFocal.x / z, 0., 0.,  0., uFocal.y / z, 0.,  uFocal.x * cam.x / (z*z), uFocal.y * cam.y / (z*z), 0.);
  mat3 T = J * Wv;
  mat3 cov = T * Sigma * transpose(T);
  float a = cov[0][0] + 0.3, b = cov[0][1], c = cov[1][1] + 0.3;
  float mid = 0.5 * (a + c), rad = length(vec2(0.5 * (a - c), b));
  float l1 = mid + rad, l2 = max(mid - rad, 0.1);
  if (l2 < 0.0) { gl_Position = vec4(0., 0., 2., 1.); return; }
  vec2 d1 = normalize(vec2(b, l1 - a)); if (abs(b) < 1e-9) d1 = a >= c ? vec2(1., 0.) : vec2(0., 1.);
  vec2 d2 = vec2(-d1.y, d1.x);
  vec2 centerPx = vec2(uFocal.x * cam.x / z, uFocal.y * cam.y / z);
  vec2 off = aCorner * 3.0;
  vec2 px = centerPx + off.x * sqrt(l1) * d1 + off.y * sqrt(l2) * d2;
  float zn = (uFar + uNear) / (uFar - uNear) - 2.0 * uFar * uNear / ((uFar - uNear) * z);
  gl_Position = vec4(px / (0.5 * uViewport), zn, 1.0);
  vPos = off;
  vColor = vec4(texelFetch(uColor, at(aIndex, W), 0).rgb, L.w);
}`;
export const FS = `#version 300 es
precision highp float;
in vec4 vColor; in vec2 vPos; out vec4 o;
void main() {
  float r2 = dot(vPos, vPos); if (r2 > 9.0) discard;
  float al = min(0.99, vColor.a * exp(-0.5 * r2)); if (al < 1.0/255.0) discard;
  o = vec4(vColor.rgb * al, al);
}`;

