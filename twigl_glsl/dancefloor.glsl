// twigl -> https://twigl.app/?ol=true&ss=-OButejRYLzegJQRpS2f

//Pose Director :Lox9973

precision highp float;
uniform vec2 resolution;
uniform vec2 mouse;
uniform float time;
uniform sampler2D backbuffer;
out vec4 outColor;

#define repeat(x,a) mod(x,a) - a * 0.5
#define repeatIndex(x,a) floor(x/a)
#define stepFunc(x,a) floor(x / a) * a
#define modInt(x,a) int(mod(float(x),float(a)))
#define inside(a,b,c) (a >= b && a <= c)
#define saturate(a) clamp(a,0.0,1.0)


ivec2 font_data[84] = ivec2[](
    //0
    ivec2(0x00000000,0x00000000), //space

    //11~36
    ivec2(0x1e11110e,0x00000001), //a
    ivec2(0x0e11117f,0x00000000), //b
    ivec2(0x0a11110e,0x00000000), //c
    ivec2(0x7f11110e,0x00000000), //d
    ivec2(0x0815150e,0x00000000), //e
    ivec2(0x48483f08,0x00000000), //f
    ivec2(0x3e494930,0x00000000), //g
    ivec2(0x0708087f,0x00000000), //h
    ivec2(0x012f0900,0x00000000), //i
    ivec2(0x5e111102,0x00000000), //j
    ivec2(0x000b047f,0x00000000), //k
    ivec2(0x017f4100,0x00000000), //l
    ivec2(0x0807080f,0x00000007), //m
    ivec2(0x0708080f,0x00000000), //n
    ivec2(0x06090906,0x00000000), //o
    ivec2(0x1824243f,0x00000000), //p
    ivec2(0x3f242418,0x00000000), //q
    ivec2(0x0010081f,0x00000000), //r
    ivec2(0x0012150d,0x00000000), //s
    ivec2(0x11113e10,0x00000000), //t
    ivec2(0x0f01010e,0x00000000), //u
    ivec2(0x000e010e,0x00000000), //v
    ivec2(0x010e010e,0x0000000f), //w
    ivec2(0x0a040a11,0x00000011), //x
    ivec2(0x3e090930,0x00000000), //y
    ivec2(0x00191513,0x00000000), //z

    //36~63
    ivec2(0x7f88887f,0x00000000), //A
    ivec2(0x6e9191ff,0x00000000), //B
    ivec2(0x4281817e,0x00000000), //C
    ivec2(0x7e8181ff,0x00000000), //D
    ivec2(0x919191ff,0x00000000), //E
    ivec2(0x909090ff,0x00000000), //F
    ivec2(0x4685817e,0x00000000), //G
    ivec2(0xff1010ff,0x00000000), //H
    ivec2(0x0081ff81,0x00000000), //I
    ivec2(0x80fe8182,0x00000000), //J
    ivec2(0x413608ff,0x00000000), //K
    ivec2(0x010101ff,0x00000000), //L
    ivec2(0x601060ff,0x000000ff), //M
    ivec2(0x0c1060ff,0x000000ff), //N
    ivec2(0x7e81817e,0x00000000), //O
    ivec2(0x609090ff,0x00000000), //P
    ivec2(0x7f83817e,0x00000001), //Q
    ivec2(0x619698ff,0x00000000), //R
    ivec2(0x4e919162,0x00000000), //S
    ivec2(0x80ff8080,0x00000080), //T
    ivec2(0xfe0101fe,0x00000000), //U
    ivec2(0x0e010ef0,0x000000f0), //V
    ivec2(0x031c03fc,0x000000fc), //W
    ivec2(0x340834c3,0x000000c3), //X
    ivec2(0x300f30c0,0x000000c0), //Y
    ivec2(0xe1918d83,0x00000081), //Z

    //63~
    ivec2(0x00007d00,0x00000000), //!
    ivec2(0x60006000,0x00000000), //"
    ivec2(0x3f123f12,0x00000012), //#
    ivec2(0x52ff5224,0x0000000c), //$
    ivec2(0x33086661,0x00000043), //%
    ivec2(0x374d5926,0x00000001), //&
    ivec2(0x00006000,0x00000000), //'
    ivec2(0x0081423c,0x00000000), //(
    ivec2(0x003c4281,0x00000000), //)
    ivec2(0x00143814,0x00000000), //*
    ivec2(0x00103810,0x00000000), //+
    ivec2(0x00020100,0x00000000), //,
    ivec2(0x08080808,0x00000000), //-
    ivec2(0x00000100,0x00000000), //.
    ivec2(0x30080601,0x00000040), ///
    ivec2(0x00240000,0x00000000), //:
    ivec2(0x00240200,0x00000000), //;
    ivec2(0x41221408,0x00000000), //<
    ivec2(0x00141414,0x00000000), //=
    ivec2(0x08142241,0x00000000), //>
    ivec2(0xa999423c,0x0000007c), //@
    ivec2(0x008181ff,0x00000000), //[
    ivec2(0x06083040,0x00000001), //\
    ivec2(0x00000000,0x00000000), //] 何故か表示されない
    ivec2(0x00ff8181,0x00000000), //]
    ivec2(0x20402010,0x00000010), //^
    ivec2(0x01010101,0x00000000), //_
    ivec2(0x40408080,0x00000000), //`
    ivec2(0x41413608,0x00000000), //{
    ivec2(0x00ff0000,0x00000000), //|
    ivec2(0x08364141,0x00000000), //}
    ivec2(0x08101008,0x00000010) //~

);

#define FontWidth 8
#define FontHeight 8
#define LineMaxLength 40

vec3 font(vec2 uv,int id){
    vec2 uv1 = uv;
    uv = uv * 8.0;
    ivec2 texel = ivec2(uv);
    int bit_offset = texel.x * FontWidth + texel.y;

    int s,t;
    s = font_data[id].x;
    t = font_data[id].y;

    int tex = 0;
    
    if(bit_offset <= 31){
        s = s >> bit_offset;
        s = s & 0x00000001;
        tex = s;
    }
    else{
        t = t >> (bit_offset - 32);
        t = t & 0x00000001;
        tex = t;
    }

    tex = (abs(uv1.x - 0.5) < 0.5 && abs(uv1.y - 0.5) < 0.5) ? tex : 0;
    return vec3(tex); 
}

//IntegerHash by IQ
//https://www.shadertoy.com/view/XlXcW4
//https://www.shadertoy.com/view/4tXyWN

float PI = acos(-1.0);
float TAU = acos(-1.0) * 2.0;
vec3 CENTER = vec3(0.5,0.0,0.5);

#define C_HASH 2309480282U 
float TIME;
float DANCETIME;
float DANCETIME_F;
int SCENE = 0;

float SCENE_TIME = 0.0;
#define START_TIME 5.0
#define SCENE1 30.0
#define SCENE2 60.0
#define SCENE3 100.0

bool Playing;
vec2 cellID_global;

bool FINISH;

vec2 rot(vec2 uv, float a){
    return vec2(uv.x * cos(a) - uv.y * sin(a), uv.x * sin(a) + uv.y * cos(a));
}

vec3 worldtoLoacal(vec3 v,vec3 lx, vec3 ly,vec3 lz){
    return vec3(v.x * lx.x + v.y* lx.y + v.z * lx.z,
                 v.x * ly.x + v.y * ly.y + v.z * ly.z,
                 v.x * lz.x + v.y * lz.y + v.z * lz.z);
}

vec3 localToWorld(const vec3 v, const vec3 lx, const vec3 ly,
                   const vec3 lz)
{
    return vec3(v.x * lx.x + v.y * ly.x + v.z * lz.x,
                 v.x * lx.y + v.y * ly.y + v.z * lz.y,
                 v.x * lx.z + v.y * ly.z + v.z * lz.z);
}


void tangentSpaceBasis(vec3 normal,inout vec3 t,inout vec3 b){
    if (abs(normal.y) < 0.99)
    {
        t = cross(normal, vec3(0, 1, 0));
    }
    else
    {
        t = cross(normal, vec3(0, 0, -1));
    }
    t = normalize(t);
    b = cross(t, normal);
    b = normalize(b);
}

int selectInt(int a, int b,float f){
    return clamp(int(f * float(b+1 - a) + float(a)),a,b);
}

float hash11( float p )
{
    uint x = floatBitsToUint(p);
    x = C_HASH * ((x>>8U)^x);
    x = C_HASH * ((x>>8U)^x);
    x = C_HASH * ((x>>8U)^x);
    
    return float(x)*(1.0/float(0xffffffffU));
}

vec2 hash21( float p )
{
    uvec2 x = floatBitsToUint(vec2(p,230));
    x = C_HASH * ((x>>8U)^x.yx);
    x = C_HASH * ((x>>8U)^x.yx);
    x = C_HASH * ((x>>8U)^x.yx);
    
    return vec2(x)*(1.0/float(0xffffffffU));
}

vec3 hash31( float p )
{
    uvec3 x = floatBitsToUint(vec3(p,390,503));
    x = C_HASH * ((x>>8U)^x.yzx);
    x = C_HASH * ((x>>8U)^x.yzx);
    x = C_HASH * ((x>>8U)^x.yzx);
    
    return vec3(x)*(1.0/float(0xffffffffU));
}

float hash12( vec2 p )
{
    uvec2 x = floatBitsToUint(p);
    x = C_HASH * ((x>>8U)^x.yx);
    x = C_HASH * ((x>>8U)^x.yx);
    x = C_HASH * ((x>>8U)^x.yx);
    
    return float(x.x)*(1.0/float(0xffffffffU));
}

vec2 hash22( vec2 p )
{
    uvec2 x = floatBitsToUint(p);
    x = C_HASH * ((x>>8U)^x.yx);
    x = C_HASH * ((x>>8U)^x.yx);
    x = C_HASH * ((x>>8U)^x.yx);
    
    return vec2(x)*(1.0/float(0xffffffffU));
}

vec3 hash32( vec2 p )
{
    uvec3 x = floatBitsToUint(vec3(p,129));
    x = C_HASH * ((x>>8U)^x.yzx);
    x = C_HASH * ((x>>8U)^x.yzx);
    x = C_HASH * ((x>>8U)^x.yzx);
    
    return vec3(x)*(1.0/float(0xffffffffU));
}

float hash13(vec3 p){
    uvec3 x = floatBitsToUint(p);
    x = C_HASH * ((x>>8U)^x.yzx);
    x = C_HASH * ((x>>8U)^x.yzx);
    x = C_HASH * ((x>>8U)^x.yzx);
    
    return float(x.x)*(1.0/float(0xffffffffU));
}

vec2 hash23(vec3 p){
    uvec3 x = floatBitsToUint(p);
    x = C_HASH * ((x>>8U)^x.yzx);
    x = C_HASH * ((x>>8U)^x.yzx);
    x = C_HASH * ((x>>8U)^x.yzx);
    
    return vec2(x.xy)*(1.0/float(0xffffffffU));
}

vec3 hash33( vec3 p )
{
    uvec3 x = floatBitsToUint(p);
    x = C_HASH * ((x>>8U)^x.yzx);
    x = C_HASH * ((x>>8U)^x.yzx);
    x = C_HASH * ((x>>8U)^x.yzx);
    
    return vec3(x)*(1.0/float(0xffffffffU));
}

uint seed;
uint PCGHash()
{
    seed = seed * 747796405u + 2891336453u;
    uint state = seed;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rnd1()
{
    return float(PCGHash()) / float(0xFFFFFFFFU);    
}

vec2 rnd2(){
    return vec2(rnd1(),rnd1());
}

float powEase(float x,float n){
    return 1.0 - pow(1.0 - clamp(x,0.0,1.0),n);
}

float easeLerp(float z,float f){
    float fac = powEase(f,10.0);
    return mix(z,z+1.0,clamp(fac,0.0,1.0));
}

float easeLerp(float a,float b,float f){
    float fac = powEase(f,10.0);
    return mix(a,b,clamp(fac,0.0,1.0));
}

vec3 easeLerp(vec3 a,vec3 b,float f){
    float fac = powEase(f,10.0);
    return mix(a,b,clamp(fac,0.0,1.0));
}

float remap(float value,float low1,float high1,float low2,float high2){
    float fac = (value - low1) / (high1 - low1);
    return low2 + fac * (high2 - low2);
}

float Beat(float x,float n,float m){
    return pow(sin(clamp(x*n,0.0,2.0) * PI),m);
}

vec3 easeHash31(float idx,float x,float n){
    vec3 prePos = hash31(idx - 1.0);
    vec3 nowPos = hash31(idx);

    return mix(prePos,nowPos,vec3(powEase(x,n)));
}

float easeHash11(float idx,float x,float n){
    float prePos = hash11(idx - 1.0);
    float nowPos = hash11(idx);

    return mix(prePos,nowPos,powEase(x,n));
}


// Sample


//SDFs
//https://iquilezles.org/articles/distfunctions/

float sdSphere( vec3 p, float s )
{
  return length(p)-s;
}

float sdBox( vec3 p, vec3 b )
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float sdCapsule( vec3 p, vec3 a, vec3 b, float r )
{
  vec3 pa = p - a, ba = b - a;
  float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
  return length( pa - ba*h ) - r;
}

vec3 GetCameraDir(vec3 camera_ori,vec3 at_look, vec2 uv ,float Fov,float DOF,vec2 xi)
{
    vec3 camera_dir = normalize(at_look - camera_ori);
    float f = 1.0 / atan(Fov * TAU / 360.0f);

    vec3 tangent,binormal;
    tangentSpaceBasis(camera_dir,tangent,binormal);
    vec2 jitter = (xi*2.0 - 1.0) * DOF;
    camera_dir = normalize(camera_dir * f + (uv.x + jitter.x) * tangent + (uv.y + jitter.y) * binormal );
    return camera_dir;
}

#define NUM_POSE_POINTS 15
#define NUM_POSE_KEY 60
#define NUM_KEY_FRAME 4

vec3 POSE[NUM_POSE_KEY] = vec3[](
vec3 (0.0, 2.3, 0.0) ,
vec3 (0.0, 3.4, 0.0) ,
vec3 (0.0, 4.3, 0.0) ,
vec3 (-0.5, 3.7, 0.0) ,
vec3 (-0.5, 2.85, 0.0) ,
vec3 (-0.5, 2.0, 0.0) ,
vec3 (0.5, 3.7, 0.0) ,
vec3 (0.5, 2.85, 0.0) ,
vec3 (0.5, 2.0, 0.0) ,
vec3 (-0.3, 2.0, 0.0) ,
vec3 (-0.3, 1.0, 0.0) ,
vec3 (-0.3, 0.0, 0.0) ,
vec3 (0.3, 2.0, 0.0) ,
vec3 (0.3, 1.0, 0.0) ,
vec3 (0.3, 0.0, 0.0) ,
vec3 (0.0, 2.3, 0.0) ,
vec3 (0.0, 3.4, 0.0) ,
vec3 (0.0, 4.3, 0.0) ,
vec3 (-0.5, 3.7, 0.0) ,
vec3 (-1.34983, 3.68308, -0.0) ,
vec3 (-0.92898, 4.42158, -0.0) ,
vec3 (0.5, 3.7, 0.0) ,
vec3 (1.34034, 3.57221, -0.0) ,
vec3 (0.88197, 2.85639, -0.0) ,
vec3 (-0.3, 2.0, 0.0) ,
vec3 (-0.24161, 1.00171, 0.0) ,
vec3 (-0.18322, 0.00341, 0.0) ,
vec3 (0.3, 2.0, 0.0) ,
vec3 (0.3, 1.57738, 0.90631) ,
vec3 (0.3, 0.58119, 0.99346) ,
vec3 (0.0, 1.84258, 0.0) ,
vec3 (0.0, 2.94258, 0.0) ,
vec3 (0.26617, 3.80232, 0.00214) ,
vec3 (-0.5, 3.24258, 0.0) ,
vec3 (-0.80899, 4.03443, -0.0) ,
vec3 (-1.11798, 4.82628, -0.0) ,
vec3 (0.5, 3.24258, 0.0) ,
vec3 (1.13976, 2.68294, 0.0) ,
vec3 (0.63347, 2.00017, -0.0) ,
vec3 (-0.3, 1.54258, 0.0) ,
vec3 (-1.18439, 1.07583, -0.0) ,
vec3 (-1.12687, 0.07749, -0.0) ,
vec3 (0.3, 1.54258, 0.0) ,
vec3 (0.94622, 0.77943, -0.0) ,
vec3 (1.59244, 0.01628, -0.0) ,
vec3 (-0.0, 3.06759, 0.01906) ,
vec3 (-0.0, 1.96938, -0.04359) ,
vec3 (-0.05306, 1.09316, -0.24217) ,
vec3 (0.5, 1.66986, -0.06068) ,
vec3 (0.72, 0.85016, -0.10744) ,
vec3 (0.63371, 0.00592, -0.1556) ,
vec3 (-0.5, 1.66986, -0.06068) ,
vec3 (-0.72914, 0.85266, -0.1073) ,
vec3 (-0.68743, 0.00506, -0.15565) ,
vec3 (0.3, 3.36516, -0.01906) ,
vec3 (0.51377, 3.91959, -0.82337) ,
vec3 (0.72755, 4.47402, -1.62767) ,
vec3 (-0.3, 3.36516, -0.01906) ,
vec3 (-0.79755, 4.12046, 0.40751) ,
vec3 (-1.31857, 4.92315, 0.11731)
);

#define HEAD_OFFSET 2
#define UPPER_BODY_OFFSET 1
#define LOWER_BODY_OFFSET 0

#define UPPER_ARM_R_OFFSET 3
#define ELBOW_R_OFFSET 4
#define HAND_R_OFFSET 5
#define UPPER_LEG_R_OFFSET 6
#define KNEE_R_OFFSET 7
#define FOOT_R_OFFSET 8

#define UPPER_ARM_L_OFFSET 9
#define ELBOW_L_OFFSET 10
#define HAND_L_OFFSET 11
#define UPPER_LEG_L_OFFSET 12
#define KNEE_L_OFFSET 13
#define FOOT_L_OFFSET 14

#define PoseIndex(index, boneOffset) index * NUM_POSE_POINTS + boneOffset
vec3 posePoint[NUM_POSE_POINTS];

float HumanSDF(vec3 p){

    int key = int(DANCETIME) % NUM_KEY_FRAME;
    int nextKey = (key + 1) % NUM_KEY_FRAME; 

    key = (TIME < START_TIME + 5.0) ? 0 : key;
    nextKey = (TIME < START_TIME + 5.0) ? 0 : nextKey;

    key = (FINISH) ? 2 : key;
    nextKey = (FINISH) ? 2 : nextKey;


    int numKeyFrame = NUM_KEY_FRAME;
    for(int i = 0; i<NUM_POSE_POINTS; i++){
        vec3 prePoint = POSE[PoseIndex(key,i)];
        vec3 nextPoint = POSE[PoseIndex(nextKey,i)];
        if(!FINISH && (TIME > START_TIME + 5.0)){
            prePoint.xz *= (hash11(floor(DANCETIME)) < 0.5) ? -1.0 : 1.0;
            nextPoint.xz *= (hash11(floor(DANCETIME + 1.0)) < 0.5) ? -1.0 : 1.0;
        }
        vec3 resultPoint;
        resultPoint = easeLerp(prePoint,nextPoint,fract(DANCETIME));
        if(Playing){
            resultPoint.y *= remap(pow(sin(fract(DANCETIME * 0.5 - 0.5) * PI * 2.0),4.0),0.0,1.0,1.0,0.8);
            resultPoint.x *= remap(pow(sin(fract(DANCETIME * 0.5 - 0.5) * PI * 2.0),4.0),0.0,1.0,1.0,1.1);
        }
        
        posePoint[i] = resultPoint;
      
    }

    vec3 headPoint = posePoint[HEAD_OFFSET];  
    vec3 upperBody = posePoint[UPPER_BODY_OFFSET]; 
    vec3 lowerBody = posePoint[LOWER_BODY_OFFSET];

    vec3 upperArmR = posePoint[UPPER_ARM_R_OFFSET];
    vec3 elbowR = posePoint[ELBOW_R_OFFSET];
    vec3 handR = posePoint[HAND_R_OFFSET];

    vec3 upperArmL = posePoint[UPPER_ARM_L_OFFSET];
    vec3 elbowL = posePoint[ELBOW_L_OFFSET];
    vec3 handL = posePoint[HAND_L_OFFSET];

    vec3 upperLegR = posePoint[UPPER_LEG_R_OFFSET];
    vec3 kneeR = posePoint[KNEE_R_OFFSET];
    vec3 footR = posePoint[FOOT_R_OFFSET];

    vec3 upperLegL = posePoint[UPPER_LEG_L_OFFSET];
    vec3 kneeL = posePoint[KNEE_L_OFFSET];
    vec3 footL = posePoint[FOOT_L_OFFSET];

    float limb = 0.12;
    float body = 0.4;
    float head = 0.4;

    float d = 10000.0;

    float headD = sdSphere(p - headPoint,head);
    d = min(d,headD);

    float upperArmRD = sdCapsule(p,upperArmR,elbowR,limb);
    float lowerArmRD = sdCapsule(p,elbowR,handR,limb);
    d = min(d,upperArmRD);
    d = min(d,lowerArmRD);

    float upperArmLD = sdCapsule(p,upperArmL,elbowL,limb);
    float lowerArmLD = sdCapsule(p,elbowL,handL,limb);
    d = min(d,upperArmLD);
    d = min(d,lowerArmLD);

    float upperLegRD = sdCapsule(p,upperLegR,kneeR,limb);
    float lowerLegRD = sdCapsule(p,kneeR,footR,limb);
    d = min(d,upperLegRD);
    d = min(d,lowerLegRD);

    float upperLegLD = sdCapsule(p,upperLegL,kneeL,limb);
    float lowerLegLD = sdCapsule(p,kneeL,footL,limb);
    d = min(d,upperLegLD);
    d = min(d,lowerLegLD);

    float upperBodyD = sdCapsule(p,upperBody,lowerBody,body);
    d = min(d,upperBodyD);

    return d;
}

// https://www.shadertoy.com/view/Dsj3RW
struct GridCell {
  vec2 cell;
  float d;
};

vec2 CellPoint( vec2 p )
{
    return floor(p) + 0.5;
}

GridCell gridTraversal( vec3 ro, vec3 rd ) {
  GridCell r;

  r.cell = CellPoint( ro.xz + rd.xz * 1E-3); 

  vec2 src = -( ro.xz - r.cell ) / rd.xz;
  vec2 dst = abs( 0.5 / rd.xz );
  vec2 bv = src + dst;
  r.d = min( bv.x, bv.y);

  return r;
}

struct MapInfo{
    int index;
    vec3 localPos;
};

float Map(vec3 p,vec3 dir, vec2 cellPoint,inout MapInfo info){

    cellID_global = cellPoint;
    //Point
    vec2 cellID = cellPoint - vec2(0.5);    
    float s = min(abs(cellID.x) , abs(cellID.y));
    float c = abs(cellID.x) + abs(cellID.y);
    float m = max(abs(cellID.x) , abs(cellID.y));
    bool backdancerON = c == 3.0 && c < 5.0;
    bool mainStage = m < 5.0;

    vec3 localPos;
    vec3 worldP = p;
    worldP -= CENTER;

    if(SCENE >= 2){
        worldP.y -= mix(0.0,20.0,saturate((TIME - SCENE2) * 0.1));
    }

    float room = sdBox(worldP,vec3(5.0,5.0,5.0));
    float room_butinuki = sdBox(worldP + vec3(0.0,0.2,0.0),vec3(4.5,4.9,4.5));
    float dWall = max(room,-room_butinuki);

    int index = 0;
    p.xz -= cellPoint;

    vec3 floorP = p;
    float r = length(cellID);
    if(SCENE >= 1){
        // Stage Up
        floorP.y -= powEase(clamp(TIME - SCENE1 - 5.0,0.0,1.0),2.0) * (pow(sin(r + DANCETIME),2.0) * clamp(r,0.0,2.0) * 0.1 + step(length(cellID),0.0) * 1.0);
    }
    if(SCENE == 1){
        floorP.y -= -6.0 * pow(sin(clamp(SCENE_TIME*0.1,0.0,1.0) * PI),2.0) * float(backdancerON);
    }
    if(!mainStage){
        floorP.y += 7.0;
        floorP.y -= exp((r - 1.0) * 0.01)* 6.0 * easeHash11(floor(abs(cellID.x) + abs(cellID.y)),fract(DANCETIME),5.0);
    }

    float scale = 5.0 - hash12(cellID);
    // float d1 = sdSphere(floorP,0.3);
    float d1 = HumanSDF(floorP * scale) / scale;

    float dT = 10000.0;
    if(mainStage){
        if(SCENE >= 1 && backdancerON && (TIME - SCENE1) * 0.1 > 0.5) dT = d1;
        if(c == 0.0) dT = d1;
    }
    else{
        if(int(m) % 3 == 0) dT = d1;
    }
    d1 = dT;

    vec3 boxCenter = vec3(0.0,-4.0,0.0);
    vec3 boxP = floorP - boxCenter;
    float d2 = sdBox(boxP,vec3(0.5,4.0,0.5));
    float d = min(d1,d2);
    localPos = floorP;
    index = (d1 < d2) ? index : 1;
    index = (dWall < d) ? 2 : index;
    localPos = (dWall < d) ? worldP : localPos;
    d = min(d,dWall);

    info.index = index;
    info.localPos = localPos;
    return d;
}

vec3 GetNormal(vec3 p, vec2 cellPoint){
    vec2 eps = vec2(0.001,0.0);
    vec3 dammyDir = vec3(0.0,1.0,0.0);
    MapInfo dammy;

    return normalize(vec3(
        Map(p + eps.xyy,dammyDir,cellPoint,dammy) - Map(p - eps.xyy,dammyDir,cellPoint,dammy),
        Map(p + eps.yxy,dammyDir,cellPoint,dammy) - Map(p - eps.yxy,dammyDir,cellPoint,dammy),
        Map(p + eps.yyx,dammyDir,cellPoint,dammy) - Map(p - eps.yyx,dammyDir,cellPoint,dammy)
    ));
}

struct HitPoint{
    bool hit;
    float t;
    vec3 position;
    vec3 normal;
    vec3 basecolor;
    float roughtness;
    vec3 emission;
};

int DanceFont[6] = int[](30, 27, 40, 29, 31, 53);
int SessionsFont[8] = int[](45, 31, 45, 45, 35, 41, 40, 45);

#define MAX_STEPS 70
bool Raymarching(vec3 pos, vec3 dir, inout HitPoint hit){
    float rl = 1E-2;
    vec3 rp = pos + dir * rl;
    float gridlen = 0.0;
    GridCell grid;
    float dist;
    
    MapInfo info;
    for( int i = 0; i < MAX_STEPS; i ++ ) {
        grid = gridTraversal( rp, dir );
        dist = Map( rp, dir, grid.cell,info);
        rl = rl + min(dist,grid.d + 1E-3);
        rp = pos + dir * rl;
    }
    
    vec3 p = rp;

    if(dist < 0.001){
        hit.hit = true;
        hit.t = rl;
        hit.position = p;
        hit.normal = GetNormal(p,grid.cell);
        hit.roughtness = 0.0;

        vec3 emission = vec3(0.0);
        float roughness = 0.0;
        if(info.index == 0){
            hit.basecolor = vec3(0.5);
            hit.emission = vec3(0.2,1.0,0.2);
            hit.emission = (grid.cell.x == 0.5 && grid.cell.y == 0.5) ? hit.emission : normalize(hash32(grid.cell));
            hit.emission *= dot(-dir, hit.normal);
        }
        else if(info.index == 1){
            hit.basecolor = vec3(1.0);
            vec3 lightPos = mod(p * 3.0,1.0);
            vec2 lightID = floor(p.xz * 3.0) - 1.0;
            roughness = 0.3;

            //Floor
            if(dot(hit.normal, vec3(0.0,1.0,0.0)) > 0.9){
                lightPos = lightPos * 2.0 - 1.0;
                hit.basecolor = vec3(0.4);
                if(abs(lightPos.x) < 0.99 && abs(lightPos.z) < 0.99){
                    float factor = 0.0;
                    emission = (mod(abs(lightID.x) + abs(lightID.y),2.0) < 1.0) ? vec3(0.8,0.2,0.8) : vec3(0.2,0.5,1.0);
                    emission *= (1.5 - 2.0 *smoothstep(max(abs(lightPos.x), abs(lightPos.z)),0.0,0.5));

                    factor = 0.1;
                    factor += float(Playing) * float(abs(lightID.x) + abs(lightID.y) == floor(powEase(fract(DANCETIME),0.5) * 30.0));

                    if(SCENE >= 1){
                        vec2 crossLightID = lightID;
                        crossLightID = rot(crossLightID,floor(DANCETIME) * PI * 0.25);
                        factor += (abs(crossLightID.x) <= 1.0 || abs(crossLightID.y) <= 1.0) ? powEase(fract(1.0 - DANCETIME),2.0): 0.0;
                    }
                    if(SCENE >= 2){
                        factor += step(hash12(lightID + floor(TIME * 5.0)),0.3) * powEase(fract(1.0 - DANCETIME),2.0);
                    }

                    emission *= factor;

                }
                roughness = 0.0;
            }
            else{
                roughness = 0.2;
            }

            hit.emission = emission;
            hit.roughtness = roughness;
        }
        else if(info.index == 2){
            hit.basecolor = vec3(0.8);
            roughness = 0.2;
            if(abs(info.localPos.y - 2.0) < 1.0) {
                vec3 uvPos = info.localPos;
                vec2 uv = (abs(dot(hit.normal, vec3(0.0,0.0,1.0))) < 0.1) ? uvPos.zy *  vec2(-sign(dot(hit.normal, vec3(1.0,0.0,0.0))),1.0) : uvPos.xy * vec2(sign(dot(hit.normal, vec3(0.0,0.0,1.0))),1.0);
                vec2 uv1 = uv;
                uv.x += uv.y + TIME;
                uv.x *= 3.0;
                uv.x = repeat(uv.x,1.0);
                emission = (abs(uv.x) < 0.1) ? vec3(1.0) : vec3(0.0,0.0,0.0);
                emission = (abs(uv.y - 2.0) < 0.7) ? vec3(0.0) : emission;

                // Sign
                uv1.y -= 0.5;
                bool si = int(DANCETIME * 0.5) % 2 == 0;
                vec3 fontEmission = vec3(0.0);
                bool b1 = int(DANCETIME) % 2 == 0;
                if(si){
                    uv1.x *= 1.2;
                    uv1.x += 4.0;
                    int index = int(abs(floor(uv1.x + 1000.0))) % 8;
                    fontEmission += font(mod(uv1,1.0),SessionsFont[index]+ int(hash11(floor(TIME * 50.0)) * 50.0) * int(!b1)) * float(abs(uv1.y - 1.5) < 0.5) * float(0.0 < uv1.x && uv1.x < 8.0); 
                }
                else{
                    uv1.x += 4.5;
                    int index = int(abs(floor(uv1.x + 1000.0))) % 6;
                    fontEmission += font(mod(uv1,1.0),DanceFont[index] + int(hash11(floor(TIME * 50.0)) * 50.0) * int(!b1)) * float(abs(uv1.y - 1.5) < 0.5) * float(2.0 < uv1.x && uv1.x < 8.0); 
                }

                emission += fontEmission * float(b1 || TIME > SCENE1) * float(Playing);
                if(TIME > SCENE1)emission = mix(emission,1.0 - emission,float(inside(uv.y - powEase(DANCETIME_F,5.0) * 2.0 - 1.0,0.0,0.2)));
                emission = clamp(emission,0.0,1.0);
                emission *= float(TIME - SCENE2 - 10.0 < 0.0);
                roughness = 0.2;
            }
            hit.roughtness = roughness;
            hit.emission = emission;
        }

        return true;
    }
    else {
        return false;
    }
}

vec3 cosineSampling(vec2 uv){
    float theta = acos(1.0 - 2.0f * uv.x) * 0.5;
    float phi = 2.0 * PI * uv.y;
    return vec3(sin(theta) * cos(phi),cos(theta),sin(theta) * sin(phi));
}

vec3 LambertBRDF(vec3 wo,inout vec3 wi,vec3 basecol){
    wi = cosineSampling(rnd2()); 
    return basecol;
}

float GGX_D(vec3 wm,float ax,float ay){
	float term1 = wm.x * wm.x / (ax * ax) + wm.z * wm.z / (ay * ay) + wm.y * wm.y;
	float term2 = PI * ax * ay * term1 * term1;
	return 1.0f / term2;
}

float GGX_Lambda(vec3 w, float ax, float ay)
{
	float term = 1.0 + (ax * ax * w.x * w.x + ay * ay * w.z * w.z) / (w.y * w.y);
	return (-1.0 + sqrt(term)) * 0.5;
}

float GGX_G2(vec3 wo, vec3 wi, float ax, float ay)
{
	return 1.0 / (1.0 + GGX_Lambda(wo,ax,ay) + GGX_Lambda(wi,ax,ay));
}

vec3 Shlick_Fresnel(vec3 F0, vec3 v, vec3 n)
{
	return F0 + (1.0 - F0) * pow(1.0 - dot(v, n), 5.0);
}

vec3 WalterSampling(vec2 xi,float alpha){
    float phi = TAU * xi.x;
    float theta = atan(alpha * sqrt(xi.y) / (sqrt(1.0 - xi.y)));
    return vec3(sin(theta) * cos(phi),cos(theta),sin(theta) * sin(phi));
}

vec3 MicrofacetBRDF(vec3 wo,inout vec3 wi,vec3 F0,float alpha){
    float alpha_x = clamp(alpha * alpha,0.001,1.0);
    float alpha_y = clamp(alpha * alpha,0.001,1.0);
    vec3 normal = vec3(0,1,0);

    vec3 wm = WalterSampling(rnd2(),alpha_x);
	wi = reflect(-wo, wm);


	float ggx_D = GGX_D(wm, alpha_x, alpha_y);
	float ggx_G = GGX_G2(wo, wi, alpha_x, alpha_y);
	vec3 ggx_F = Shlick_Fresnel(F0, wo, wm);

    float pdf = 0.25 * ggx_D * wm.y / (dot(wm, wo)); // Walter
	return  ggx_G * ggx_F * ggx_D / ((4.0 * wo.y) * pdf);
}
vec3 Sky(vec3 dir){
    dir = normalize(dir);
    vec3 col = vec3(0.0); 
    // vec2 uv = vec2(acos(dir.x / length(dir.xz)),acos(dir.y) / PI);
    vec2 sphereUV = vec2(
        0.5 + atan(dir.z, dir.x) / TAU,
        // 0.5 - asin(dir.y)
        0.5 * (dir.y + 1.0)
    );

    bool si = int(DANCETIME * 0.5) % 2 == 0;
    vec3 fontEmission = vec3(0.0);
    bool b1 = int(DANCETIME) % 2 == 0;

    sphereUV += 0.1 * (hash22(hash21(floor(DANCETIME * 2.0)) + floor(sphereUV * 100.0)) - 0.5) * float(!b1) * float(!FINISH);

    vec2 uv1 = sphereUV * 100.0;
    uv1.y *= 0.5;
    uv1.x += DANCETIME + powEase(DANCETIME_F,2.0);
    // b1 = true;
    if(si){
        uv1.x *= 1.2;
        uv1.x += 4.0;
        int index = int(abs(floor(uv1.x + 1000.0))) % 10;
        fontEmission += font(mod(uv1,1.0),SessionsFont[index]+ int(hash11(floor(TIME * 50.0)) * 50.0) * int(!b1)); 
    }
    else{
        uv1.x += 4.5;
        int index = int(abs(floor(uv1.x + 1000.0))) % 6;
        fontEmission += font(mod(uv1,1.0),DanceFont[index] + int(hash11(floor(TIME * 50.0)) * 50.0) * int(!b1)); 
    }

    // return vec3(sphereUV,0.0);
    // col = vec3((sphereUV),0.0);
    
    col = (abs(sphereUV.y - 0.53) < 0.001) ? vec3(1.0) : col;
    col += (abs(sphereUV.y - 0.57) < 0.001) ? vec3(1.0) : col;
    col += fontEmission * float(abs(sphereUV.y - 0.55) < 0.01);
    col += fontEmission * float(hash12(floor(uv1) + hash21(floor(DANCETIME * 2.0))) < 0.1) * float(!FINISH);
    return col * vec3(1.0) * (1.0 -powEase(DANCETIME_F,1.0));
}

#define SAMPLE 2
vec3 SimplePathtrace(vec3 ro, vec3 rd){

    vec3 LTE = vec3(0.0);

    for(int s = 0; s < SAMPLE; s++){
        vec3 ray_ori = ro;
        vec3 ray_dir = rd;
        vec3 throughput = vec3(1.0);

        for(int i = 0; i < 2; i++){
            HitPoint hit;
            hit.hit = false;

            bool hitting = Raymarching(ray_ori,ray_dir,hit); 
            throughput += mix(vec3(0.2,0.1,0.2) * .0, vec3(0.0),exp(-0.02 * hit.t));
            if(!hitting){
                if(i == 0) LTE += Sky(ray_dir);
                break;
            }

            if(dot(hit.emission,hit.emission) > 0.0){
                LTE += hit.emission * throughput * 1.;
                if(dot(hit.emission, hit.emission) > 0.5){
                    break;
                }
            }

            vec3 normal = hit.normal;
            vec3 t,b;
            tangentSpaceBasis(normal,t,b);
            vec3 local_wo = worldtoLoacal(-ray_dir,t,normal,b);
            vec3 local_wi;

            vec3 bsdf;
            bsdf = MicrofacetBRDF(local_wo,local_wi,hit.basecolor,hit.roughtness);

            vec3 wi = localToWorld(local_wi,t,normal,b);

            throughput *= bsdf;

            ray_ori = hit.position + wi * 0.001;
            ray_dir = wi;
        }
    }

    LTE /= float(SAMPLE);


    return LTE;
}

#define Resolution resolution
void main()
{
    TIME = time;
    TIME = mod(TIME,120.0);
    // TIME += SCENE2;

    DANCETIME = TIME * 2.0;
    DANCETIME_F = fract(DANCETIME);

    SCENE_TIME = TIME;

    if(TIME > SCENE1){
        SCENE = 1;
        SCENE_TIME = TIME - SCENE1;
    }
    if(TIME > SCENE2){
        SCENE = 2;
        SCENE_TIME = TIME - SCENE2;
    }
    if(TIME > SCENE3){
        SCENE = 3;
        SCENE_TIME = TIME - SCENE3;
    }

    FINISH = false;
    FINISH = (TIME > SCENE3 + 10.0);
    if(SCENE == 3) {
        if(!FINISH) DANCETIME += pow((TIME - SCENE3),2.0) * mix(0.0,1.0, saturate((TIME - SCENE3) * 0.05));
    }
    seed = uint(TIME * 64.0) * uint(gl_FragCoord.x + gl_FragCoord.y * Resolution.x);
    vec2 uv = (2.0 * (gl_FragCoord.xy) - Resolution.xy)/Resolution.y;

    Playing = true;
    Playing = Playing && !(SCENE == 0 && SCENE_TIME < START_TIME);

    vec3 at_look = vec3(0.5,0.5,0.5);


    if(SCENE == 1){
        at_look = mix(at_look,vec3(0.5,1.5,0.5),saturate(SCENE_TIME - 5.0));
        if(SCENE_TIME > 15.0){
            at_look = floor(2.0 * (hash31(floor(TIME)) * 2.0 - 1.0)) + 0.5;
            at_look.y = clamp(at_look.y,0.5,1.0);
        }
    }

    if(SCENE >= 2){
        at_look = vec3(0.5,1.5,0.5);
    }

    float radius = 2.0 + easeHash11(floor(TIME),fract(TIME),4.0);
    vec3 direction = easeHash31(floor(TIME * 2.0),fract(TIME * 2.0),4.0);
    direction.x = direction.x * 1.0 - 0.5;
    direction.y = clamp(2.0 * direction.y - 1.0,-0.01,0.2);
    vec3 camera_ori = radius * normalize(direction) + at_look;

    float r1 = 4.0 - 2.0 * float(SCENE_TIME > 15.0);
    camera_ori = SCENE == 1 ? easeLerp(camera_ori,vec3(cos(TIME) * r1,1.0,sin(TIME) * r1),saturate(SCENE_TIME)) : camera_ori;

    if(SCENE == 2){
        camera_ori = vec3(cos(TIME) * r1,3.0,sin(TIME) * r1),saturate(SCENE_TIME);
        if(SCENE_TIME > 10.0){
            r1 = 10.0;
            camera_ori = (int(DANCETIME) % 3 == 1) ? vec3(-cos(TIME) * r1 * 0.5,10.0, 0.1 * sin(TIME) * r1) : camera_ori;
            camera_ori = (int(DANCETIME) % 3 == 2) ? vec3(-cos(TIME * 2.0) * r1 ,2.0, -0.1 * sin(TIME) * r1) : camera_ori;
        }
    }

    float Fov = 15.0; 
    Fov = (SCENE == 1) ? Fov+ 10.0 + Beat(saturate(SCENE_TIME),2.0,1.0) * 30.0 : Fov;
    Fov += 30.0 * easeHash11(floor(TIME),fract(TIME),4.0);
    if(SCENE == 2){
        Fov = mix(10.0, Fov, saturate(SCENE_TIME)); 
    }

    if(TIME < START_TIME + 5.0){
        camera_ori = mix(vec3(0.5,1.0,4.0),camera_ori,saturate((TIME - START_TIME - 5.0)));
        Fov = 30.0;
    }

    if(SCENE == 3){
        camera_ori = mix( vec3(0.5,2.0,2.0), vec3(0.5,2.0,10.0),saturate((10.0 - SCENE_TIME) * 0.1));
        Fov = mix( 10.0, 30.0,saturate((10.0 - SCENE_TIME) * 0.1));

        if(SCENE_TIME > 10.0){
            camera_ori = mix(vec3(0.5,2.0,2.0), vec3(0.5,2.0,10.0 +SCENE_TIME - 10.0),powEase(saturate(SCENE_TIME - 10.0),3.0));
            Fov = mix(10.0,30.0, powEase(saturate(SCENE_TIME - 10.0),3.0));
        }
        else{
            at_look += mix( vec3(0.0), hash31(TIME)*0.1 * saturate((SCENE_TIME - 5.0) * 0.1), pow(saturate((10.0 - SCENE_TIME) * 0.1),0.02)) ;
        } 
    }


    float DOF = 0.000;
    vec2 xi = hash21(time);

    vec3 camera_dir = GetCameraDir(camera_ori,at_look,uv,Fov,DOF,xi);

    vec3 color = SimplePathtrace(camera_ori,camera_dir);
    color = clamp(color,0.0,1.0);
    color = pow(color,vec3(1.2));
    color = mix(color,vec3(0.0),saturate((TIME - SCENE3 - 11.0) * 0.1));
    
    vec2 texuv = gl_FragCoord.xy /Resolution.xy;
    color = color * 0.7 + 0.3 * texture(backbuffer,texuv).xyz;
    outColor = vec4(color,1.0);
}
