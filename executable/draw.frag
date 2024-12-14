/* framework header */
#version 430
layout(location = 0) out vec4 fragColor;
layout(location = 0) uniform vec4 iResolution;
layout(location = 1) uniform int iFrame;

 
//--------------------------------------------------------------------------------------------------------------------------------
// Math
//--------------------------------------------------------------------------------------------------------------------------------
const float PI = acos(-1.0);

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

void tangentSpaceBasis(vec3 normal,inout vec3 t,inout vec3 b){
    t = (abs(normal.y) < 0.99) ? normalize(cross(normal, vec3(0, 1, 0))) : normalize(cross(normal, vec3(0, 0, -1)));
    b = normalize(cross(t, normal));
}

vec3 worldtoLoacal(vec3 v,vec3 lx, vec3 ly,vec3 lz){
    return vec3(dot(v, lx),dot(v,ly),dot(v,lz));
}

vec3 localToWorld(vec3 v, vec3 lx, vec3 ly, vec3 lz)
{
    return vec3(dot(v ,vec3(lx.x,ly.x,lz.x)),dot(v ,vec3(lx.y,ly.y,lz.y)),dot(v ,vec3(lx.z,ly.z,lz.z)));
}

#define C_HASH 2309480282U 
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


float hash13(vec3 p){
    uvec3 x = floatBitsToUint(p);
    x = C_HASH * ((x>>8U)^x.yzx);
    x = C_HASH * ((x>>8U)^x.yzx);
    x = C_HASH * ((x>>8U)^x.yzx);
    
    return float(x.x)*(1.0/float(0xffffffffU));
}

// https://www.shadertoy.com/view/4sfGzS
float noise(vec3 x)
{
    ivec3 i = ivec3(floor(x));
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
    ivec2 id = ivec2(0,1);
    return mix(mix(mix( hash13(i+id.xxx), 
                        hash13(i+id.yxx),f.x),
                   mix( hash13(i+id.xyx), 
                        hash13(i+id.yyx),f.x),f.y),
               mix(mix( hash13(i+id.xxy), 
                        hash13(i+id.yxy),f.x),
                   mix( hash13(i+id.xyy), 
                        hash13(i+id.yyy),f.x),f.y),f.z);
}

vec2 rot(vec2 uv, float a){
    return vec2(uv.x * cos(a) - uv.y * sin(a), uv.x * sin(a) + uv.y * cos(a));
}

//--------------------------------------------------------------------------------------------------------------------------------
// SDF 
//--------------------------------------------------------------------------------------------------------------------------------

// https://iquilezles.org/articles/distfunctions2d/
float sdCircle(vec2 p){
    return length(p) - 0.5;
}

float sdRing(vec2 p, vec2 n, float r, float th )
{
    p.x = abs(p.x);
   
    p = mat2x2(n.x,n.y,-n.y,n.x)*p;

    return max( abs(length(p)-r)-th*0.5,
                length(vec2(p.x,max(0.0,abs(r-p.y)-th*0.5)))*sign(p.x) );
}

float sdBox(vec2 p, vec2 b )
{
    vec2 d = abs(p)-b;
    return length(max(d,0.0)) + min(max(d.x,d.y),0.0);
}

float sdS(vec2 p){
    p -= vec2(0.0, 0.45);
    p.y *= 1.1;
    float theta = PI * 0.75;
    float d1 = sdRing(rot(p,PI*1.75), vec2(cos(theta), sin(theta)), 0.5, 0.3);
    float d2 = sdRing(rot(p + vec2(0.0,1.0),PI*0.75) , vec2(cos(theta), sin(theta)), 0.5, 0.3);

    float d = min(d1, d2);
    return d;
}

float sdE(vec2 p){
    p.y = abs(p.y);
    float d1 = sdBox(p + vec2(0.3,0.0), vec2(0.15, 1.0));
    float d2 = sdBox(p - vec2(0.2,0.0), vec2(0.4, 0.15));
    float d3 = sdBox(p - vec2(0.2,0.85), vec2(0.4, 0.15));

    float d = min(d1, d2);
    d = min(d, d3);
    return d;
}

float sdI(vec2 p){
    float d1 = sdBox(p, vec2(0.15, 1.0));
    return d1;
}

float sdN(vec2 p){
    p *= vec2(1.2,1.0);
    float d1 = sdBox(p + vec2(0.75,0.0), vec2(0.15, 1.0));
    float d2 = sdBox(p - vec2(0.75,0.0), vec2(0.15, 1.0));
    float d3 = sdBox(rot(p,PI*0.288), vec2(1.156, 0.15));
    float d = min(d1, d2);
    d = min(d, d3);
    return d;
}

float sdO(vec2 p){
    p *= vec2(1.2,1.1);
    float d1 = sdRing(p, vec2(cos(PI), sin(PI)), 1.0, 0.3);
    return d1;
}

float sdSESSIONS(vec2 p){
    float d = sdS(p + vec2(6.0,0.0));
    d = min(d,sdE(p + vec2(4.5,0.0)));
    d = min(d,sdS(p + vec2(3.0,0.0)));
    d = min(d,sdS(p + vec2(1.5,0.0)));
    d = min(d,sdI(p + vec2(0.3,0.0)));
    d = min(d,sdO(p - vec2(1.2,0.0)));
    d = min(d,sdN(p - vec2(3.2,0.0)));
    d = min(d,sdS(p - vec2(4.8,0.0)));
    return d;
}

float sdBox(vec3 p, vec3 s ) {
  vec3 d = abs( p ) - s;
  return length( max( d, 0.0 ) ) + min( 0.0, max( max( d.x, d.y ), d.z ) );
}

float sdCappedCylinder( vec3 p, float h, float r )
{
  vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(r,h);
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

//https://iquilezles.org/articles/distfunctions/
float extrudeSDF(vec3 p, float h, float sdf){
    float d = sdf;
    vec2 w = vec2( d, abs(p.z) - h );
    return min(max(w.x,w.y),0.0) + length(max(w,0.0));
}


//--------------------------------------------------------------------------------------------------------------------------------
// BSDF 
//--------------------------------------------------------------------------------------------------------------------------------
vec3 cosineSampling(vec2 uv){
    float theta = acos(1.0 - 2.0f * uv.x) * 0.5;
    float phi = 2.0 * PI * uv.y;
    return vec3(sin(theta) * cos(phi),cos(theta),sin(theta) * sin(phi));
}

vec3 LambertBRDF(vec3 wo,inout vec3 wi,vec3 basecol){
    wi = cosineSampling(rnd2()); 
    return basecol;
}

float GGX_D(vec3 wm,float alpha){
	float term1 = wm.x * wm.x / (alpha * alpha) + wm.z * wm.z / (alpha * alpha) + wm.y * wm.y;
	float term2 = PI * alpha * alpha * term1 * term1;
	return 1.0f / term2;
}

float GGX_Lambda(vec3 w, float alpha)
{
	float term = 1.0 + (alpha * alpha * w.x * w.x + alpha * alpha * w.z * w.z) / (w.y * w.y);
	return (-1.0 + sqrt(term)) * 0.5;
}

float GGX_G2(vec3 wo, vec3 wi, float alpha)
{
	return 1.0 / (1.0 + GGX_Lambda(wo,alpha) + GGX_Lambda(wi,alpha));
}

vec3 Shlick_Fresnel(vec3 F0, vec3 v, vec3 n)
{
	return F0 + (1.0 - F0) * pow(1.0 - dot(v, n), 5.0);
}

vec3 WalterSampling(vec2 xi,float alpha){
    float phi = 2.0 * PI * xi.x;
    float theta = atan(alpha * sqrt(xi.y) / (sqrt(1.0 - xi.y)));
    return vec3(sin(theta) * cos(phi),cos(theta),sin(theta) * sin(phi));
}

vec3 BSDF(vec3 wo,inout vec3 wi,vec3 basecol){

float pdf = 0.5;
vec3 wm;
if(pdf > rnd1()){
	wm = WalterSampling(rnd2(),0.01);
	wi = reflect(-wo, wm);
}
else{
	wi = cosineSampling(rnd2());
	wm = normalize(wo + wi);
}

if(wi.y < 0.0) return vec3(0.0);

float D = GGX_D(wm,0.01);
float G = GGX_G2(wo, wi, 0.01);
vec3 F = Shlick_Fresnel(vec3(0.04), wo, wm);

pdf = 0.25 * D * wm.y / dot(wo,wm) * pdf + (1.0 - pdf) * wi.y;

return (F * D * G / (4.0 * wo.y * wi.y) + basecol / PI) * wi.y / pdf ;

}

//--------------------------------------------------------------------------------------------------------------------------------
// Pathrace
//--------------------------------------------------------------------------------------------------------------------------------
vec3 BASECOLOR[6] = vec3[](vec3(1.0),vec3(0.2,0.5,0.2),vec3(0.93,0.8,0.63),vec3(0.34,0.24,0.05),vec3(1.0),vec3(0.3));
vec3 EMISSION[6] = vec3[](vec3(0.1),vec3(0.0),vec3(0.0),vec3(0.0),vec3(2.0,1.9,1.5) * 10.0, vec3(0.0));

int MatIndex;
float blickMap(vec3 p, inout int index){
    vec3 p1 = p * 0.2;
    float d1 = extrudeSDF(p1, 0.2, sdSESSIONS(p1.xy));
    float height = noise(p * 0.1) * 20.0;
    height = mix(height, -noise(p * 0.2) * 30.0, clamp(-p.z *0.01,0.0,1.0));
    height = mix(5.0, height, clamp(abs(p.z) * 0.05,0.0,1.0));
    float d2 = p.y + height;

    float d = 10000.0;
    index = (d2 < d) ? ((height < 7.0) ? (height < 2.0) ? 5 :1 : 2 ): 0;
    index = hash13(floor(p)) < 0.02 ? 4 : index;
    d = min(d, d2);

    index = (d1 < d) ? 0 : index;
    d = min(d,d1);

    return d;
}

vec3 gridCell;
float gridTraversal( vec3 ro, vec3 rd ) {
  gridCell = floor( ro + rd * 1E-3 ) + 0.5;
  vec3 src = -( ro - gridCell ) / rd;
  vec3 dst = abs( 0.5 / rd );
  vec3 bv = src + dst;
  return min( min( bv.x, bv.y), bv.z );
}

float map( vec3 p) {
  vec3 worldP = p;
  p -= gridCell;
  float kind = mod( dot( gridCell, vec3( 1.0 ) ), 3.0 );
  
  float d = 10000.0;
  
  int index;
  bool blick = blickMap(gridCell, MatIndex) < 0.02;
  bool haveDownBlick = blickMap(gridCell - vec3(0.0,1.0,0.0), index) < 0.02;
  if(blick){
    float dBlick = sdBox(p - vec3(0.0,0.02,0.0), vec3(0.5,0.46,0.5));
    dBlick = max(dBlick, -sdBox(p + vec3(0.0,0.2,0.0), vec3(0.45,0.6,0.45)));
    dBlick = min(dBlick, sdCappedCylinder(p + vec3(0.0,0.03,0.0), 0.47, 0.35));
    dBlick = max(dBlick, -sdCappedCylinder(p + vec3(0.0,0.3,0.0), 0.6, 0.32));
    d = dBlick;
  }
  if (haveDownBlick){
    float dCylinder = sdCappedCylinder(p + vec3(0.0,0.5,0.0), 0.2, 0.3);
    MatIndex = (dCylinder < d) ? index : MatIndex;
    d = min(d, dCylinder);
  }

  return d;
}

vec3 mapGradient( vec3 p) {
  vec2 d = vec2( 0, 1E-4 );
  return normalize( vec3(
    map( p + d.yxx) - map( p - d.yxx),
    map( p + d.xyx) - map( p - d.xyx),
    map( p + d.xxy) - map( p - d.xxy)
  ) );
}

struct HitInfo{
    vec3 pos;
    vec3 normal;
    vec3 basecolor;
    vec3 emission;
};

bool Raymarching(vec3 ro, vec3 rd, inout HitInfo hit){
  float rl = 1E-2;
  vec3 rp = ro + rd * rl;
  float dist;

  for( int i = 0; i < 500; i ++ ) {
    float dm = gridTraversal( rp, rd );
    dist = abs(map(rp));
    if(dist < 1E-2) break;
    rl = rl + min(dist,dm);
    rp = ro + rd * rl;
  }

  hit.pos = rp;
  hit.normal = mapGradient(rp);
  hit.basecolor = BASECOLOR[MatIndex];
  hit.emission = EMISSION[MatIndex];
  return dist < 1E-2;
}

vec3 Sky(vec3 rd){
    float phi = atan(rd.z, rd.x);
    float theta = atan(rd.y, length(rd.xz));

    vec2 uv = vec2(0.5 * phi / PI, theta / PI + 0.5);
    return mix(vec3(0.88,0.88,1.0),vec3(0.4,0.6,1.0),pow(uv.y,0.7)) * 3.0;
}

vec3 Pathtracing(vec3 ro, vec3 rd){

	vec3 ray_ori = ro;
	vec3 ray_dir = rd;
    vec3 LTE = vec3(0.0);
    vec3 throughput = vec3(1.0);
    
    for(int depth = 0; depth < 5; depth++){
        float rossianP = clamp(max(throughput.x, max(throughput.y,throughput.z)),0.0,1.0);
        if(rnd1() > rossianP){
            break;
        }
        throughput /= rossianP;


        HitInfo hit;
        if(!Raymarching(ray_ori,ray_dir,hit)){
            LTE += throughput * Sky(ray_dir) * 1.5;
			break;
		}

        if(dot(hit.emission,hit.emission) > 0.5){
            LTE += throughput * hit.emission; 
            break;
        }
        
        vec3 normal = hit.normal; 
        vec3 b,t;
        tangentSpaceBasis(normal,t,b);
    
        vec3 localwo = worldtoLoacal(-ray_dir,t,normal,b);
        vec3 localwi;
        vec3 bsdf = BSDF(localwo,localwi,hit.basecolor);

        vec3 wi = localToWorld(localwi,t,normal,b);

        throughput = throughput * bsdf; 
        
        ray_ori = hit.pos + wi * 0.005;
        ray_dir = wi;
    }
    
    return LTE;
}

float sdHexagon(vec2 p, float r )
{
    const vec3 k = vec3(-0.866025404,0.5,0.577350269);
    p = abs(p);
    p -= 2.0*min(dot(k.xy,p),0.0)*k.xy;
    p -= vec2(clamp(p.x, -k.z*r, k.z*r), r);
    return length(p)*sign(p.y);
}

vec3 GetThinLensCameraDir(vec3 camera_ori,vec3 at_look, vec2 uv , inout vec3 ray_ori, inout float cameraWeight)
{
    float L = length(at_look - camera_ori); 
    vec3 camera_dir = normalize(at_look - camera_ori);
    vec3 tangent,binormal;
    tangentSpaceBasis(camera_dir,tangent,binormal);

    float f = 7.0;
    float V = L * f / (L - f);
    float R = f / 3.0;

    float phi = rnd1() * 2 * PI;
    float r = R * sqrt(rnd1());
    vec2 samp = vec2(r * cos(phi),r * sin(phi));
    vec3 S = camera_ori + samp.x * tangent + samp.y * binormal;

    vec3 X = camera_ori - V * camera_dir - uv.x * tangent - uv.y * binormal;

    vec3 e = normalize(camera_ori - X);
    vec3 P = camera_ori + e * L / (dot(e,camera_dir));
    
    cameraWeight = sdHexagon(samp,R-0.5) < 0.0 ? 1.0 : 0.0;
    ray_ori = S;
    return normalize(P - S);
}

void main()
{
	// seed the RNG (again taken from Devour)
	seed = uint(iFrame*73856093)^(uint(gl_FragCoord.x)*19349663^uint(gl_FragCoord.y)*83492791)%38069;


	vec3 color = vec3(0.0);
    for(int i = 0; i < 4; i++){
		vec2 uv = (2. * (gl_FragCoord.xy + rnd2()) - iResolution.xy)/iResolution.y;
		vec3 cam_ori = vec3(-40.,40.,120.);
		float cameraWeight;
		vec3 ray_ori;
		vec3 ray_dir = GetThinLensCameraDir(cam_ori,vec3(-5.0,0.0,0.0),uv,ray_ori, cameraWeight);

		color += Pathtracing(ray_ori,ray_dir) * cameraWeight;
    }

	fragColor = vec4(color,4);
}
