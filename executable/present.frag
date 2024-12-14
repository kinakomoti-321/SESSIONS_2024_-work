/* framework header */
#version 430
layout(location = 0) out vec4 fragColor;
layout(location = 0) uniform vec4 iResolution;
layout(binding = 0) uniform sampler2D accumulatorTex;

vec3 ACESFilm(vec3 x)
{
	float a = 2.51;
	float b = 0.03;
	float c = 2.43;
	float d = 0.59;
	float e = 0.14;
	return vec3((x*(a*x+b))/(x*(c*x+d)+e));
}

void main()
{
	// readback the buffer
	vec4 tex = texelFetch(accumulatorTex,ivec2(gl_FragCoord.xy),0);

	// divide accumulated color by the sample count
	vec3 color = tex.rgb / tex.a;

	/* perform any post-processing you like here */

	// for example, some B&W with an S-curve for harsh contrast
	//color = smoothstep(0.,1.,color.ggg);
	color = ACESFilm(color);
	color = pow(color,vec3(1.2));
	// present for display
	fragColor = vec4(color,1);
}
