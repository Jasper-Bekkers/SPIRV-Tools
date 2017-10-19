#version 450

struct MyStruct
{
  float x, y, z, w;
};

struct MyStruct2
{
  vec4 data;
};

layout(std430, binding = 0) buffer TestThing1 { vec4 jasper[]; };
layout(std430, binding = 1) buffer TestThing2 { MyStruct otherJasper[]; };
layout(std430, binding = 2) buffer TestThing3 { MyStruct2 yetAnotherJasper[]; };

void main()
{
  jasper[0].x = 1;
  jasper[0].y = 2;
  jasper[0].z = 3;
  jasper[0].w = 4;
  
  jasper[1] = vec4(1, 2, 3, 4);
  
  otherJasper[0].x = 1;
  otherJasper[0].y = 2;
  otherJasper[0].z = 3;
  otherJasper[0].w = 4;
  
  MyStruct str;
  str.x = 1;
  str.y = 2;
  str.z = 3;
  str.w = 4;
  
  otherJasper[1] = str;
  
  yetAnotherJasper[0].data.x = 0;
  yetAnotherJasper[0].data.y = 1;
  yetAnotherJasper[0].data.z = 2;
  yetAnotherJasper[0].data.w = 3;
  
  MyStruct2 more;
  more.data = vec4(1, 2, 3, 4);
  yetAnotherJasper[1] = more;
}