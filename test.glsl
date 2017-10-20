#version 450

struct MyStruct
{
  float x, y, z, w;
};

struct MyStruct2
{
  vec4 data;
};

struct MyStruct3
{
  vec4 data[56];
};

layout(std430, binding = 0) buffer TestThing1 { vec4 jasper[]; };
layout(std430, binding = 1) buffer TestThing2 { MyStruct otherJasper[]; };
layout(std430, binding = 2) buffer TestThing3 { MyStruct2 yetAnotherJasper[]; };
layout(std430, binding = 2) buffer TestThing4 { MyStruct3 yetAnotherAnotherJasper[]; };

void main()
{
// folded:
  jasper[0].x = 1;
  jasper[0].y = 2;
  jasper[0].z = 3;
  jasper[0].w = 4;
  
// untouched, but correct store
  jasper[1] = vec4(1, 2, 3, 4);
  
// todo: struct
  otherJasper[0].x = 1;
  otherJasper[0].y = 2;
  otherJasper[0].z = 3;
  otherJasper[0].w = 4;

// todo: struct
  MyStruct str;
  str.x = 1;
  str.y = 2;
  str.z = 3;
  str.w = 4;
  
  otherJasper[1] = str;
  
// folded:
  yetAnotherJasper[0].data.x = 0;
  yetAnotherJasper[0].data.y = 1;
  yetAnotherJasper[0].data.z = 2;
  yetAnotherJasper[0].data.w = 3;

// untouched, but correct
  MyStruct2 more;
  more.data = vec4(1, 2, 3, 4);
  yetAnotherJasper[1] = more;
  
  for(int idx = 0; idx < 20; idx++)
  {
  // folded:
    jasper[idx].x = idx;
    jasper[idx].y = idx;
    jasper[idx].z = idx;
    jasper[idx].w = idx;
    
    // todo; missing CSE on 'idx + 1'
    yetAnotherAnotherJasper[idx + 1].data[idx].x = 1;
    yetAnotherAnotherJasper[idx + 1].data[idx].y = 2;
    yetAnotherAnotherJasper[idx + 1].data[idx].z = 3;
    yetAnotherAnotherJasper[idx + 1].data[idx].w = 4;
    
    // folded:
    int k = idx + 1;
    yetAnotherAnotherJasper[k].data[idx].x = 1;
    yetAnotherAnotherJasper[k].data[idx].y = 2;
    yetAnotherAnotherJasper[k].data[idx].z = 3;
    yetAnotherAnotherJasper[k].data[idx].w = 4;
  }
  
}