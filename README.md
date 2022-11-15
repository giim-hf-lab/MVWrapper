# HikVision Robotics MV Wrapper

This is a simple C++20 wrapper of the HikVision Robotics SDKs (including the Vision SDK) compiled as a single VS recognisable C++ module interface file (suffix `.ixx`).

Noticed that, as per Vision SDK Version 4.2.1, there are mistakes exist in one of the header files (`Common/MVDShapeCpp.h`) which will break the compilation. Those classes are expected to be abstract classes, so all of their methods must be pure virtual but are not declared as such. A simple fix woule be to add `= 0` to the end of the method declarations where the compiler complains.
