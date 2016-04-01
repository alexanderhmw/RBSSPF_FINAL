#-------------------------------------------------
#
# Project created by QtCreator 2015-07-31T20:59:21
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

TARGET = RBPFAckermannTracker
TEMPLATE = app

SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h \
    rbsspf_motion.cuh \
    rbsspf_geometry.cuh \
    rbsspf_share.cuh \
    rbsspf_tracker.cuh

DISTFILES += \
    rbsspf_motion.cu \
    rbsspf_geometry.cu \
    rbsspf_share.cu \
    rbsspf_tracker.cu

FORMS    += mainwindow.ui

include($$(ROBOTSDKCUDA))

ROBOTSDKVER=4.0
INCLUDEPATH += $$(HOME)/SDK/RobotSDK_$${ROBOTSDKVER}/Kernel/include
CONFIG(debug, debug|release){
    LIBS += -L$$(HOME)/SDK/RobotSDK_$${ROBOTSDKVER}/Kernel/lib -lKernel_Debug
}
else{
    LIBS += -L$$(HOME)/SDK/RobotSDK_$${ROBOTSDKVER}/Kernel/lib -lKernel_Release
}

INCLUDEPATH += /usr/include/eigen3

ROS = $$(ROS_DISTRO)
isEmpty(ROS){
    error(Please install ROS first or run via terminal if you have ROS installed)
}
else{
    LIBS *= -L/opt/ros/$$ROS/lib -lroscpp
    LIBS *= -L/opt/ros/$$ROS/lib -lrosconsole
    LIBS *= -L/opt/ros/$$ROS/lib -lroscpp_serialization
    LIBS *= -L/opt/ros/$$ROS/lib -lrostime
    LIBS *= -L/opt/ros/$$ROS/lib -lxmlrpcpp
    LIBS *= -L/opt/ros/$$ROS/lib -lcpp_common
    LIBS *= -L/opt/ros/$$ROS/lib -lrosconsole_log4cxx
    LIBS *= -L/opt/ros/$$ROS/lib -lrosconsole_backend_interface
    LIBS *= -L/opt/ros/$$ROS/lib -ltf
    LIBS *= -L/opt/ros/$$ROS/lib -ltf2
    LIBS *= -L/opt/ros/$$ROS/lib -ltf2_ros
    LIBS *= -L/opt/ros/$$ROS/lib -lpcl_ros_tf
    LIBS *= -L/opt/ros/$$ROS/lib -ltf_conversions
    LIBS *= -L/opt/ros/$$ROS/lib -lactionlib
    LIBS *= -L/opt/ros/$$ROS/lib -lcv_bridge
    LIBS *= -L/opt/ros/$$ROS/lib -lrosbag
    LIBS *= -L/opt/ros/$$ROS/lib -lrosbag_storage
    LIBS *= -L/usr/lib/x86_64-linux-gnu -lboost_system
    INCLUDEPATH += /opt/ros/$$(ROS_DISTRO)/include
}
LIBS *= -L/usr/lib/x86_64-linux-gnu -lglut -lGLU
