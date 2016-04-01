#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include<rbsspf_tracker.cuh>

#include<rosinterface.h>
#include<sensor_msgs/LaserScan.h>
#include<sensor_msgs/PointCloud2.h>
#include<visualization_msgs/Marker.h>
#include<Eigen/Dense>

#include<QMainWindow>
#include<QGraphicsView>
#include<QGraphicsScene>
#include<QGraphicsLineItem>
#include<QGraphicsEllipseItem>
#include<QGraphicsPathItem>
#include<QPainterPath>
#include<QLayout>
#include<QMouseEvent>
#include<QKeyEvent>
#include<QWheelEvent>
#include<QPointF>
#include<QListWidget>
#include<QMap>
#include<QTime>
#include<QList>
#include<QPair>
#include<QSplitter>
#include<QFile>

namespace Ui {
class MainWindow;
}

#define NOHEIGHT

using namespace RobotSDK;

class LaserScanView : public QGraphicsView
{
    Q_OBJECT
public:
    LaserScanView(QWidget * parent=NULL);
public:
    void showLaserScan(std::vector<double> & scan);
    void clear();
    void showRect(double * cx, double * cy, bool pfflag);
//    void showRectPoint(std::vector<double> & scan, int beamnum, int * beamid);
//    void showParticle(ObjectState * particle, int pnum);
protected:
    void mousePressEvent(QMouseEvent *event);
    void wheelEvent(QWheelEvent *event);
    void keyPressEvent(QKeyEvent *event);
    void keyReleaseEvent(QKeyEvent *event);
protected:
    bool pressflag=0;
    bool ctrlflag=0;
    QPointF point1,point2;
    QGraphicsEllipseItem * pointitem1=NULL;
    QGraphicsEllipseItem * pointitem2=NULL;
    QGraphicsLineItem * line=NULL;
    int cornerid=0;
    QPointF corner[4];
public:
    QGraphicsLineItem * rectline[6]={NULL,NULL,NULL,NULL,NULL,NULL};
    double sx=1,sy=1;
    QGraphicsPathItem * pathitem=NULL;
    QPainterPath path;
    QGraphicsScene * scene=NULL;
public:
    QPainterPath egopath;
signals:
    void signalStart(double x, double y, double theta);
    void signalMeasure(QPointF * corners);
};

struct EGOMOTION
{
    double x,y,theta;
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
private:
    Ui::MainWindow *ui;
public:
    ROSSub<sensor_msgs::LaserScanConstPtr> * scansub;
    ROSTFSub * tfsub;
    QList< QPair<double,std::vector<double> > > scanlist;
    QList< QPair<double, EGOMOTION> > tflist;
    LaserScanView * view;
    QListWidget * list;
    bool inittrackflag=1;
    bool initmotionflag=0;
    int discontinuecount=0;
    std::vector<double> scan;
    sensor_msgs::PointCloud2ConstPtr vscan;
    std::string frameid;
    ros::Time timestamp;
    int seq;
    LaserScan loglaser;
public:
    Tracker tracker;
public:
    QFile file;
public slots:
    void slotReceive();
    void slotReceiveTF();
    void slotShowScan();
    void slotStart(double x, double y, double theta);
//    void slotShowRect(int id);
    void slotStopTracking();
protected:
    void showResult();
};

#endif // MAINWINDOW_H
