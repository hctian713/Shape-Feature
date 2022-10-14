// ShapeFeatures.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/core/core.hpp>   
#include <opencv2/highgui/highgui.hpp>   
#include <opencv2/opencv.hpp>
#include <cstring>
#include <vector>

using namespace std;  
using namespace cv;   

typedef struct{
	long int cir; //周长
	long int area; //面积
	double R;//圆心度
	Point massPt;//重心
	vector<Point> inPt; //内部点
	vector<Point> edgePt;//边缘点
}Shape; //图形类型

/////////////////////////////////////////////////辅助函数//////////////////////////////////////////////////////////////
//彩色图像灰度化
Mat Color2Gray(Mat const& SrcImg) {
	int rows = SrcImg.rows;
	int cols = SrcImg.cols;
	Mat mGray;
	mGray.create(rows, cols, CV_8UC1);
	int nMaxValue;
	int bValue, gValue, rValue;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			bValue = SrcImg.at<Vec3b>(i, j)[0];
			gValue = SrcImg.at<Vec3b>(i, j)[1];
			rValue = SrcImg.at<Vec3b>(i, j)[2];
			nMaxValue = bValue;
			if (nMaxValue < gValue)
			{
				nMaxValue = gValue;
			}
			if (nMaxValue < rValue)
			{
				nMaxValue = rValue;
			}
			mGray.at<uchar>(i, j) = saturate_cast<uchar>(nMaxValue);
		}
	}
	return mGray;
}
//图像二值化(状态法)
Mat StateMethord(const Mat& srcImg)
{
	Mat gray_img, dst;
	int cn = srcImg.channels();//通道数
	if (cn == 3)
		gray_img = Color2Gray(srcImg);
	else
		gray_img = srcImg.clone();
	int height = gray_img.rows;//行数
	int width = gray_img.cols;//列数
	dst.create(gray_img.size(), gray_img.type());
	//峰谷法获取阙值-
	int nThreshold = 0, nNewThreshold = 0;
	int hist[256]; memset(hist, 0, sizeof(hist));//定义灰度统计数组
	int lS1, lS2;//类1,2像素数
	double lP1, lP2;//类1,2质量矩
	double meanvalue1, meanvalue2;//类1,2灰度均值
	int IterationTimes;//迭代次数
	int graymax = 255, graymin = 0;//灰度最大最小值
	int i, j, k;
	for (i = 0; i < height; i++)//统计出各个灰度值的像素数
		for (j = 0; j < width; j++) {
			int gray = gray_img.at<uchar>(i, j);
			hist[gray]++;
		}
	//给阙值赋迭代初值
	nNewThreshold = int((graymax + graymin) / 2);
	//迭代求最佳阙值
	for (IterationTimes = 0; nThreshold != nNewThreshold && IterationTimes < 100; IterationTimes++)
	{
		nThreshold = nNewThreshold;
		lP1 = lP2 = 0.0;
		lS1 = lS2 = 0;
		//求类1的质量矩和像素总数
		for (k = graymin; k <= nThreshold; k++) {
			lP1 += double(k) * hist[k];
			lS1 += hist[k];
		}
		//计算类1均值
		meanvalue1 = lP1 / lS1;
		//求类2的质量矩和像素总数
		for (k = nThreshold + 1; k <= graymax; k++) {
			lP2 += double(k) * hist[k];
			lS2 += hist[k];
		}
		//计算类2均值
		meanvalue2 = lP2 / lS2;
		//获取新的阙值
		nNewThreshold = int((meanvalue1 + meanvalue2) / 2);
	}
	//二值化图像
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++) {
			if (gray_img.at<uchar>(i, j) < nNewThreshold)
				dst.at<uchar>(i, j) = 0;
			else dst.at<uchar>(i, j) = 255;
		}
	return dst;
}
//图像二值化(判断分析法)
Mat OSTU(const Mat& srcImg)
{
	//创建灰度图
	Mat gray_img, dst;
	int cn = srcImg.channels();//通道数
	if (cn == 3)
		gray_img= Color2Gray(srcImg);
	else
		gray_img = srcImg.clone();
	int height = gray_img.rows;//行数
	int width = gray_img.cols;//列数
	dst.create(gray_img.size(), gray_img.type());
	//判断分析法（Ostu）获取阙值
	int nThreshold = 0;
	int hist[256]; memset(hist, 0, sizeof(hist));//定义灰度统计数组
	int lS, lS1, lS2;//总像素数，类1,2像素数
	double lP, lP1;//总质量矩，类1质量矩
	double meanvalue1, meanvalue2;//类1,2灰度均值
	int IterationTimes;//迭代次数
	double Dis1, Dis2, CinDis, CoutDis, max;//方差，类1，类2，组内，组间
	int graymax = 255, graymin = 0;//灰度最大最小值
	int i, j, k;
	//统计出各个灰度值的像素数
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++) {
			int gray = gray_img.at<uchar>(i, j);
			hist[gray]++;
		}
	lP = 0.0; lS = 0;
	if (graymin == 0) graymin++;
	//计算总的质量矩lP和像素总数lS
	for (k = graymin; k <= graymax; k++) {
		lP += double(k) * double(hist[k]);
		lS += hist[k];
	}
	max = 0.0;
	lS1 = lS2 = 0;
	lP1 = 0.0;
	Dis1 = Dis2 = CinDis = CoutDis = 0.0;
	double ratio = 0.0;
	//求阙值
	for (k = graymin; k < graymax; k++) {
		lS1 += hist[k];
		if (!lS1) { continue; }
		lS2 = lS - lS1;
		if (lS2 == 0) { break; }
		lP1 += double(k) * double(hist[k]);
		meanvalue1 = lP1 / lS1;
		for (int n = graymin; n <= k; n++)
			Dis1 += double((n - meanvalue1) * (n - meanvalue1) * hist[n]);
		meanvalue2 = (lP - lP1) / lS2;
		for (int m = k + 1; m < graymax; m++)
			Dis2 += double((m - meanvalue2) * (m - meanvalue2) * hist[m]);
		CinDis = Dis1 + Dis2;
		CoutDis = double(lS1) * double(lS2) * (meanvalue1 - meanvalue2) * (meanvalue1 - meanvalue2);
		if (CinDis != 0) ratio = CoutDis / CinDis;
		if (ratio > max) {
			max = ratio; nThreshold = k;
		}
	}
	//二值化图像
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++) {
			if (gray_img.at<uchar>(i, j) < nThreshold)
				dst.at<uchar>(i, j) = 0;
			else dst.at<uchar>(i, j) = 255;
		}
	return dst;
}
//滑动条回调函数
void on_ThreshChange(int, void*)//回调函数
{

}
//找到图形中面积的最值
void findM(vector<Shape>& shapes, long int& areaMax, int& areaMin) {
	int max = 0;
	int min = 99999999;
	for (vector<Shape>::iterator iter = shapes.begin() + 1; iter != shapes.end(); iter++) {
		if (min > iter->area) min = iter->area;
		if (max < iter->area) max = iter->area;
	}
	areaMax = max;
	areaMin = min;
}
//读取二进制文件
Mat readbyte(const char* path)
{	
	FILE* fp;
	if (fopen_s(&fp, path, "rb") != 0)
	{
		printf("cannot open file for read\n");
		waitKey();//opencv中常用waitKey(n),n<=0表示一直
		exit(0);
	}
	int cols = 2076;
	int rows = 2816;
	int bands = 3;// 波段数
	int pixels = cols * rows * bands;
	unsigned char* data = new uchar[pixels * sizeof(uchar)];//uchar等效于char，此处可以借鉴这样的预分配内存方法
	fread(data, sizeof(uchar), pixels, fp);//fread函数的用法
	fclose(fp);
	Mat M(rows, cols, CV_8UC3, Scalar(0, 0, 0));
	unsigned char* ptr = M.data;//data是存数据的部分指针
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			for (int k = 0; k < 3; k++)
			{
				ptr[(i * cols + j) * 3 + k] = data[(i * cols + j) * 3 + k];//BIP
			}
		}
	}
	return M;
}

/////////////////////////////////////////////////核心函数////////////////////////////////////////////////////////////

//①连通成分标记
        //种子填充法
int Cc_FeedFill(const Mat& binImg, Mat& label)
{
	//如果为空或并非单通道图像，则返回-1
	if (binImg.empty() || binImg.type() != CV_8UC1)return -1;
	int labels = 0;
	int cols = binImg.cols;
	int rows = binImg.rows;
	label = Mat::zeros(rows, cols, CV_8UC1);
	vector<int>vt;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (binImg.at<uchar>(i * cols + j) == 255 && label.at<uchar>(i * cols + j) == 0) {
				vt.push_back(i * cols + j);
				labels++;
				while (vt.size() != 0) {
					int px = *(vt.end() - 1);
					vt.pop_back();
					label.at<uchar>(px) = labels;
					//查询上下右左是否为连通分量
					if (px > cols && binImg.at<uchar>(px - cols) == 255 && label.at<uchar>(px - cols) == 0)
						vt.push_back(px - cols);
					if (px < (rows - 1) * cols && binImg.at<uchar>(px + cols) == 255 && label.at<uchar>(px + cols) == 0)
						vt.push_back(px + cols);
					if ((px + 1) % cols != 0 && binImg.at<uchar>(px + 1) == 255 && label.at<uchar>(px + 1) == 0)
						vt.push_back(px + 1);
					if ((px) % cols != 0 && binImg.at<uchar>(px - 1) == 255 && label.at<uchar>(px - 1) == 0)
						vt.push_back(px - 1);
				}
			}
		}
	}
	return labels;
}
        //两次扫描法
int Cc_TwoPass(const Mat& binImg, Mat& labelImg) 
{ 
	//判断是否为二值图像
    if (binImg.empty() || binImg.type() != CV_8UC1)  return-1;
	//两次扫描法，binImg为输入图像，labelImg为连通成分标记后的图像，返回存储连通区域的数量

	//第一次遍历
	int label = 0;
	vector<int> labelSet(1);//labelSet存储标记之间的树形结构关系，labelSet[i]表示标记为i的双亲结点
						  //labelSet[i] = 0时，i为根节点，0号单元弃用
	int rows = binImg.rows;
	int cols = binImg.cols;

	int** labelMatrix = new int*[rows];
	for (int k = 0; k < rows; k++) labelMatrix[k] = new int[cols]();//临时标记矩阵,初始化为0

	labelImg = Mat::zeros(binImg.size(), binImg.type());//背景标记为0

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (binImg.at<uchar>(i, j) == 255) {//前景区域，需标记
				int left = j - 1 < 0 ? 0 : labelMatrix[i][j - 1];//左像素值，外围标0
				int top = i - 1 < 0 ? 0 : labelMatrix[i - 1][j];//上像素值，外围标0
				if (left == 0 && top == 0) {//新增标记
					labelMatrix[i][j] = label++;
					labelSet.push_back(0);
				}
				else {
					if (left != 0 && top != 0) {//将left和top归入同一树中
						int pmin = min(left, top), pmax = max(left, top);
						labelMatrix[i][j] = pmin;//取较小值
						while (labelSet[pmin] != 0)
							pmin = labelSet[pmin];
						while (labelSet[pmax] != 0)
							pmax = labelSet[pmax];
						if (pmin != pmax) //若二者根节点不同，设较小值为较大值的双亲结点
							labelSet[pmax] = pmin;
					}
					else
						labelMatrix[i][j] = max(left, top);//若一个为0，则选不为0的值为标记
				}
			}
		}
	}
	//第二次遍历
	map<int, int> label2idx;//存储根节点与序号的对应关系
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (binImg.at<uchar>(i, j) == 255) {
				int p = labelMatrix[i][j];
				while (labelSet[p] != 0) {
					p = labelSet[p];//找到对应根节点
				}
				map<int, int>::iterator iter;
				iter = label2idx.find(p);
				if (iter == label2idx.end()) {
					label2idx[p] = label2idx.size();//建立根节点与序号的对应关系
				}
				labelImg.at<uchar>(i, j) = label2idx[p];//最终标记为根节点对应的序号
			}
		}
	}
	int shapeSum = label2idx.size()-1;
	delete[] labelMatrix;
	return shapeSum;
}
//②形状特征提取
void Feature(const Mat& labelImg, int& shapeSum, vector<Shape>& shapes) { //输入的是一个mat，且各图像不同编号
	Shape shape0;
	shapes.push_back(shape0);//0号位置舍弃
	for (int k = 1; k <= shapeSum; k++) {
		Shape t_shape;
		shapes.push_back(t_shape);
		int width = labelImg.cols;
		int height = labelImg.rows;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if(labelImg.at<uchar>(i, j) == k) {//该点属于图形
					if (i == 0 || j == 0 || i == height - 1 || j == width - 1)
					{//位于图片边缘
					    Point E1(i, j);//边缘点
					    shapes[k].edgePt.push_back(E1); 
					}
					else if (labelImg.at<uchar>(i, j + 1) == 0 || labelImg.at<uchar>(i, j - 1) == 0 || labelImg.at<uchar>(i + 1, j) == 0 || labelImg.at<uchar>(i - 1, j) == 0)
					{//位于图形边缘
					    Point E2(i, j);//边缘点
						shapes[k].edgePt.push_back(E2);
					}
					else
					{//位于图片内部
						Point I(i, j);
					    shapes[k].inPt.push_back(I);
					}
				}
			}
		}		
		//面积
		shapes[k].area = (int)shapes[k].inPt.size() + (int)shapes[k].edgePt.size() ;
		//周长
		float cir = 0;//周长
		Mat Gri = Mat::zeros(labelImg.size(), labelImg.type());//只有边缘的mat
		for (vector<Point>::iterator iter = shapes[k].edgePt.begin(); iter != shapes[k].edgePt.end(); iter++) {
			Gri.at<uchar>(iter->x, iter->y) = 1;
		}
		double grid[9] = { sqrt(2),1,sqrt(2),1,0,1,sqrt(2),1,sqrt(2) };
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (i == 0 || j == 0 || i == height - 1 || j == width - 1) {
					if (Gri.at<uchar>(i, j) == 1) cir = cir + 2;
				}
				else {
					if (Gri.at<uchar>(i, j) == 1) {
						double griSum = 0;//模板卷积的和
						int idx = 0;//模板索引
						for (int n = -1; n < 2; n++) {
							for (int m = -1; m < 2; m++) {
								griSum += Gri.at<uchar>(i + n, j + m) * grid[idx];
								idx++;
							}
						}
						while (griSum > 3)griSum -= sqrt(2);//判断不相邻的边缘点是否被当作相邻的边缘点
						cir = cir + griSum;
					}
				}
			}
		}
		shapes[k].cir = (long int)cir / 2;//相邻边缘点的长度被算了两遍
		//圆形度
		shapes[k].R = 4 * CV_PI * shapes[k].area / (shapes[k].cir * shapes[k].cir);
		//重心
		long  mx = 0, my = 0;
		for (int i = 0; i < shapes[k].inPt.size(); i++) {
			mx += shapes[k].inPt[i].x;
			my += shapes[k].inPt[i].y;
		}
		for (int i = 0; i < shapes[k].edgePt.size(); i++) {
			mx += shapes[k].edgePt[i].x;
			my += shapes[k].edgePt[i].y;
		}
		shapes[k].massPt.x = (int)mx / shapes[k].area;
		shapes[k].massPt.y = (int)my / shapes[k].area;	
	}
}
//③依条件筛选
void Choose(const Mat& labelImg, vector<Shape>& shapes,vector<Shape>& Cshapes ,long int areaMin)
{
	Shape Shape0;
	Cshapes.push_back(Shape0);
	for (vector<Shape>::iterator iter = shapes.begin()+1; iter != shapes.end(); iter++) {
		if (iter->area >= areaMin)
			Cshapes.push_back(*iter);
	}
}
//④可视化展示
Mat Display(const vector<Shape>& shapes, const Mat& labelImg) {//形状特征显示
	//shapes为图形集合
	Scalar* colors = new Scalar[shapes.size() + 1];//颜色数组
	srand((unsigned)time(NULL));
	colors[0] = Scalar(0, 0, 0);
	for (size_t i = 1; i <= shapes.size(); i++) {//图形颜色随机
		uchar b = rand() % 256, g = rand() % 256, r = rand() % 256;
		colors[i] = Scalar(b, g, r);
	}
	//背景白色
	Mat display=Mat::zeros(labelImg.rows, labelImg.cols, CV_8UC3);
	for (int i = 0; i < labelImg.rows; i++)
		for (int j = 0; j < labelImg.cols; j++)
			for (int k = 0; k < 3; k++)
				display.at<Vec3b>(i, j)[k] = 255;
	//填充图形
	for (int i = 1; i < shapes.size(); i++) {
		for (int j = 0; j < shapes[i].inPt.size(); j++) {
			Point pt = shapes[i].inPt[j]; 
			for (int k = 0; k < 3; k++) 
				display.at<Vec3b>(pt.x,pt.y)[k] = colors[i][k];
		}
		for (int j = 0; j < shapes[i].edgePt.size(); j++) {
			Point pt = shapes[i].edgePt[j];
			for (int k = 0; k < 3; k++)
				display.at<Vec3b>(pt.x, pt.y)[k] = Scalar(0,0,255)[k];
		}
	}
	delete[] colors;

	//写入参数
	for (int k = 1; k < shapes.size(); k++) {

		//设置绘制文本的相关参数
		string area = "S:" + to_string(shapes[k].area);
		string perimeter = "C:" + to_string(shapes[k].cir);
		string cd = "R:" + to_string(shapes[k].R);
		string masspot = "(" + to_string(shapes[k].massPt.x) + "," + to_string(shapes[k].massPt.y) + ")";
		int font_face = FONT_HERSHEY_COMPLEX;
		double font_scale = 0.35;
		int thickness = 0.7;
		int baseline;
		//获取文本框的长宽
		Size area_size = getTextSize(area, font_face, font_scale, thickness, &baseline);
		Size per_size = getTextSize(perimeter, font_face, font_scale, thickness, &baseline);
		Size cd_size = getTextSize(cd, font_face, font_scale, thickness, &baseline);
		Size mp_size = getTextSize(masspot, font_face, font_scale, thickness, &baseline);

		//将文本框居中绘制
		Point origin;
		origin.x = shapes[k].massPt.y - area_size.width / 2 ;
		origin.y = shapes[k].massPt.x + area_size.height / 2;
		putText(display, area, origin, font_face, font_scale, Scalar(0, 0, 0), thickness, 8, 0);

		origin.x = shapes[k].massPt.y - per_size.width / 2;
	
		origin.y = origin.y + per_size.height + 3;
		putText(display, perimeter, origin, font_face, font_scale, Scalar(0, 0, 0), thickness, 8, 0);

		origin.x = shapes[k].massPt.y - cd_size.width / 2;
		origin.y = origin.y + cd_size.height + 3;
		putText(display, cd, origin, font_face, font_scale, Scalar(0, 0, 0), thickness, 8, 0);

		origin.x = shapes[k].massPt.y - mp_size.width / 2;
		origin.y = origin.y + mp_size.height + 3;
		putText(display, masspot, origin, font_face, font_scale, Scalar(0, 0, 0), thickness, 8, 0);

	}

	return display;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
int main()
{
	//读取原图
	char imageName[] = "test7.bmp";
	//char imageName[] = "ik_beijing_p.bmp";
	//Mat srcImg = readbyte("E:\\VS.pr\\resource\\20180620-tianjin-2076x2816x3BIP");
	Mat srcImg = imread(imageName, IMREAD_ANYCOLOR);
	
    // 判断文件是否正常打开 
	if (srcImg.empty()){
		fprintf(stderr, "Can not load image %s\n", imageName);
		waitKey(6000);  
	}
	//二值化
	Mat binImg,grayImg;
	binImg = StateMethord(srcImg);
	//binImg = OSTU(srcImg);
	//cvtColor(srcImg, grayImg, COLOR_BGR2GRAY);//转化为灰度图
	//threshold(grayImg, binImg, 0, 255, THRESH_OTSU);//cv自带二值化算法Ostu

	//连通域标记
	Mat labelImg;
	int shapeSum;
	shapeSum = Cc_TwoPass(binImg, labelImg);
	//shapeSum = Cc_FeedFill(binImg, labelImg);

	//窗口设计
	vector<Shape>shapes;
	Feature(labelImg, shapeSum, shapes);
	cout << shapeSum;
	Mat disImg;
	int thr_Area ;
	long int maxArea ;
	string w1 = "srcImg";
	string w2 = "dstImg";
	string t = "thr_Area";
	namedWindow(w1, WINDOW_AUTOSIZE);//显示原图
	//resizeWindow(w1, 400, 600);
	imshow(w1, srcImg);
	findM(shapes, maxArea, thr_Area);
	createTrackbar(t, w1, &thr_Area, maxArea, on_ThreshChange);
	vector<Shape>Cshapes;
	waitKey(0);
	while (1) { 
		Cshapes.clear();
		Choose(labelImg, shapes, Cshapes, thr_Area);//筛选
		disImg = Display(Cshapes, labelImg);//可视化显示
		namedWindow(w2, WINDOW_AUTOSIZE);
		imshow(w2, disImg);
		if (waitKey(0) == 'o') break;//点击o退出程序
	}
}


