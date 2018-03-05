#include "nn.h"
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
void print(int m, int n, const float*x) {
  //xは一次元配列だがm*nの行列とみなしている
    int i;
    for(int j = 0; j < m; j++){
     for (i = 0; i < n; i++) {
      printf("%.4f ", x[n*j + i]);
     }
     putchar('\n');
    }
}
void fc(int m, int n, const float * x, const float * A, const float * b, float * y) {
    //m*n行列のA,n行1列のx,m行1列のb,m行1列のy　実際は一次元配列
    float z[m-1]; //仮の総和z
    for(int s = 0; s < m; s++ ){z[s] = 0;}
    for(int k = 0; k < m; k++){
      for(int i = 0; i < n; i++){z[k] += A[k*n + i]*x[i];}
      y[k] = z[k] + b[k];
    }
}
void relu(int n, const float * x, float * y) {
    //n行1列のx,y　実際は一次元配列
    for(int i = 0; i < n; i++){
      if(x[i] > 0){y[i] = x[i];}else{y[i] = 0;}
    }
}
void softmax(int n, const float * x, float * y){
    //n行1列のx,y　実際は一次元配列
    float max = 0;
    //zは総和
    float z = 0;
    for(int i = 0; i < n; i++){
      if(x[i] > max){max = x[i];}
    }
    for(int i = 0; i < n; i++){
      z += exp(x[i] - max);
    }
    for(int i = 0; i < n; i++) {
      y[i] = exp(x[i] - max) / z;
    }
}
void softmaxwithloss_bwd(int n, const float * y, unsigned char t, float * dx){
    //nは要素数、yは推論で計算した出力、tは正解の数字、dxは勾配
    for(int i = 0; i < n; i++){
      dx[i] = 0;
      dx[i] = y[i];
      if(i == t){
        dx[i]--;
      }
    }
}
void relu_bwd(int n, const float * x, const float * dy, float * dx){
    //nは要素数、xは前の層からの入力、dyをdxに代入するか0をdxに代入する
    for(int i = 0; i < n; i++){
      if(x[i] > 0){
        dx[i] = dy[i];
      } else {
        dx[i] = 0;
      }
    }
}
void fc_bwd(int m, int n, const float * x, const float * dy, const float * A, float * dA, float * db, float * dx){
    //m*n行列A, dA. n行1列のx,dx m行1列のdy,m行１列のdb
    for(int i = 0;i < n;i++){
      dx[i] = 0;
    }
    for(int i = 0;i< n;i++){
      for(int j = 0;j<m;j++){
        dA[n*j + i] = 0;
        dA[n*j + i] = dy[j] * x[i];
        dx[i] += A[n*j + i] * dy[j];
      }
    }
    for(int i = 0;i< m;i++){
      db[i] = 0;
      db[i] = dy[i];
    }
}
void shuffle(int N, int * x){
    //要素数Nのint配列x
    srand(time(NULL));
    int s = 0;
    for(int k = 0; k < N;k++){
     int j = (int)(rand()*(N+1.0) / (1.0 + RAND_MAX));
     s = x[k];
     x[k] = x[j];
     x[j] = s;
    }
}
void shuffle_f(int N, float * x){
    //要素数Nのfloat 配列x
    srand(time(NULL));
    float s;
    for(int k = 0; k < N;k++){
      int j = (int)(rand()*(N+1.0) / (1.0 + RAND_MAX));
      s = x[k];
      x[k] = x[j];
      x[j] = s;
    }
}
float cross_entropy_error(const float * y, int t){
    //inferenceの出力y, 正解の数字t
    return -log(y[t] + 1e-7);
}
void add(int n, const float * x, float * o){
    //要素数n,oにxを加算する
    for(int i = 0;i < n;i++){
      o[i] += x[i];
    }
}
void scale(int n, float x, float * o){
    //nは要素数。oをx倍する
    for(int i = 0;i < n;i++){
      o[i] *= x;
    }
}
void init(int n, float x, float * o){
    //nは要素数。float 配列oをfloat xで初期化する
    for(int i = 0;i < n;i++){
      o[i] = x;
    }
}
void rand_norm_init(int n,int j, float o[]){
    //整数nは要素の数、iは入力の次元
    srand(time(NULL));
    double sigma = sqrt((double)2/j);
    for(int i = 0;i<n;i++){
     double z = sqrt( -2.0*log(((double)rand() + 1.0)/((double)RAND_MAX + 2.0)) ) * sin(2.0 * M_PI * ((double)rand() + 1.0)/((double)RAND_MAX + 2.0));
     o[i] = sigma*z;
     rand(); rand();
    }
}
void rand_init(int n, float o[]){
  //要素数n, float の配列o
    srand(time(NULL));
    for(int i = 0;i<n;i++){
      o[i] = -1 + (double)(rand()*(1+1.0) / (1.0 + RAND_MAX));
    }
}
int inference6_simple(const float * A1,const float * A2,const float * A3,const float * b1,const float * b2,const float * b3,const float * x, float * y){
    //A1,A2,A3,b1,b2, b3は係数。xは入力,yは出力
    float y1[50];
    float y2[100];
    fc(50, 784, x, A1, b1, y1);
    relu(50, y1, y1);
    fc(100, 50, y1, A2, b2, y2);
    relu(100, y2, y2);
    fc(10, 100, y2, A3, b3, y);
    softmax(10, y, y);
    //最大値を与える番号を計算する
    float max = y[0];
    int answer = 0;
    for(int i = 0;i < 10;i++){
      if(max < y[i]){
        max = y[i];
        answer = i;
      }
    }
    return answer;
}
//relu(int n, const float * x, float * y)

int inference6(const float * A1,const float * A2,const float * A3,const float * b1,const float * b2,const float * b3,const float * x, float * y1, float * y2, float * y3, float * y1r, float *y2r){
//それぞれの係数A1, A2, A3, b1, b2, b3, 入力x,fc1の出力y1  relu1の出力y1r, fc2の出力y2, relu2の出力y2r,fc3の出力y3temp, 最終的な出力はy3
    float y3temp[10];
    fc(50, 784, x, A1, b1, y1);
    relu(50, y1, y1r);
    fc(100, 50, y1r, A2, b2, y2);
    relu(100, y2, y2r);
    fc(10, 100, y2r, A3, b3, y3temp);
    softmax(10, y3temp, y3);
    float max = y3[0];
    int answer = 0;
    for(int i = 0;i < 10;i++){
      if(max < y3[i]){
        max = y3[i];
        answer = i;
      }
    }
    return answer;
}
void backward6(const float * A1, const float * b1, const float * A2, const float * b2, const float * A3, const float * b3, const float * x, unsigned char t, float * dA1, float * db1, float * dA2, float * db2, float * dA3, float * db3){
//それぞれの係数A1, b1, A2, b2, A3, b3, 入力x, 正解t, それぞれの係数の微分dA1, dA2, dA3, db1, db2, db3
    float y1[50];
    float y1r[50];
    float y2[100];
    float y2r[100];
    float y3[10];
    float dx1[10];
    float dx2[100];
    float dx2r[100];
    float dx3[50];
    float dx3r[50];
    float dx4[784];
    inference6(A1, A2, A3, b1, b2, b3, x, y1, y2, y3, y1r, y2r);
    softmaxwithloss_bwd(10, y3, t, dx1);
    fc_bwd(10, 100, y2r, dx1, A3, dA3, db3, dx2);
    relu_bwd(100, y2, dx2, dx2r);
    fc_bwd(100, 50, y1r, dx2r, A2, dA2, db2, dx3);
    relu_bwd(50, y1, dx3, dx3r);
    fc_bwd(50, 784, x, dx3r, A1, dA1, db1, dx4);
}
void save(const char * filename, int m, int n, const float * A, const float * b){
  //保存するファイル名がfilename, m*n行列A, m行ベクトルb
    FILE * fp;
    if((fp = fopen(filename, "w")) == NULL){printf("読み込みエラー");}
    fwrite(A, sizeof(float), m*n, fp);
    fwrite(b, sizeof(float), m, fp);
    fclose(fp);
}
void load(const char * filename, int m, int n, float * A, float * b){
  //読み込むファイル名がfilename, m*n行列A, m行ベクトルb
    FILE * fp;
    if((fp = fopen(filename, "r")) == NULL){printf("読み込みエラー");}
    fread(A, sizeof(float), m*n, fp);
    fread(b, sizeof(float), m, fp);
    fclose(fp);
}
/*
inference6(A1,A2,A3,b1,b2,b3,x,y1,y2,y3,y1r,y2r)それぞれの入力を記憶
softmaxwithloss_bwd(int n, const float * y, unsigned char t, float * dx) 要素数nで
fc_bwd(int m, int n, const float * x, const float * dy, const float * A, float * dA, float * db, float * dx)
relu_bwd(int n, const float * x, const float * dy, float * dx)
*/
int main(int argc, char * argv[]) {
    float * train_x = NULL;
    unsigned char * train_y = NULL;
    int train_count = -1;
    float * test_x = NULL;
    unsigned char * test_y = NULL;
    int test_count = -1;
    int width = -1;
    int height = -1;
    load_mnist(&train_x, &train_y, &train_count, &test_x, &test_y, &test_count, &width, &height);
    //必要なもののメモリを確保
    //A1は50×784 A2は100×50 A3は10×100 b1は50, b2は100, b3は10
    float * A1 = malloc(sizeof(float)*50*784);
    float * A2 = malloc(sizeof(float)*100*50);
    float * A3 = malloc(sizeof(float)*10*100);
    float * b1 = malloc(sizeof(float)*50);
    float * b2 = malloc(sizeof(float)*100);
    float * b3 = malloc(sizeof(float)*10);
    float * dA1 = malloc(sizeof(float)*50*784);
    float * dA2 = malloc(sizeof(float)*100*50);
    float * dA3 = malloc(sizeof(float)*10*100);
    float * db1 = malloc(sizeof(float)*50);
    float * db2 = malloc(sizeof(float)*100);
    float * db3 = malloc(sizeof(float)*10);
    float * avedA1 = malloc(sizeof(float)*50*784);
    float * avedA2 = malloc(sizeof(float)*100*50);
    float * avedA3 = malloc(sizeof(float)*10*100);
    float * avedb1 = malloc(sizeof(float)*50);
    float * avedb2 = malloc(sizeof(float)*100);
    float * avedb3 = malloc(sizeof(float)*10);
    int * index = malloc(sizeof(int)*train_count);
    double learningratio = 0;
    int minibatchsize = 100;
    int handan;
    int epoch;
    double goal;
    printf("Epoch times or Recognition accuracy?\nEpoch : 1\nRecognition Accuracy : 2\n"); scanf("%d", &handan);
    if(handan == 1){
       printf("Input epoch :"); scanf("%d", &epoch);
       goal = 100.0;
       printf("Wait a minute...\n");
    }else if(handan == 2){
       printf("Input Goal Accuracy :"); scanf("%lf", &goal);
       epoch = 100000000;
       printf("Wait a minute...\n");
    }else{
       printf("Invalid Input!");
       exit(0);
    }
       //それぞれを乱数で初期化
    rand_norm_init(50*784, 784, A1);
    rand_norm_init(50,784, b1);
    rand_norm_init(100*50, 50, A2);
    rand_norm_init(100, 50, b2);
    rand_norm_init(10*100, 10, A3);
    rand_norm_init(10, 10, b3);
    //index配列を初期化する
    for(int i = 0;i < 60000;i++){index[i] = i;}
    for(int i = 0;i < epoch;i++){//以下をエポック回数回行う
         shuffle(60000, index);//配列indexをシャッフル
         for(int j = 0;j < 60000/minibatchsize;j++){ //ミニパッチ学習スタート600回
           //(1)平均勾配を初期化
           init(50*784, 0, avedA1);
           init(100*50, 0, avedA2);
           init(10*100, 0, avedA3);
           init(50, 0, avedb1);
           init(100, 0, avedb2);
           init(10, 0, avedb3);
           //(2)逆伝搬を行う(100回)
           //indexの先頭100ずつをもってくる
           for(int k = 0;k < minibatchsize;k++){
             backward6(A1, b1, A2, b2, A3, b3, train_x + 784 * index[k + 100*j], train_y[index[k + 100*j]], dA1, db1, dA2, db2, dA3, db3);
             add(50*784, dA1, avedA1);
             add(100*50, dA2, avedA2);
             add(10*100, dA3, avedA3);
             add(50, db1, avedb1);
             add(100, db2, avedb2);
             add(10, db3, avedb3);
           }
           //(3)平均勾配をnで割り、学習率にあわせて変更
           if(i == 0){learningratio = 0.15;}
           if(i == 1){learningratio = 0.08;}
           if(i == 2){learningratio = 0.06;}
           if(i == 3){learningratio = 0.05;}
           if(i == 4){learningratio = 0.04;}
           if(i == 5){learningratio = 0.03;}
           if(i == 6){learningratio = 0.01;}
           

           scale(50*784, -learningratio*0.01, avedA1);
           scale(100*50, -learningratio*0.01, avedA2);
           scale(10*100, -learningratio*0.01, avedA3);
           scale(50, -learningratio*0.01, avedb1);
           scale(100, -learningratio*0.01, avedb2);
           scale(10, -learningratio*0.01, avedb3);

           //A,bを更新
           add(50*784, avedA1, A1);
           add(100*50, avedA2, A2);
           add(10*100, avedA3, A3);
           add(50, avedb1, b1);
           add(100, avedb2, b2);
           add(10, avedb3, b3);
         }
         //正解率、損失を求める
         int answer = 0;
         int sum = 0;
         float lossum = 0;
         float y[10];
         for(int k = 0;k < test_count; k++){
           answer = inference6_simple(A1, A2, A3, b1, b2, b3, test_x + 784 * k, y);
           if(answer == test_y[k]){sum++;}
           else {
        //     printf("計算は%d,正解は %d\n",answer,test_y[k]);
           }
           lossum += cross_entropy_error(y, test_y[k]);
         }
         printf("%d エポック目完了\n精度は%.2f%%\n", i + 1, sum * 100.0 / test_count);
         printf("損失は%.4f\n", (double)lossum / test_count);
         if(sum*100.0 / test_count > 96.0){learningratio = 0.00004;} //精度が高くなれば学習率を下げる
         if(sum*100.0 / test_count > goal){
           printf("目標精度を達成しました\n");
           break;
         }
      } //指定回数回のエポック終了
      //保存するかを尋ねる
      int flag;
      printf("Do you want to save ?\nyes:1\nno:0\n"); scanf("%d", &flag);
      if(flag == 1){//1ならセーブ開始
        save(argv[1], 50, 784, A1, b1);
        save(argv[2], 100, 50, A2, b2);
        save(argv[3], 10, 100, A3, b3);
        printf("Complete!");
      }//セーブ終了
    return 0;
}
