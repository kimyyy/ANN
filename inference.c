#include "nn.h"
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
void print(int m, int n, const float*x) {
    int i;
    for(int j = 0; j < m; j++){
     for (i = 0; i < n; i++) {
      printf("%.4f ", x[n*j + i]);
     }
     putchar('\n');
    }
}
void fc(int m, int n, const float * x, const float * A, const float * b, float * y) {
    float z[m-1];
    for(int s = 0; s < m; s++ ){z[s] = 0;}
    for(int k = 0; k < m; k++){
      for(int i = 0; i < n; i++){z[k] += A[k*n + i]*x[i];}
      y[k] = z[k] + b[k];
    }
}
void relu(int n, const float * x, float * y) {
    for(int i = 0; i < n; i++){
      if(x[i] > 0){y[i] = x[i];}else{y[i] = 0;}
    }
}
void softmax(int n, const float * x, float * y){
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
int inference3(const float * A, const float * b, const float * x, float * y, float * y1) {
    int m = 0;
    fc(10,784, x, A, b, y1);
    relu(10, y1, y);
    softmax(10, y, y);
    float max = y[0];
    for(int i = 0; i < 10; i++) {
      if(y[i] > max){
       max = y[i];
       m = i;
      }
    }
    return m;
}
void softmaxwithloss_bwd(int n, const float * y, unsigned char t, float * dx){
    for(int i = 0; i < n; i++){
      dx[i] = 0;
      dx[i] = y[i];
      if(i == t){
        dx[i]--;
      }
    }
}
void relu_bwd(int n, const float * x, const float * dy, float * dx){
    for(int i = 0; i < n; i++){
      if(x[i] > 0){
        dx[i] = dy[i];
      } else {
        dx[i] = 0;
      }
    }
}
void fc_bwd(int m, int n, const float * x, const float * dy, const float * A, float * dA, float * db, float * dx){
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
void backward3(const float * A, const float * b, const float * x, unsigned char t, float * y, float * dA, float * db){
    float * dy = malloc(sizeof(float)*10);
    float * y1 = malloc(sizeof(float)*10);
    inference3(A, b, x, y, y1);
    float * dx = malloc(sizeof(float)*10);
    softmaxwithloss_bwd(10, y, t, dx);
    relu_bwd(10, y1, dx, dy);
    fc_bwd(10, 784, x, dy, A, dA, db, dx);
}
void shuffle(int N, int * x){
    srand(time(NULL));
    int s = 0;
    for(int k = 0; k < N;k++){
     int j = (int)(rand()*(N*1.0) / (1.0 + RAND_MAX));
     s = x[k];
     x[k] = x[j];
     x[j] = s;
    }
}
void shuffle_f(int N, float * x){
    srand(time(NULL));
    float s;
    for(int k = 0; k < N;k++){
      int j = (int)(rand()*(N*1.0) / (1.0 + RAND_MAX));
      s = x[k];
      x[k] = x[j];
      x[j] = s;
    }
}
float cross_entropy_error(const float * y, int t){
    return -log(y[t] + 1e-7);
}
void add(int n, const float * x, float * o){
    for(int i = 0;i < n;i++){
      o[i] += x[i];
    }
}
void scale(int n, float x, float * o){
    for(int i = 0;i < n;i++){
      o[i] *= x;
    }
}
void init(int n, float x, float * o){
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
    srand(time(NULL));
    for(int i = 0;i<n;i++){
      o[i] = -1 + (double)(rand()*(1+1.0) / (1.0 + RAND_MAX));
    }
}
int inference6_simple(const float * A1,const float * A2,const float * A3,const float * b1,const float * b2,const float * b3,const float * x, float * y){
    float y1[50];
    float y2[100];
    fc(50, 784, x, A1, b1, y1);
    relu(50, y1, y1);
    fc(100, 50, y1, A2, b2, y2);
    relu(100, y2, y2);
    fc(10, 100, y2, A3, b3, y);
    softmax(10, y, y);
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
    FILE * fp;
    if((fp = fopen(filename, "w")) == NULL){printf("読み込みエラー");}
    fwrite(A, sizeof(float), m*n, fp);
    fwrite(b, sizeof(float), m, fp);
    fclose(fp);
}
void load(const char * filename, int m, int n, float * A, float * b){
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
    //必要なもののメモリを確保
    //A1は50×784 A2は100×50 A3は10×100 b1は50, b2は100, b3は10
    float * A1 = malloc(sizeof(float)*50*784);
    float * A2 = malloc(sizeof(float)*100*50);
    float * A3 = malloc(sizeof(float)*10*100);
    float * b1 = malloc(sizeof(float)*50);
    float * b2 = malloc(sizeof(float)*100);
    float * b3 = malloc(sizeof(float)*10);
    float * y = malloc(sizeof(float)*10);
    float * x = load_mnist_bmp(argv[4]);//画像データを読み込み、配列xとする
    load(argv[1], 50, 784, A1, b1);//係数データを読み込む
    load(argv[2], 100, 50, A2, b2);
    load(argv[3], 10, 100, A3, b3);
    int answer = inference6_simple(A1, A2, A3, b1, b2, b3,x, y);//推論を行う
    printf("この数字は%dです\n", answer);
    return 0;
}
