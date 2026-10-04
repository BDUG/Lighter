/* Test-only TFLite C ABI shim. No NPU hardware is simulated or claimed. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
typedef struct {int type; int rank; int dims[3]; size_t bytes; unsigned char data[128];} Tensor;
typedef struct {int delegate;} Options;
typedef struct {Tensor input[3],output; int delegate; int fail_alloc; int fail_invoke;} Interpreter;
static int delegates=0,interpreters=0,models=0;
void *TfLiteModelCreateFromFile(const char *path){models++;return strdup(path);}
void TfLiteModelDelete(void *p){if(p){models--;free(p);}}
void *TfLiteInterpreterOptionsCreate(void){return calloc(1,sizeof(Options));}
void TfLiteInterpreterOptionsDelete(void *p){free(p);}
void TfLiteInterpreterOptionsSetNumThreads(void *p,int n){(void)p;(void)n;}
void TfLiteInterpreterOptionsAddDelegate(void *p,void *d){((Options*)p)->delegate=d!=NULL;}
void *TfLiteInterpreterCreate(void *model,void *options){
 Interpreter *p=calloc(1,sizeof(*p)); interpreters++;
 p->delegate=((Options*)options)->delegate;
 p->fail_alloc=strstr((char*)model,"allocate-fail")!=NULL;
 p->fail_invoke=strstr((char*)model,"invoke-fail")!=NULL;
 for(int i=0;i<3;i++){p->input[i].type=i==2?4:2;p->input[i].rank=2;p->input[i].dims[0]=1;p->input[i].dims[1]=4;p->input[i].bytes=i==2?32:16;}
 p->output.type=1;p->output.rank=3;p->output.dims[0]=1;p->output.dims[1]=4;p->output.dims[2]=4;p->output.bytes=64;
 if(strstr((char*)model,"bad-shape"))p->input[0].dims[0]=2;
 return p;
}
void TfLiteInterpreterDelete(void *p){if(p){interpreters--;free(p);}}
int TfLiteInterpreterAllocateTensors(void *p){Interpreter *i=p;return i->delegate&&i->fail_alloc?1:0;}
int TfLiteInterpreterInvoke(void *p){
 Interpreter *i=p;if(i->fail_invoke)return 1;
 int32_t *ids=(int32_t*)i->input[0].data,*mask=(int32_t*)i->input[1].data;
 int64_t *positions=(int64_t*)i->input[2].data;
 float *out=(float*)i->output.data;
 for(int row=0;row<4;row++)for(int v=0;v<4;v++)out[row*4+v]=(float)(ids[row]*10+positions[row]+mask[row]+v);
 return 0;
}
int TfLiteInterpreterGetInputTensorCount(void *p){(void)p;return 3;}
int TfLiteInterpreterGetOutputTensorCount(void *p){(void)p;return 1;}
void *TfLiteInterpreterGetInputTensor(void *p,int n){return n>=0&&n<3?&((Interpreter*)p)->input[n]:NULL;}
const void *TfLiteInterpreterGetOutputTensor(void *p,int n){return n==0?&((Interpreter*)p)->output:NULL;}
int TfLiteTensorType(const void *p){return ((const Tensor*)p)->type;}
int TfLiteTensorNumDims(const void *p){return ((const Tensor*)p)->rank;}
int TfLiteTensorDim(const void *p,int n){return ((const Tensor*)p)->dims[n];}
size_t TfLiteTensorByteSize(const void *p){return ((const Tensor*)p)->bytes;}
int TfLiteTensorCopyFromBuffer(void *p,const void *src,size_t n){Tensor *t=p;if(n!=t->bytes)return 1;memcpy(t->data,src,n);return 0;}
int TfLiteTensorCopyToBuffer(const void *p,void *dst,size_t n){const Tensor *t=p;if(n!=t->bytes)return 1;memcpy(dst,t->data,n);return 0;}
void *tflite_plugin_create_delegate(const char *const *keys,const char *const *values,int n,void (*report)(const char*)){(void)report;for(int j=0;j<n;j++)if(!strcmp(keys[j],"fail")&&!strcmp(values[j],"true"))return NULL;delegates++;return malloc(1);}
void tflite_plugin_destroy_delegate(void *p){delegates--;free(p);}
int lighter_test_live_handles(void){return delegates+interpreters+models;}
