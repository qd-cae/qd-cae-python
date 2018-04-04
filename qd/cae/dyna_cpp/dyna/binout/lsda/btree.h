#ifdef __cplusplus
extern "C"{
#endif
typedef int BT_CompFunc(void *,void *);
void *BT_new(BT_CompFunc *compare);
#define BT_store(a,b) BT_lookup((a),(b),1)
void BT_delete(void *bt, void *key);
void *BT_lookup(void *bt, void *data, int store);
int BT_numentries(void *bt);
void *BT_enumerate(void *bt, int *cont);
void BT_flush(void *bt);
void BT_free(void *bt);
void **BT_list(void *bt);
#ifdef __cplusplus
}
#endif
