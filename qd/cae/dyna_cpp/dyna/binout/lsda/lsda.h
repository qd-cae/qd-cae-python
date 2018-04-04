#ifndef LSDA_H
#define LSDA_H

#ifdef __cplusplus
extern "C" {
#endif

#undef EXTERN
#ifdef __BUILD_LSDA__
#define EXTERN
#else
#define EXTERN extern
#endif

typedef size_t LSDA_Offset;
typedef size_t LSDA_Length;
typedef unsigned char LSDA_Command;
typedef unsigned char LSDA_TypeID;

typedef struct _dbdir
{
  void* btree;
  void* daf;
  int cont;
} LSDADir;

EXTERN int
lsda_test(char* filen);
EXTERN int
lsda_open_many(char** filen, int num);
EXTERN int
lsda_open(char* filen, int mode);
EXTERN LSDA_Length
lsda_write(int handle, int type_id, char* name, LSDA_Length length, void* data);
EXTERN int
lsda_cd(int handle, char* path);
EXTERN LSDA_Length
lsda_fsize(int handle);
EXTERN int
lsda_nextfile(int handle);
EXTERN int
lsda_setmaxsize(int handle, size_t size);
EXTERN int
lsda_close(int handle);
EXTERN int
lsda_flush(int handle);
EXTERN int
lsda_sync(int handle);
EXTERN LSDA_Length
lsda_read(int handle,
          int type_id,
          char* name,
          LSDA_Length offset,
          LSDA_Length number,
          void* data);
EXTERN LSDA_Length
lsda_lread(int handle,
           int type_id,
           char* name,
           LSDA_Length offset,
           LSDA_Length number,
           void* data);
EXTERN LSDADir*
lsda_opendir(int handle, char* path);
EXTERN char*
lsda_getpwd(int handle);
EXTERN void
lsda_query(int handle, char* name, int* type_id, LSDA_Length* length);
EXTERN void
lsda_lquery(int handle, char* name, int* type_id, LSDA_Length* length);
EXTERN void
lsda_queryvar(int handle,
              char* name,
              int* type_id,
              LSDA_Length* length,
              int* filenum);
EXTERN void
lsda_readdir(LSDADir* dir,
             char* name,
             int* type_id,
             LSDA_Length* length,
             int* filenum);
EXTERN size_t
lsda_totalmemory(int handle);
EXTERN void
lsda_closedir(LSDADir* dir);
EXTERN int
lsda_util_countdir(int fhandle, char* dirname, int* ndir);
EXTERN int
lsda_util_id2size(int type_id);
EXTERN char*
lsda_getname(int handle);
EXTERN char*
lsda_getbasename(int handle);
EXTERN int
lsda_gethandle(char* name);
EXTERN int
lsda_util_db2sg(int type_id);
EXTERN void
free_all_lsda(void);

// qd additions
EXTERN char**
lsda_get_children_names(int handle,
                        char* name,
                        int follow,
                        LSDA_Length* length);

#define LSDA_READONLY 0
#define LSDA_WRITEONLY 1
#define LSDA_READWRITE 2
#define LSDA_WRITEREAD 3
#define LSDA_APPEND 4
#define LSDA_SUCCESS 0

EXTERN int*
_lsda_errno();
#define lsda_errno (*_lsda_errno())

/*
  Defined constants for the available data types
  The constants here must (I think) match the order
  of creation in lsda_createbasictypes()
  Types > 11 get aliased to lower numbers, so in fact
  only types 0 (directory) through 11 actually occur
  in an LSDA file
*/

#define LSDA_I1 1
#define LSDA_I2 2
#define LSDA_I4 3
#define LSDA_I8 4
#define LSDA_U1 5
#define LSDA_U2 6
#define LSDA_U4 7
#define LSDA_U8 8
#define LSDA_R4 9
#define LSDA_R8 10
#define LSDA_LINK 11
#define LSDA_INT 12
#define LSDA_SHORT 13
#define LSDA_LONG 14
#define LSDA_UINT 15
#define LSDA_USHORT 16
#define LSDA_ULONG 17
#define LSDA_FLOAT 18
#define LSDA_DOUBLE 19
#define LSDA_INTEGER 20
#define LSDA_REAL 21
#define LSDA_DP 22

#ifdef __cplusplus
}
#endif

#endif
